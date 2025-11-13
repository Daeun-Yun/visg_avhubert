# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys,logging
import contextlib
import tempfile
from argparse import Namespace
from typing import Any, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from fairseq import checkpoint_utils, tasks, utils
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import BaseFairseqModel, FairseqEncoder, FairseqEncoderDecoderModel, register_model
from fairseq.models.hubert.hubert import MASKING_DISTRIBUTION_CHOICES
from fairseq.tasks import FairseqTask
from omegaconf import II, MISSING

DBG=True if len(sys.argv) == 1 else False

if DBG:
    from hubert import AVHubertModel
    from decoder import TransformerDecoder
else:
    from .hubert import AVHubertModel
    from .decoder import TransformerDecoder

logger = logging.getLogger(__name__)
class WeightMix(nn.Module):
    def __init__(self, num_layers: int, initial_scalar_parameters: List[float] = None, trainable: bool = True):
        super().__init__()
        self.num_layers = num_layers
        if initial_scalar_parameters is None:
            self.scalar_parameters = nn.Parameter(torch.zeros(num_layers), requires_grad=trainable)
        else:
            assert len(initial_scalar_parameters) == num_layers
            self.scalar_parameters = nn.Parameter(torch.tensor(initial_scalar_parameters, dtype=torch.float), requires_grad=trainable)
        self.gamma = nn.Parameter(torch.tensor([1.0]), requires_grad=trainable)
        self.register_buffer("normalized_weights", torch.zeros(num_layers)) # This buffer is used for logging
    
    def forward(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            tensors: A list of tensors, each of shape (T, B, C) or (B, T, C).
                     The tensors must have the same shape.
        
        Returns:
            A single tensor of the same shape as the input tensors.
        """
        assert len(tensors) == self.num_layers, f"Expected {self.num_layers} tensors, but got {len(tensors)}"
        
        # Normalize weights using softmax
        norm_weights = torch.nn.functional.softmax(self.scalar_parameters, dim=0)
        with torch.no_grad():
            self.normalized_weights.copy_(norm_weights.detach())
        # Stack tensors and compute the weighted sum
        # (T, B, C) -> (N_layers, T, B, C)
        stacked_tensors = torch.stack(tensors)
        
        # (N_layers, T, B, C) * (N_layers, 1, 1, 1) -> (T, B, C)
        # Weighted sum for the features from each layer
        weighted_sum = (norm_weights.view(-1, *([1] * (stacked_tensors.dim() - 1))) * stacked_tensors).sum(dim=0)

        return self.gamma * weighted_sum


@dataclass
class AVHubertAsrConfig(FairseqDataclass):
    w2v_path: str = field(
        default=MISSING, metadata={"help": "path to hubert model"}
    )
    no_pretrained_weights: bool = field(
        default=False,
        metadata={"help": "if true, does not load pretrained weights"},
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    final_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout after transformer and before final projection"
        },
    )
    dropout: float = field(
        default=0.0,
        metadata={"help": "dropout probability inside hubert model"},
    )
    attention_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability for attention weights "
            "inside hubert model"
        },
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN "
            "inside hubert model"
        },
    )

    # masking
    apply_mask: bool = field(
        default=False, metadata={"help": "apply masking during fine-tuning"}
    )
    mask_length: int = field(
        default=10, metadata={"help": "repeat the mask indices multiple times"}
    )
    mask_prob: float = field(
        default=0.5,
        metadata={
            "help": "probability of replacing a token with mask "
            "(normalized by length)"
        },
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose masks"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indices"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )

    # channel masking
    mask_channel_length: int = field(
        default=10,
        metadata={"help": "length of the mask for features (channels)"},
    )
    mask_channel_prob: float = field(
        default=0.0,
        metadata={"help": "probability of replacing a feature with 0"},
    )
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indices"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False,
        metadata={"help": "whether to allow channel masks to overlap"},
    )
    freeze_finetune_updates: int = field(
        default=0,
        metadata={"help": "dont finetune hubert for this many updates"},
    )
    feature_grad_mult: float = field(
        default=0.0,
        metadata={"help": "reset feature grad mult in hubert to this"},
    )
    layerdrop: float = field(
        default=0.0,
        metadata={"help": "probability of dropping a layer in hubert"},
    )
    normalize: bool = II("task.normalize")
    data: str = II("task.data")

    # this holds the loaded hubert args
    w2v_args: Any = None


@dataclass
class AVHubertCtcConfig(AVHubertAsrConfig):
    pass


@register_model("av_hubert_ctc", dataclass=AVHubertCtcConfig)
class AVHubertCtc(BaseFairseqModel):
    def __init__(self, cfg: AVHubertCtcConfig, w2v_encoder: BaseFairseqModel):
        super().__init__()
        self.cfg = cfg
        self.w2v_encoder = w2v_encoder

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, cfg: AVHubertCtcConfig, task: FairseqTask):
        """Build a new model instance."""
        w2v_encoder = HubertEncoder(cfg, task.target_dictionary)
        
        return cls(cfg, w2v_encoder)

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = net_output["encoder_out"]
        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

    def get_logits(self, net_output):
        logits = net_output["encoder_out"]
        padding = net_output["encoder_padding_mask"]
        if padding is not None and padding.any():
            padding = padding.T
            logits[padding][..., 0] = 0
            logits[padding][..., 1:] = float("-inf")

        return logits

    def forward(self, **kwargs):
        x = self.w2v_encoder(**kwargs)
        return x


@dataclass
class AVHubertSeq2SeqConfig(AVHubertAsrConfig):
    viseme_classification: bool = field(
        default=False,
        metadata={
            "help": "if set, add a linear layer after the encoder for viseme classification"
        },
    )
    use_ctc: bool = field(
        default=False,
        metadata={
            "help": "if set, use CTC loss for the viseme classification task"
        },)
    weight_mix: bool = field(
        default=False,
        metadata={
            "help": "if set, use a weighted sum of per layer features for the viseme classification task"
        },
    )
    decoder_embed_dim: int = field(
        default=768, metadata={"help": "decoder embedding dimension"}
    )
    decoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "decoder embedding dimension for FFN"}
    )
    decoder_layers: int = field(
        default=6, metadata={"help": "num of decoder layers"}
    )
    decoder_layerdrop: float = field(
        default=0.0, metadata={"help": "decoder layerdrop chance"}
    )
    decoder_attention_heads: int = field(
        default=4, metadata={"help": "num decoder attention heads"}
    )
    decoder_learned_pos: bool = field(
        default=False,
        metadata={"help": "use learned positional embeddings in the decoder"},
    )
    decoder_normalize_before: bool = field(
        default=False,
        metadata={"help": "apply layernorm before each decoder block"},
    )
    no_token_positional_embeddings: bool = field(
        default=False,
        metadata={
            "help": "if set, disables positional embeddings "
            "(outside self attention)"
        },
    )
    decoder_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability in the decoder"}
    )
    decoder_attention_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability for attention weights "
            "inside the decoder"
        },
    )
    decoder_activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN "
            "inside the decoder"
        },
    )
    max_target_positions: int = field(
        default=2048, metadata={"help": "max target positions"}
    )
    share_decoder_input_output_embed: bool = field(
        default=False,
        metadata={"help": "share decoder input and output embeddings"},
    )
    no_scale_embedding: bool = field(default=True, metadata={'help': 'scale embedding'})

class HubertEncoder(FairseqEncoder):
    def __init__(self, cfg: AVHubertAsrConfig, tgt_dict=None):
        self.apply_mask = cfg.apply_mask

        arg_overrides = {
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
        }

        if cfg.w2v_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(
                cfg.w2v_path, arg_overrides
            )
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            cfg.w2v_args = w2v_args
        else:
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, Namespace):
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(
                    w2v_args
                )

        assert cfg.normalize == w2v_args.task.normalize, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for "
            "both pre-training and here"
        )

        w2v_args.task.data = cfg.data

        task = tasks.setup_task(w2v_args.task)
        model = task.build_model(w2v_args.model)

        if state is not None and not cfg.no_pretrained_weights:
            # set strict=False because we omit some modules
            model.load_state_dict(state["model"], strict=False)

        model.remove_pretraining_modules()

        super().__init__(task.source_dictionary)

        d = model.encoder.embedding_dim

        self.w2v_model = model

        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0

        if tgt_dict is not None:
            self.proj = Linear(d, len(tgt_dict))
        elif getattr(cfg, "decoder_embed_dim", d) != d:
            self.proj = Linear(d, cfg.decoder_embed_dim)
        else:
            self.proj = None

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def forward(self, source, padding_mask, tbc=True, **kwargs):

        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
            "mask": self.apply_mask and self.training,
        }
        ft = self.freeze_finetune_updates <= self.num_updates

        with torch.no_grad() if not ft else contextlib.ExitStack():
            x, padding_mask = self.w2v_model.extract_finetune(**w2v_args)

            if tbc:
                # B x T x C -> T x B x C
                x = x.transpose(0, 1)

        x = self.final_dropout(x)

        if self.proj:
            x = self.proj(x)

        return {
            "encoder_out": x,  # T x B x C
            "encoder_padding_mask": padding_mask,  # B x T
            "padding_mask": padding_mask,
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = encoder_out[
                "encoder_out"
            ].index_select(1, new_order)
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return None

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict


class HubertEncoderWrapper(FairseqEncoder):
    def __init__(self, w2v_model):
        super().__init__(None)
        self.w2v_model = w2v_model

    def forward(self, source, padding_mask, **kwargs):
        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
        }

        # x, padding_mask, _ = self.w2v_model.extract_finetune(**w2v_args)
        x, padding_mask, hidden_states = self.w2v_model.extract_finetune(**w2v_args)
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        hidden_states = [hidden_states[i].transpose(0, 1) for i in range(len(hidden_states))]
        layer_num = 0
        layer_feats = hidden_states[layer_num] if layer_num is not None else None
        return {
            "encoder_out": x,  # T x B x C
            "encoder_padding_mask": padding_mask,  # B x T
            "padding_mask": padding_mask,
            "layer_feats": layer_feats,  # T x B x C
        }
    
    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = encoder_out[
                "encoder_out"
            ].index_select(1, new_order)
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ].index_select(0, new_order)
        if encoder_out["padding_mask"] is not None:
            encoder_out["padding_mask"] = encoder_out[
                "padding_mask"
            ].index_select(0, new_order)
        return encoder_out

@register_model("av_hubert_seq2seq", dataclass=AVHubertSeq2SeqConfig)
class AVHubertSeq2Seq(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder, tgt_dict, cfg):
        super().__init__(encoder, decoder)
        self.cfg = cfg
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        # In case weights and sigmas are learnable parameters
        self.asr_weight = nn.Parameter(torch.tensor(0.8, dtype=torch.float32))  # weight for ASR loss
        self.viseme_weight = nn.Parameter(torch.tensor(0.2, dtype=torch.float32)) # weight for viseme classification loss
        self.s_asr = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))  # σ_asr
        self.s_vsm = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))  # σ_viseme
        if self.cfg.weight_mix:
            num_encoder_layers = 12
            self.layer_weigher = WeightMix(num_layers=num_encoder_layers)
        self.encoder_output_dim = encoder.w2v_model.encoder.embedding_dim # Get encoder dimension
        if cfg.viseme_classification: # For viseme classification task
            if cfg.use_ctc: # If CTC is used, we have 14 visemes + blank token
                self.viseme_head = nn.Sequential(
                    nn.Dropout(p=0.3),
                    Linear(self.encoder_output_dim, 15)
                    )
            else: # Otherwise, we have the normal 14 visemes of Lee's mapping (including the silence viseme)
                self.viseme_head = nn.Sequential(
                    nn.Dropout(p=0.3),
                    Linear(self.encoder_output_dim, 14)
                    )
        else:
            self.viseme_head = None

    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""

        arg_overrides = {
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
        }

        if cfg.w2v_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(
                cfg.w2v_path, arg_overrides
            )
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            cfg.w2v_args = w2v_args
        else:
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, Namespace):
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(
                    w2v_args
                )

        assert cfg.normalize == w2v_args.task.normalize, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for "
            "both pre-training and here"
        )

        w2v_args.task.data = cfg.data

        task_pretrain = tasks.setup_task(w2v_args.task)
        if state is not None:
            task_pretrain.load_state_dict(state['task_state'])

        encoder_ = task_pretrain.build_model(w2v_args.model)

        encoder = HubertEncoderWrapper(encoder_)
        if state is not None and not cfg.no_pretrained_weights:
            # set strict=False because we omit some modules
            del state['model']['mask_emb']
            encoder.w2v_model.load_state_dict(state["model"], strict=False)

        encoder.w2v_model.remove_pretraining_modules()

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx=padding_idx)
            return emb

        decoder_embed_tokens = build_embedding(tgt_dict, cfg.decoder_embed_dim)
        decoder = TransformerDecoder(cfg, tgt_dict, decoder_embed_tokens)
        
        return AVHubertSeq2Seq(encoder, decoder, tgt_dict, cfg)
    
    def forward(self, **kwargs):
        torch.autograd.set_detect_anomaly(True)
        ft = self.freeze_finetune_updates <= self.num_updates
        with torch.no_grad() if not ft else contextlib.ExitStack():
            output = self.encoder(**kwargs)
        if self.cfg.weight_mix is True:
            mixed_enc_out = self.layer_weigher(output['layer_feats'])
            output_mixed = {
            'encoder_out': mixed_enc_out,
            'encoder_padding_mask': output['encoder_padding_mask'],
            'padding_mask': output['padding_mask'],
            }
            decoder_out = self.decoder(prev_output_tokens=kwargs['prev_output_tokens'], encoder_out=output_mixed)
        elif output['layer_feats'] is not None:
            output_mixed = {
            'encoder_out': output['layer_feats'],
            'encoder_padding_mask': output['encoder_padding_mask'],
            'padding_mask': output['padding_mask'],
            }
            decoder_out = self.decoder(prev_output_tokens=kwargs['prev_output_tokens'], encoder_out=output_mixed)
        else:
            decoder_out = self.decoder(prev_output_tokens=kwargs['prev_output_tokens'], encoder_out=output)
        
        if self.cfg.viseme_classification is True:
            if self.cfg.weight_mix is True:
                hidden_states = output_mixed['encoder_out'].transpose(0, 1)  # T x B x C -> B x T x C
                # weighted_states = self.layer_weigher(hidden_states).transpose(0, 1)
                # viseme_feats = output['layer_feats']  # T x B x C
                # Convert to B x T x C for classification
                # viseme_feats = viseme_feats.transpose(0, 1)
                # Perform viseme classification
                viseme_logits = self.viseme_head(hidden_states)  # B x T x 14
            else:
                viseme_feats = output['encoder_out']  # T x B x C
                # Convert to B x T x C for classification
                viseme_feats = viseme_feats.transpose(0, 1)
                # Perform viseme classification
                viseme_logits = self.viseme_head(viseme_feats)  # B x T x 14
                # return decoder_out
            return {"decoder_out": decoder_out, "viseme_out": viseme_logits}
        else:
            return {"decoder_out": decoder_out}

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m
