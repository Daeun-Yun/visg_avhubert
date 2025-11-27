from dataclasses import dataclass, field
from typing import Optional
import math
import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion

@dataclass
class MultiTaskCriterionConfig(FairseqDataclass):
    sentence_avg: bool = field(
        default=True,
        metadata={"help": "normalize gradients by sentence length"},
    )
    viseme_weight: float = field(
        default=0.2,
        metadata={"help": "weight for viseme classification loss"},
    )
    label_smoothing: float = field(
        default=0.1,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    ignore_viseme_padding: bool = field(
        default=True,
        metadata={"help": "ignore padding tokens for viseme loss calculation"},
    )
    # This is an integer that specifies the training update step after which
    # the viseme loss starts being applied. This allows for a task curriculum learning training approach,
    # where the model first learns the AVSR task before incorporating the viseme classification task.
    viseme_start_update: int = field(
        default=0,
        metadata={"help": "training update step after which viseme loss starts being applied"},
    )
    # This is a boolean flag to indicate whether to gradually introduce the secondary task
    # during training. If True, the viseme loss weight will be gradually increased from 0 to viseme_weight
    # over a specified number of updates (e.g., 4000 updates).
    # This allows the model to first focus on the primary task (AVSR) before incorporating the secondary task (viseme classification).
    # This is useful for curriculum learning
    warmup: bool = field(
        default=True,
        metadata={"Help": "Gradually introduce the secondary task"}
    )
    # This is a boolean flag to indicate whether to use convex linear weighting
    convex_linear: bool = field(
        default=False,
        metadata={"help": "use linear weighting for viseme loss"},
    )
    # This is a boolean flag to indicate whether to use learnable weights
    # for the multi-task loss. If True, the model will learn weights for each task
    # during training, allowing it to adaptively balance the contributions of each task.
    # If False, fixed weights will be used based on viseme_weight.
    learnable_weights: bool = field(
        default=False,
        metadata={"help": "use learnable weights for multi-task loss"},
    )
    # Uncertainty-based weighting
    # This is a boolean flag to indicate whether to use uncertainty-based weighting
    # for the multi-task loss. If True, the model will learn to weight the losses
    # based on the uncertainty of each task. based on Kendall et al. (2018) paper.
    # "Multi-task learning using uncertainty to weigh losses for scene geometry and semantics"
    # https://arxiv.org/abs/1805.10196
    uncertainty: bool = field(
        default=False,
        metadata={"help": "use uncertainty-based weighting for multi-task loss"},
    )
    # This is a boolean flag to indicate whether to use UWSO (Uncertainty Weighting by Scaling the Objective)
    # Based on Kirchdorfer et al. (2020) paper.
    # Analytical Uncertainty-Based Loss Weighting in Multi-Task Learning
    uwso: bool = field(
        default=False,
        metadata={"help": "use uncertainty-based weighting with UWSO (Uncertainty Weighting by Scaling the Objectives)"},
    )
    # Random Loss Weighting
    # This is a boolean flag to indicate whether to use random loss weighting
    # for the multi-task loss. If True, the model will randomly weight the losses
    # for each task during training, allowing it to explore different task contributions.
    # https://arxiv.org/abs/2111.10603
    rlw: bool = field(
        default=False,
        metadata={"help": "use random loss weighting for multi-task loss"},
    )
    # Geometric Mean Combination
    # This is a boolean flag to indicate whether to use geometric mean combination
    # for the multi-task loss. If True, the model will combine the losses using the
    # geometric meam.
    nonlinear: bool = field(
        default=False,
        metadata={"help": "use geometric mean combination for multi-task loss"},
    )
    # Dynamic Weight Averaging 
    # This is a boolean flag to indicate whether to use dynamic weight averaging
    # for the multi-task loss. If True, the model will dynamically adjust the weights
    # for each task based on their historical losses, allowing it to adaptively balance
    # the contributions of each task during training. Based on Liu et al. 
    # End-to-End Multi-Task Learning with Attention https://arxiv.org/abs/1803.10704 
    dwa: bool = field(
        default=False,
        metadata={"help": "use dynamic weight averaging for multi-task loss"},
    )
    # Use CTC for viseme classification task
    # This is a boolean flag to indicate whether to use CTC (Connectionist Temporal Classification)
    # for the viseme classification task. If True, the model will use CTC loss
    # for the viseme classification task, which is suitable for sequence-to-sequence tasks
    # where the input and output sequences may have different lengths.
    # Based on Graves et al. (2006) paper.
    # Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks
    use_ctc: bool = field(
        default=False,
        metadata={"help": "use CTC for viseme classification task"},
    )
    



@register_criterion("multitask_criterion", dataclass=MultiTaskCriterionConfig)
class MultiTaskCriterion(FairseqCriterion):
    """Multi-task criterion for AVSR and Viseme classification tasks.
    This criterion combines the losses from the ASR task and the viseme classification task.
    It supports various loss combination strategies, including:
    - Uncertainty-based weighting (Kendall et al., 2018).
    - Uncertainty Weighting by Scaling the Objectives (UWSO).
    - Learnable weights (convex or unconstrained).
    - Fixed weights (convex or unconstrained).
    - Random Loss Weighting (RLW).
    - Dynamic Weight Averaging (DWA).
    - Nonlinear combination using geometric mean.
    
    """
    def __init__(
        self,
        task,
        sentence_avg=True,
        viseme_weight=0,
        label_smoothing=0.1,
        ignore_viseme_padding=True,
        viseme_start_update=0,
        convex_linear=False,
        uncertainty=False,
        learnable_weights=False,
        warmup=True,
        uwso=False,
        nonlinear=False,
        rlw=False,
        dwa=False,
        use_ctc=False,
    ):
        super().__init__(task)
        self.viseme_start_update = viseme_start_update
        self.sentence_avg = sentence_avg
        self.viseme_weight = viseme_weight
        self.eps = label_smoothing
        self.ignore_viseme_padding = ignore_viseme_padding
        self.pad_idx = task.target_dictionary.pad()
        self.convex_linear = convex_linear
        self.uncertainty = uncertainty
        self.uwso = uwso
        self.rlw = rlw
        self.non_linear = nonlinear
        self.warmup = warmup
        self.dwa = dwa
        self.use_ctc = use_ctc
        if self.dwa is True:
            self.avg_asr_loss = None
            self.avg_viseme_loss = None
            self.temp = 2.0  # Temperature for DWA
        self.learnable_weights = learnable_weights
        self.label_smoothed_criterion = LabelSmoothedCrossEntropyCriterion(
        task,
        sentence_avg=sentence_avg,
        label_smoothing=label_smoothing,)
    
    def dwa_loss(self, asr_loss, viseme_loss, current_weight):
        """Dynamic Weight Averaging (DWA) for multi-task loss.
        This method computes the weights for ASR and viseme losses based on their
        historical losses. It uses the ratio of the last two losses to determine
        the weights dynamically, allowing the model to adaptively balance the contributions
        of each task based on their recent performance.
        Args:
            asr_loss (torch.Tensor): The loss for the ASR task.
            viseme_loss (torch.Tensor): The loss for the viseme classification task.
            current_weight (float): The current weight for the viseme loss, used during warmup.
        Returns:
            tuple: A tuple containing:
                - asr_weight (torch.Tensor): The computed weight for the ASR loss.
                - viseme_weight (torch.Tensor): The computed weight for the viseme loss.
        """
        if not hasattr(self, "loss_history_asr"):
                self.loss_history_asr = []
                self.loss_history_viseme = []
        # Append current detached losses
        self.loss_history_asr.append(asr_loss.detach())
        self.loss_history_viseme.append(viseme_loss.detach())

        if len(self.loss_history_asr) < 2:
            # Not enough history to compute DWA, use fixed weights
            w_asr = torch.tensor(1 - current_weight, device=asr_loss.device)
            w_viseme = torch.tensor(current_weight, device=asr_loss.device)
        else:
            # Use the ratio L(t-1) / L(t-2) for each task
            if len(self.loss_history_asr) >= 3:
                w_asr = self.loss_history_asr[-1] / self.loss_history_asr[-2]
                w_viseme = self.loss_history_viseme[-1] / self.loss_history_viseme[-2]
            else:
                # For step=2, just compare current and first loss
                w_asr = self.loss_history_asr[-1] / self.loss_history_asr[0]
                w_viseme = self.loss_history_viseme[-1] / self.loss_history_viseme[0]

            # Stack and apply softmax with temperature
        weights = torch.stack([w_asr, w_viseme])
        softmax_weights = 2 * F.softmax(weights / self.temp, dim=0) # 2 is used, as we have two tasks https://github.com/lorenmt/mtan/issues/2#issuecomment-576666099

        asr_weight = softmax_weights[0]
        viseme_weight = softmax_weights[1]
        return asr_weight, viseme_weight
        
    def uwso_multitask_loss(self, loss_asr, loss_viseme, temp=1, eps=1e-8):
        """
        Calculates the combined loss using the UWSO method.
        Args:
            loss_asr (torch.Tensor): The loss for the ASR task.
            loss_viseme (torch.Tensor): The loss for the viseme classification task.
            eps (float): A small value for numerical stability to avoid division by zero.

        Returns:
            A tuple containing:
                - total_loss (torch.Tensor): The final combined loss, weighted by the UWSO method.
                - weights (torch.Tensor): The computed weights for [asr_loss, viseme_loss].
        """
        # # Convert to tensor if needed
        losses = torch.stack([loss_asr.detach(), loss_viseme.detach()])
        weights = 0.5 * (losses + eps) # Following the UWSO method for Cross-Entropy Loss
        scaled = weights / temp # Temperature = 1
        asr_weight, vsm_weight = F.softmax(scaled, dim=0)  # Softmax to normalize weights
        
        # # Weighted sum of original (non-detached) losses
        total_loss = asr_weight * loss_asr + vsm_weight * loss_viseme
        return total_loss, asr_weight, vsm_weight
    
    def combine_losses(self, model, asr_loss, viseme_loss, current_weight=0.0):
        """Combines ASR and viseme losses based on a configured strategy.

        This method serves as a dispatcher to select the appropriate loss combination
        technique for the multi-task framework. The supported strategies include:
        - Uncertainty-based weighting (Kendall et al., 2018).
        - Uncertainty Weighting by Scaling the Objectives (UWSO).
        - Learnable weights (convex or unconstrained).
        - Fixed weights (convex or unconstrained).

        Args:
            model (torch.nn.Module): The model, used to access learnable parameters
                such as sigmas or weights for certain strategies.
            asr_loss (torch.Tensor): The calculated loss for the ASR task.
            viseme_loss (torch.Tensor): The calculated loss for the viseme task.

        Returns:
            tuple: 
            A tuple containing:
                - The final, combined loss tensor.
                - A dictionary with new metrics for logging (e.g., task weights or sigmas).
        """

        logging_updates = {}
    
        if self.rlw is True:  # Random Loss Weighting
            asr_weight, viseme_weight = F.softmax(torch.randn(size=[2]), dim=0)  # Normalize to sum to 1
            loss = asr_weight * asr_loss + viseme_weight * viseme_loss
            
        elif self.dwa is True:  # Dynamic Weight Averaging
            asr_weight, viseme_weight = self.dwa_loss(asr_loss, viseme_loss, current_weight)
            # Combine the losses using calculated DWA weights
            loss = asr_weight * asr_loss + viseme_weight * viseme_loss
        
        elif self.uwso is True: # Uncertainty Weighting by Scaling the Objectives (UWSO)
            loss, asr_weight, viseme_weight = self.uwso_multitask_loss(asr_loss, viseme_loss)
        
        elif self.uncertainty is True: # Uncertainty Weighting (Kendall et al., 2018)
            sigma_asr = model.s_asr
            sigma_viseme = model.s_vsm
            loss = (
                    torch.exp(-sigma_asr) * asr_loss + sigma_asr / 2 + # Task 1
                    torch.exp(-sigma_viseme) * viseme_loss + sigma_viseme / 2 # Task 2
                )
            logging_updates.update({
                "sigma_asr": sigma_asr.item(),
                "sigma_viseme": sigma_viseme.item(),
            })

        elif self.learnable_weights is True: # Learnable Weights
            if self.convex_linear is True: # Convex Linear combination with learnable weights
                weights = torch.stack([model.asr_weight, model.viseme_weight])
                norm_weights = F.softmax(weights, dim=0)
                asr_weight, viseme_weight = norm_weights[0], norm_weights[1]
            else: # Non Convex Linear combination with learnable positive weights
                asr_weight = model.asr_weight.abs()
                viseme_weight = model.viseme_weight.abs()
            loss = asr_weight * asr_loss + viseme_weight * viseme_loss
        else: # Fixed weights
            viseme_weight = current_weight if self.warmup else self.viseme_weight
            if self.convex_linear is True: # Convex Linear Combination with Fixed Weights
                asr_weight = 1 - viseme_weight
                loss = asr_weight * asr_loss + viseme_weight * viseme_loss
    
            elif self.non_linear is True: # Non Linear Combination with Weighted Geometric Mean
                eps = 1e-8  # Numerical stability
                asr_weight = 1 - viseme_weight
                loss = torch.exp((asr_weight) * torch.log(asr_loss + eps) + viseme_weight * torch.log(viseme_loss + eps))

            else:
                # Non-Convex Linear combination but fixed weight with full AVSR loss
                loss = asr_loss + viseme_weight * viseme_loss
                asr_weight = torch.tensor(1.0, device=asr_loss.device)  # ASR weight is always 1.0 in this case
        # Log information for monitoring
        if self.uncertainty is True:
             logging_updates.update({
                "sigma_asr": sigma_asr.item(),
                "sigma_viseme": sigma_viseme.item(),
            })
        else:
            asr_weight = torch.as_tensor(asr_weight, device=asr_loss.device)
            viseme_weight = torch.as_tensor(viseme_weight, device=asr_loss.device)
            logging_updates.update({
                    "asr_weight": asr_weight.item(),
                    "viseme_weight": viseme_weight.item(),
                })
            
        return loss, logging_updates

    def compute_viseme_loss(self, viseme_output, viseme_target, padding_mask=None, reduce=True):
        """
        Compute Viseme classification loss
        Args:
            viseme_output: Model's logits
            viseme_target: Labels for the viseme classification task
            padding_mask: Padding mask for each sample
            reduce: Flag whether to reduce the loss computation or not
        Returns:
            viseme_loss: the loss for viseme classification
            n_correct: Correct classifications
            total: total number of samples
        """

        B, T, C = viseme_output.size()
        target_B, target_T = viseme_target.size()
        
        # If shapes differ, pad the shorter sequence with ignore_index (-1)
        if T > target_T:
            # Pad target to match output length
            pad_length = T - target_T
            target_padding = torch.full((B, pad_length), -1, 
                                    dtype=viseme_target.dtype,
                                    device=viseme_target.device)
            viseme_target = torch.cat([viseme_target, target_padding], dim=1)
        elif T < target_T:
            # Pad output to match target length
            pad_length = target_T - T
            output_padding = torch.zeros((B, pad_length, C),
                                    dtype=viseme_output.dtype,
                                    device=viseme_output.device)
            viseme_output = torch.cat([viseme_output, output_padding], dim=1)
    
        # Flatten for loss computation
        pred_flat = viseme_output.contiguous().view(-1, C)
        target_flat = viseme_target.contiguous().view(-1)

        viseme_loss = F.cross_entropy(
            pred_flat,
            target_flat,
            ignore_index=-1,
            reduce=reduce,
            # label_smoothing=0.1 # TODO: Try
        )

        if padding_mask is not None and self.ignore_viseme_padding:
            viseme_loss = viseme_loss * (~padding_mask)

        # Calculate accuracy
        pred = pred_flat.argmax(dim=-1)
        mask = target_flat != -1
        n_correct = (pred[mask] == target_flat[mask]).sum()
        total = mask.sum()

        return viseme_loss.sum(), n_correct, total

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample."""
        output = model(**sample["net_input"])
        update = getattr(model, "num_updates")
        sample_size = sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        logging_output = {"sample_size": sample_size,
                          "ntokens": sample["ntokens"],  
                          "nsentences": sample["target"].size(0)}
        # ASR Loss
        net_output = (output["decoder_out"][0], None)  # (logits, extra)
        asr_loss, nll_loss = self.label_smoothed_criterion.compute_loss(model, net_output, sample, reduce=reduce,)
        
        n_correct, total = self.label_smoothed_criterion.compute_accuracy(
            model, net_output, sample)
        logging_output["asr_n_correct"] = utils.item(n_correct.data)
        logging_output["asr_total"] = utils.item(total.data)
        logging_output["asr_loss"] = utils.item(asr_loss.data)
        logging_output["nll_loss"] = utils.item(nll_loss.data)
        # Starting values for loss and viseme weight
        loss = asr_loss
        current_weight = self.viseme_weight  
        # Viseme Loss
        if update >= self.viseme_start_update: # Start viseme training after certain number of updates
            if self.use_ctc is True:
                viseme_target = sample['viseme_target']
                viseme_input_lengths = sample['viseme_input_lengths']
                viseme_lengths_target = sample['viseme_target_lengths']
                viseme_pad_idx = 0
                viseme_logits = output['viseme_out']
                viseme_lprobs = F.log_softmax(viseme_logits, dim=-1)
                viseme_lprobs = viseme_lprobs.transpose(0, 1).double()
                viseme_loss = F.ctc_loss(viseme_lprobs, viseme_target, viseme_input_lengths, viseme_lengths_target, blank=viseme_pad_idx, reduction='sum' if reduce else 'none')
                logging_output.update({
                    "viseme_loss": viseme_loss.item(),
                    "viseme_ntokens": viseme_lengths_target.sum().item(),
                })
            # Compute viseme loss
            else:
                viseme_padding_mask = sample["viseme_padding_mask"] if "viseme_padding_mask" in sample else None
                viseme_loss, vsm_n_correct, vsm_total = self.compute_viseme_loss(
                    output["viseme_out"],
                    sample["viseme_target"],
                    viseme_padding_mask,
                    reduce=True
                )
                logging_output.update({
                        "viseme_loss": viseme_loss.item(),
                        "vsm_n_correct": vsm_n_correct.item(),
                        "viseme_ntokens": vsm_total.item(),
                    })
            if self.warmup is True: # If warmup is enabled, gradually increase the viseme loss weight
                warmup_updates = 4000
                progress = min(1.0, (update - self.viseme_start_update) / warmup_updates)
                current_weight = self.viseme_weight * progress
                # current_weight = self.viseme_weight * (1 - math.(exp-5 * progress)) # TODO: Exponential decay
                # current_weight = self.viseme_weight* (1 - math.cos(math.pi * progress / 2)) # TODO: Cosine decay
                loss, log_updates = self.combine_losses(model, asr_loss, viseme_loss, current_weight)
                logging_output.update(log_updates)
            else: # Else use non-warmup viseme weight
                loss, log_updates = self.combine_losses(model, asr_loss, viseme_loss, current_weight=current_weight)
                # Log
                logging_output.update(log_updates)

        # Weight for  weighted sum of per layer features
        if hasattr(model, "layer_weigher"):
            scalar_mix_weights = model.layer_weigher.normalized_weights.cpu().tolist() 
            for i, weight in enumerate(scalar_mix_weights):
                logging_output[f"scalar_mix_weights/layer_{i}"] = weight
            # If the model has a weight mix, log the normalized weights
        
        logging_output["loss"] = utils.item(loss.data) 

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        asr_loss_sum = utils.item(sum(log.get("asr_loss", 0) for log in logging_outputs))
        nll_loss_sum = utils.item(sum(log.get("nll_loss", 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        sample_size = utils.item(sum(log.get("sample_size", 0) for log in logging_outputs))

        # Log main metrics
        metrics.log_scalar("loss", loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar("asr_loss", asr_loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar("nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_derived("ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg))
        
        asr_total = utils.item(sum(log.get("asr_total", 0) for log in logging_outputs))
        asr_n_correct = utils.item(sum(log.get("asr_n_correct", 0) for log in logging_outputs))
        if asr_total > 0:
            metrics.log_scalar("asr_total", asr_total)
            metrics.log_scalar("asr_n_correct", asr_n_correct)
            metrics.log_derived(
                "asr_accuracy",
                lambda meters: round(
                    meters["asr_n_correct"].sum * 100.0 / meters["asr_total"].sum, 3
                ))


        # Log viseme metrics if present
        if any("viseme_loss" in log for log in logging_outputs):
            viseme_loss = sum(log.get("viseme_loss", 0) for log in logging_outputs)
            n_correct = sum(log.get("vsm_n_correct", 0) for log in logging_outputs)
            viseme_ntokens = sum(log.get("viseme_ntokens", 0) for log in logging_outputs)
            metrics.log_scalar("viseme_loss", viseme_loss / viseme_ntokens / math.log(2), viseme_ntokens, round=3)
            if n_correct != 0:
                metrics.log_scalar("viseme_accuracy", n_correct * 100.0 / viseme_ntokens,  viseme_ntokens, round=3)
        if any("sigma_asr" in log for log in logging_outputs):
            avg_sigma_asr = sum(log.get("sigma_asr", 0) for log in logging_outputs) / len(logging_outputs)
            avg_sigma_viseme = sum(log.get("sigma_viseme", 0) for log in logging_outputs) / len(logging_outputs)
            metrics.log_scalar("sigma_asr", avg_sigma_asr)
            metrics.log_scalar("sigma_viseme", avg_sigma_viseme)
        
        if any("asr_weight" in log for log in logging_outputs):
            asr_weight = utils.item(sum(log.get("asr_weight", 0) for log in logging_outputs)) / len(logging_outputs)
            viseme_weight = utils.item(sum(log.get("viseme_weight", 0) for log in logging_outputs)) / len(logging_outputs)
            metrics.log_scalar("asr_weight", asr_weight)
            metrics.log_scalar("viseme_weight", viseme_weight)
        
        layer_weight_keys = [k for k in logging_outputs[0].keys() if k.startswith("scalar_mix_weights/layer_")]
        num_layers = len(layer_weight_keys)
        if num_layers > 0:
            for i in range(num_layers):
                layer_key = f"scalar_mix_weights/layer_{i}"
                avg_weight = sum(log.get(layer_key, 0) for log in logging_outputs) / len(logging_outputs)
                metrics.log_scalar(layer_key, avg_weight, round=4)
        

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return True