# #! /bin/bash
#
# GROUP=test
# MODALITIES="video"
# MODEL_PATH=/home/aristosp/models/exp2/avhubert/checkpoints/normal_finetune/checkpoints/checkpoint_best.pt
# TOKENIZER_PATH=/home/aristosp/datasets/LRS3/spm1000/spm_unigram1000.model
# OUT_PATH=/home/aristosp/models/exp2/avhubert/checkpoints/normal_finetune/decode/
# # override.noise_prob=1
# # +override.noise_wav=/home/aristosp/datasets/musan/tsv/babble/
# # set paths
# ROOT=$(dirname "$(dirname "$(readlink -fm "$0")")")
# AV_HUBERT=${ROOT}/avhubert
# export PYTHONPATH="${ROOT}/fairseq:$PYTHONPATH"
# # start decoding
# python -B ${AV_HUBERT}/infer_s2s.py \
#     --config-dir ${AV_HUBERT}/conf \
#     --config-name s2s_decode \
#         common.user_dir=${AV_HUBERT} \
#         override.modalities=[${MODALITIES}] \
#         dataset.gen_subset=${GROUP} \
#         override.data=/home/aristosp/datasets/LRS3/30h_data \
#         override.label_dir=/home/aristosp/datasets/LRS3/30h_data \
#         common_eval.path=${MODEL_PATH} \
#         common_eval.results_path=${OUT_PATH} \
#         # override.noise_prob=1 \
#         # override.noise_snr=-10 \
#         # +override.noise_wav=/home/aristosp/datasets/musan/tsv/noise \

#! /bin/bash

GROUP=test
MODALITIES="audio,video"
MODEL_PATH=/data/DB/large_checkpoint_best.pt
DATA_PATH=/data/DB/lrs3/433h_data
OUT_PATH=/data/DB/decode_large_433h_clean
PRETRAINED_MODEL_PATH=/home/dan/projects/av_hubert/lrs3_vox_noise_pt_iter5.pt

# set paths
AV_HUBERT=$(dirname "$(dirname "$(readlink -fm "$0")")")
ROOT=$(dirname "${AV_HUBERT}")
export PYTHONPATH="${ROOT}/fairseq:$PYTHONPATH"

# start decoding
python -B ${AV_HUBERT}/infer_s2s.py \
    --config-dir ${AV_HUBERT}/conf \
    --config-name s2s_decode \
        common.user_dir=${AV_HUBERT} \
        override.modalities=[${MODALITIES}] \
        dataset.gen_subset=${GROUP} \
        override.data=${DATA_PATH} \
        override.label_dir=${DATA_PATH} \
        common_eval.path=${MODEL_PATH} \
        common_eval.results_path=${OUT_PATH} \
        override.noise_prob=0.0 \
        distributed_training.distributed_world_size=1