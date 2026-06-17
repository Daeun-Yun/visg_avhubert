#! /bin/bash
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

GROUP=test
MODALITIES="audio,video"
MODEL_PATH=/workspace/code/visg_avhubert/large_noise_pt_noise_ft_433h.pt
BASE_OUT_PATH=/workspace/code/results/large_noise_pt_noise_ft_433h

# noise types
NOISE_TYPES=(
    "/DB/lrs3/noise/speech"
)

# SNR levels
SNR_LEVELS=(-10)

# set paths
AV_HUBERT=$(dirname "$(dirname "$(readlink -fm "$0")")")
ROOT=$(dirname "${AV_HUBERT}")
export PYTHONPATH="${ROOT}/fairseq:$PYTHONPATH"

# summary file (CSV)
SUMMARY_FILE="${BASE_OUT_PATH}/summary.csv"
mkdir -p "${BASE_OUT_PATH}"
echo "SNR,NoiseType,WER" > "$SUMMARY_FILE"

for NOISE in "${NOISE_TYPES[@]}"; do
  for SNR in "${SNR_LEVELS[@]}"; do

    if [[ "$NOISE" == *"/musan/"* ]]; then
        NOISE_NAME="MUSAN_$(basename "$NOISE")"
    else
        NOISE_NAME=$(basename "$NOISE")
    fi

    OUT_PATH="${BASE_OUT_PATH}/${NOISE_NAME}_snr${SNR}"
    mkdir -p "$OUT_PATH"

    echo ">>> Running decode with noise=${NOISE_NAME}, SNR=${SNR}"

    # capture stdout
    LOGFILE="${OUT_PATH}/log.txt"
    python -B ${AV_HUBERT}/infer_s2s.py \
        --config-dir ${AV_HUBERT}/conf \
        --config-name s2s_decode \
            common.user_dir=${AV_HUBERT} \
            override.modalities=[${MODALITIES}] \
            dataset.gen_subset=${GROUP} \
            override.data=/DB/lrs3/433h_data \
            override.label_dir=/DB/lrs3/433h_data \
            common_eval.path=${MODEL_PATH} \
            common_eval.results_path=${OUT_PATH} \
            override.noise_prob=1 \
            override.noise_snr=${SNR} \
            override.noise_wav=${NOISE} \
            override.noise_num=1 \
            distributed_training.distributed_world_size=1 | tee "$LOGFILE"

    # Extract WER (format: "WER: 9.979777553083924%")
    WER=$(grep -oP "WER:\s*\K[0-9.]+(?=%)" "$LOGFILE")

    if [ -n "$WER" ]; then
        WER=$(awk -v val="$WER" 'BEGIN { printf "%.4f", val }')
    else
        WER="N/A"
    fi


    # append to summary CSV
    echo "${SNR},${NOISE_NAME},${WER}" >> "$SUMMARY_FILE"
  done
done

# sort summary.csv in place by SNR (numeric) then NoiseType (alphabetical)
(head -n 1 "$SUMMARY_FILE" && tail -n +2 "$SUMMARY_FILE" | sort -t, -k1,1n -k2,2) > "${SUMMARY_FILE}.tmp" && mv "${SUMMARY_FILE}.tmp" "$SUMMARY_FILE"

echo
echo ">>> All decoding runs completed. Sorted CSV summary saved at: ${SUMMARY_FILE}"
awk -F, '{printf "%-8s %-25s %s\n", $1, $2, $3}' "$SUMMARY_FILE"



# #! /bin/bash

# GROUP=test
# MODALITIES="audio,video"
# MODEL_PATH=/data/DB/large_checkpoint_best.pt
# BASE_OUT_PATH=/data/DB/decode_large_433h_noise

# # noise types
# NOISE_TYPES=(
#     "/data/DB/lrs3/noise/speech"
#     "/data/DB/lrs3/noise/babble"
#     "/data/DB/musan/tsv/babble"
#     "/data/DB/musan/tsv/music"
#     "/data/DB/musan/tsv/noise"
# )

# # SNR levels
# SNR_LEVELS=(-10 -5 0 5 10)

# # set paths
# AV_HUBERT=$(dirname "$(dirname "$(readlink -fm "$0")")")
# ROOT=$(dirname "${AV_HUBERT}")
# export PYTHONPATH="${ROOT}/fairseq:$PYTHONPATH"

# # summary file (CSV)
# SUMMARY_FILE="${BASE_OUT_PATH}/summary.csv"
# echo "SNR,NoiseType,WER" > "$SUMMARY_FILE"

# for NOISE in "${NOISE_TYPES[@]}"; do
#   for SNR in "${SNR_LEVELS[@]}"; do

#     if [[ "$NOISE" == *"/musan/"* ]]; then
#         NOISE_NAME="MUSAN_$(basename "$NOISE")"
#     else
#         NOISE_NAME=$(basename "$NOISE")
#     fi

#     OUT_PATH="${BASE_OUT_PATH}/${NOISE_NAME}_snr${SNR}"
#     mkdir -p "$OUT_PATH"

#     echo ">>> Running decode with noise=${NOISE_NAME}, SNR=${SNR}"

#     # capture stdout
#     LOGFILE="${OUT_PATH}/log.txt"
#     python -B ${AV_HUBERT}/infer_s2s.py \
#         --config-dir ${AV_HUBERT}/conf \
#         --config-name s2s_decode \
#             common.user_dir=${AV_HUBERT} \
#             override.modalities=[${MODALITIES}] \
#             dataset.gen_subset=${GROUP} \
#             override.data=/data/DB/lrs3/433h_data \
#             override.label_dir=/data/DB/lrs3/433h_data \
#             common_eval.path=${MODEL_PATH} \
#             common_eval.results_path=${OUT_PATH} \
#             override.noise_prob=1 \
#             override.noise_snr=${SNR} \
#             override.noise_wav=${NOISE} \
#             override.w2v_path=/data/DB/large_vox_iter5.pt \
#             distributed_training.distributed_world_size=1 | tee "$LOGFILE"

#     # Extract WER (format: "WER: 9.979777553083924%")
#     WER=$(grep -oP "WER:\s*\K[0-9.]+(?=%)" "$LOGFILE")

#     if [ -n "$WER" ]; then
#         WER=$(awk -v val="$WER" 'BEGIN { printf "%.4f", val }')
#     else
#         WER="N/A"
#     fi


#     # append to summary CSV
#     echo "${SNR},${NOISE_NAME},${WER}" >> "$SUMMARY_FILE"
#   done
# done

# # sort summary.csv in place by SNR (numeric) then NoiseType (alphabetical)
# (head -n 1 "$SUMMARY_FILE" && tail -n +2 "$SUMMARY_FILE" | sort -t, -k1,1n -k2,2) > "${SUMMARY_FILE}.tmp" && mv "${SUMMARY_FILE}.tmp" "$SUMMARY_FILE"

# echo
# echo ">>> All decoding runs completed. Sorted CSV summary saved at: ${SUMMARY_FILE}"
# awk -F, '{printf "%-8s %-25s %s\n", $1, $2, $3}' "$SUMMARY_FILE"
