#! /bin/bash

MODEL_PATH=/workspace/code/visg_large_best.pt
W2V_PATH=/workspace/code/large_vox_iter5.pt

AV_HUBERT=$(dirname "$(dirname "$(readlink -fm "$0")")")
ROOT=$(dirname "${AV_HUBERT}")
export PYTHONPATH="${ROOT}/fairseq:$PYTHONPATH"

data=/DB/lrs3/433h_data
MASS_OUT_PATH=$(dirname "$(readlink -fm "$0")")/results/large_noise_pt_noise_ft_433h

GROUP=test
MODALITIES="audio,video"

SNR=-10
NOISE="/DB/lrs3/noise/speech"
NOISE_NAME="speech"
SPEECH_REPEATS=3

SUMMARY_FILE="${MASS_OUT_PATH}/summary_speech_snr${SNR}.csv"
mkdir -p "${MASS_OUT_PATH}"
echo "SNR,NoiseType,WER" > "$SUMMARY_FILE"

WER_SUM=0
WER_COUNT=0

for RUN in $(seq 1 $SPEECH_REPEATS); do
    NOISY_OUT_PATH="${MASS_OUT_PATH}/${NOISE_NAME}_snr${SNR}_run${RUN}"
    mkdir -p "$NOISY_OUT_PATH"
    LOGFILE="${NOISY_OUT_PATH}/log.txt"

    python -B ${AV_HUBERT}/infer_s2s.py \
        --config-dir ${AV_HUBERT}/conf \
        --config-name s2s_decode \
            common.user_dir=${AV_HUBERT} \
            override.modalities=[${MODALITIES}] \
            dataset.gen_subset=${GROUP} \
            override.data=${data} \
            override.label_dir=${data} \
            common_eval.path=${MODEL_PATH} \
            common_eval.results_path=${NOISY_OUT_PATH} \
            override.noise_prob=1 \
            "override.noise_snr=${SNR}" \
            +override.noise_wav=${NOISE} \
            override.w2v_path=${W2V_PATH} \
            distributed_training.distributed_world_size=1 | tee "$LOGFILE"

    WER=$(grep -oP "WER:\s*\K[0-9.]+(?=%)" "$LOGFILE")
    if [ -n "$WER" ]; then
        WER_SUM=$(awk -v sum="$WER_SUM" -v val="$WER" 'BEGIN { printf "%.6f", sum + val }')
        WER_COUNT=$((WER_COUNT + 1))
    fi
done

if [ "$WER_COUNT" -gt 0 ]; then
    AVG_WER=$(awk -v sum="$WER_SUM" -v count="$WER_COUNT" 'BEGIN { printf "%.4f", sum / count }')
else
    AVG_WER="N/A"
fi

echo "${SNR},${NOISE_NAME}_avg${SPEECH_REPEATS}runs,${AVG_WER}" >> "$SUMMARY_FILE"

echo ""
echo ">>> Speech decode complete (SNR=${SNR}, ${SPEECH_REPEATS} runs)"
awk -F, '{printf "%-8s %-30s %s\n", $1, $2, $3}' "$SUMMARY_FILE"
