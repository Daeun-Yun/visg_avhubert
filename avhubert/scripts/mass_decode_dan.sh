# # visg 모델 30h로 학습 warmup o 
PRETRAINED_MODEL_PATH=/home/dan/projects/av_hubert/lrs3_vox_noise_pt_iter5.pt
result=/data/results/visg/visg_noise_ft_30h_ctcwarm_false
OUT_PATH="${result}/s2s/decode"

ROOT=$(dirname "$(dirname "$(readlink -fm "$0")")")
AV_HUBERT=${ROOT}

export PYTHONPATH="/home/dan/projects/av_hubert/fairseq:$PYTHONPATH"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
data=/data/DB/lrs3/30h_data
bpe_model=/data/DB/lrs3/spm1000/spm_unigram1000.model


# mass decode (babble): 1회 측정
if [ -f "${result}/s2s/babble_finish.txt" ]; then
    echo "=== [SKIP babble decode] already finished ==="
else
    SNR_LEVELS=(-10 -5 0 5 10)
    MASS_OUT_PATH="${result}/s2s/mass_decode"
    SUMMARY_FILE="${MASS_OUT_PATH}/summary_babble.csv"
    mkdir -p "${MASS_OUT_PATH}"
    echo "SNR,NoiseType,WER" > "$SUMMARY_FILE"

    for SNR in "${SNR_LEVELS[@]}"; do
        NOISE="/data/DB/lrs3/noise/babble"
        NOISE_NAME="babble"
        NOISY_OUT_PATH="${MASS_OUT_PATH}/${NOISE_NAME}_snr${SNR}"
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
                common_eval.path=${result}/checkpoints/checkpoint_best.pt \
                common_eval.results_path=${NOISY_OUT_PATH} \
                override.noise_prob=1 \
                "override.noise_snr=${SNR}" \
                override.noise_wav=${NOISE} \
                distributed_training.distributed_world_size=1 | tee "$LOGFILE"
        WER=$(grep -oP "WER:\s*\K[0-9.]+(?=%)" "$LOGFILE")
        [ -n "$WER" ] && WER=$(awk -v val="$WER" 'BEGIN { printf "%.4f", val }') || WER="N/A"
        echo "${SNR},${NOISE_NAME},${WER}" >> "$SUMMARY_FILE"
    done

    (head -n 1 "$SUMMARY_FILE" && tail -n +2 "$SUMMARY_FILE" | sort -t, -k1,1n -k2,2) > "${SUMMARY_FILE}.tmp" && mv "${SUMMARY_FILE}.tmp" "$SUMMARY_FILE"
    echo ">>> Babble decode complete. Summary: ${SUMMARY_FILE}"
    column -s, -t "$SUMMARY_FILE"
    echo "finished : $(date '+%Y-%m-%d %H:%M:%S')" > "${result}/s2s/babble_finish.txt"
fi

# mass decode (speech): 3회 반복 후 평균
if [ -f "${result}/s2s/speech_finish.txt" ]; then
    echo "=== [SKIP speech decode] already finished ==="
else
    SNR_LEVELS=(-10 -5 0 5 10)
    SPEECH_REPEATS=3
    MASS_OUT_PATH="${result}/s2s/mass_decode"
    SUMMARY_FILE="${MASS_OUT_PATH}/summary_speech.csv"
    mkdir -p "${MASS_OUT_PATH}"
    echo "SNR,NoiseType,WER" > "$SUMMARY_FILE"

    for SNR in "${SNR_LEVELS[@]}"; do
        NOISE="/data/DB/lrs3/noise/speech"
        NOISE_NAME="speech"
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
                    common_eval.path=${result}/checkpoints/checkpoint_best.pt \
                    common_eval.results_path=${NOISY_OUT_PATH} \
                    override.noise_prob=1 \
                    "override.noise_snr=${SNR}" \
                    override.noise_wav=${NOISE} \
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
    done

    (head -n 1 "$SUMMARY_FILE" && tail -n +2 "$SUMMARY_FILE" | sort -t, -k1,1n -k2,2) > "${SUMMARY_FILE}.tmp" && mv "${SUMMARY_FILE}.tmp" "$SUMMARY_FILE"
    echo ">>> Speech decode complete. Summary: ${SUMMARY_FILE}"
    column -s, -t "$SUMMARY_FILE"
    echo "finished : $(date '+%Y-%m-%d %H:%M:%S')" > "${result}/s2s/speech_finish.txt"
fi
