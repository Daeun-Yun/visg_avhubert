# visg 모델 433h base 실험
# viseme_weight: {0.15, 0.2} x viseme_start: {20000, 0} 4가지 조합 순차 실행
PRETRAINED_MODEL_PATH=/workspace/code/visg_avhubert/base_vox_iter5.pt
ROOT=$(dirname "$(dirname "$(readlink -fm "$0")")")
AV_HUBERT=${ROOT}

export PYTHONPATH="/workspace/code/visg_avhubert/fairseq:$PYTHONPATH"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=0,1

data=/DB/lrs3/433h_data
bpe_model=/DB/lrs3/spm1000/spm_unigram1000.model

GROUP=test
MODALITIES="audio,video"

# 조합: "viseme_weight viseme_start result_suffix"
COMBINATIONS=(
    # "0.15 20000 vw0.15_vs20k"
    "0.2  20000 vw0.2_vs20k"
    # "0.15 0     vw0.15_vs0" #완
    # "0.2  0     vw0.2_vs0" #완
)

for combo in "${COMBINATIONS[@]}"; do
    read -r VISEME_WEIGHT VISEME_START SUFFIX <<< "$combo"
    result=/DB/results/visg/visg_base_433h_${SUFFIX}
    OUT_PATH="${result}/s2s/decode"

    echo ""
    echo "=========================================="
    echo ">>> [START] viseme_weight=${VISEME_WEIGHT}, viseme_start=${VISEME_START}"
    echo ">>> result: ${result}"
    echo "=========================================="

    # training
    if [ -f "${result}/finish.txt" ]; then
        echo "=== [SKIP train] already finished ==="
    else
        fairseq-hydra-train \
        --config-dir ${AV_HUBERT}/conf/av-finetune \
        --config-name vsm_base_noise_pt_noise_ft_433h.yaml \
        task.data=$data \
        task.label_dir=$data \
        task.tokenizer_bpe_model=$bpe_model \
        model.w2v_path=${PRETRAINED_MODEL_PATH} \
        common.user_dir=${PWD} \
        task.noise_wav=/DB/musan/tsv/all \
        hydra.run.dir=${result} \
        task.viseme_dir=$data \
        criterion.viseme_weight=${VISEME_WEIGHT} \
        distributed_training.distributed_world_size=2 \
        distributed_training.nprocs_per_node=2 \
        distributed_training.distributed_port=0 \
        criterion.viseme_start_update=${VISEME_START} \
        dataset.max_tokens=4000 \
        optimization.update_freq=[1] &&
        echo "finished : $(date '+%Y-%m-%d %H:%M:%S')" > ${result}/finish.txt
    fi
    

    # start decoding (clean)
    if [ -f "${result}/s2s/clean_finish.txt" ]; then
        echo "=== [SKIP clean decode] already finished ==="
    else
        python -B ${AV_HUBERT}/infer_s2s.py \
            --config-dir ${AV_HUBERT}/conf \
            --config-name s2s_decode \
                common.user_dir=${AV_HUBERT} \
                override.modalities=[${MODALITIES}] \
                dataset.gen_subset=${GROUP} \
                override.data=${data} \
                override.label_dir=${data} \
                common_eval.path=${result}/checkpoints/checkpoint_best.pt \
                common_eval.results_path=${OUT_PATH} \
                override.noise_prob=0.0 \
                distributed_training.distributed_world_size=1 &&
        echo "finished : $(date '+%Y-%m-%d %H:%M:%S')" > "${result}/s2s/clean_finish.txt"
    fi

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
            NOISE="/DB/lrs3/noise/babble"
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
            NOISE="/DB/lrs3/noise/speech"
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

done

echo ""
echo "=========================================="
echo ">>> All 4 experiments complete."
echo "=========================================="
