#!/bin/bash
# Usage: ./infer_musan.sh <result_path>
# Tests inference across MUSAN noise categories × SNR levels using checkpoint_best.pt
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
result=${1:?"Usage: $0 <result_path>"}

ALL_TSV=/data/DB/musan/tsv/all/test.tsv
NOISE_CATEGORIES=("babble" "music" "noise" "speech")
SNR_LEVELS=(-10 -5 0 5 10)
data_433h=/data/DB/lrs3/433h_data
conf_name=s2s_decode

mkdir -p "${result}/s2s"
exec > >(tee "${result}/s2s/infer_musan.log") 2>&1

# ── 체크포인트 선택 (infer_all.sh와 동일 방식) ─────────────────────────
_get_best_step() {
    local result_dir="$1"
    python3 - "${result_dir}" <<'PYEOF'
import json, sys, os
result = sys.argv[1]
log_path = os.path.join(result, "hydra_train.log")
best_loss = float('inf')
best_update = -1
with open(log_path) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        if ' - ' in line:
            line = line.split(' - ', 1)[1]
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        num_updates = int(d.get("valid_num_updates", 0))
        loss = d.get("valid_loss")
        if loss is None:
            continue
        if float(loss) < best_loss:
            best_loss = float(loss)
            best_update = num_updates
if best_update < 0:
    print("ERROR: no valid_loss found in hydra_train.log", file=sys.stderr)
    sys.exit(1)
print(f"{best_update} {best_loss:.4f}")
PYEOF
}

ckpt="${result}/checkpoints/checkpoint_best.pt"
[ ! -f "${ckpt}" ] && { echo "ERROR: checkpoint_best.pt not found in ${result}/checkpoints/"; exit 1; }
best_info=$(_get_best_step "${result}")
if [ $? -eq 0 ]; then
    best_step=$(echo "${best_info}" | awk '{print $1}')
    best_loss_val=$(echo "${best_info}" | awk '{print $2}')
    echo "=== Using checkpoint_best.pt (step ${best_step}, valid_loss=${best_loss_val}) ==="
else
    echo "=== Using checkpoint_best.pt ==="
fi

# ── 카테고리별 임시 TSV 생성 ──────────────────────────────────────────
TMPDIR=$(mktemp -d)
trap "rm -rf ${TMPDIR}" EXIT

declare -A CATEGORY_PATTERN
CATEGORY_PATTERN["babble"]="musan/short-musan/babble/"
CATEGORY_PATTERN["music"]="musan/short-musan/music/"
CATEGORY_PATTERN["noise"]="musan/short-musan/noise/"
CATEGORY_PATTERN["speech"]="lrs3/noise/speech/"

for cat in "${NOISE_CATEGORIES[@]}"; do
    mkdir -p "${TMPDIR}/${cat}"
    grep "${CATEGORY_PATTERN[$cat]}" "${ALL_TSV}" > "${TMPDIR}/${cat}/test.tsv"
    count=$(wc -l < "${TMPDIR}/${cat}/test.tsv")
    echo "=== Category '${cat}': ${count} noise files ==="
done

# ── 카테고리 × SNR 추론 ───────────────────────────────────────────────
summary_file="${result}/s2s/musan_best.txt"
> "${summary_file}"
printf "%-10s %8s %8s\n" "category" "SNR(dB)" "WER" >> "${summary_file}"
echo "----------------------------------------" >> "${summary_file}"

for cat in "${NOISE_CATEGORIES[@]}"; do
    noise_tsv="${TMPDIR}/${cat}"
    for snr in "${SNR_LEVELS[@]}"; do
        tag="${cat}_snr${snr}"
        out_dir="${result}/s2s/musan_${tag}"
        mkdir -p "${out_dir}"

        if ls "${out_dir}/wer."* 2>/dev/null | grep -q .; then
            echo "=== [SKIP] ${tag} already done ==="
            wer_file=$(ls "${out_dir}/wer."* | head -1)
            wer=$(grep "^WER:" "${wer_file}" | awk '{print $2}')
            printf "%-10s %8s %8s\n" "${cat}" "${snr}" "${wer}" >> "${summary_file}"
            continue
        fi

        echo ""
        echo "=== [${cat}] SNR=${snr}dB ==="

        python3 -B infer_s2s.py \
            --config-dir ./conf/ \
            --config-name ${conf_name} \
            dataset.gen_subset=test \
            common_eval.path=${ckpt} \
            common_eval.results_path=${out_dir} \
            override.modalities=['audio','video'] \
            common.user_dir=$(pwd) \
            override.noise_prob=1.0 \
            override.noise_wav=${noise_tsv} \
            override.noise_snr=${snr} \
            override.data=${data_433h} \
            override.label_dir=${data_433h} \
            distributed_training.distributed_world_size=1

        wer_file=$(ls "${out_dir}/wer."* 2>/dev/null | head -1)
        if [ -n "${wer_file}" ]; then
            wer=$(grep "^WER:" "${wer_file}" | awk '{print $2}')
            printf "%-10s %8s %8s\n" "${cat}" "${snr}" "${wer}" >> "${summary_file}"
        else
            printf "%-10s %8s %8s\n" "${cat}" "${snr}" "N/A" >> "${summary_file}"
        fi
    done
    echo "----------------------------------------" >> "${summary_file}"
done



echo ""
echo "=== MUSAN Noise Robustness Summary ==="
cat "${summary_file}"
echo ""
echo "=== Summary saved: ${summary_file} ==="
