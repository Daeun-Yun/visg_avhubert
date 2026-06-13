
PRETRAINED_MODEL_PATH=/data/DB/large_vox_iter5.pt
result=/data/results/avhubert/clean
ROOT=$(dirname "$(dirname "$(readlink -fm "$0")")")
AV_HUBERT=${ROOT}

export PYTHONPATH="/home/dan/projects/av_hubert/fairseq:$PYTHONPATH"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
data=/data/DB/lrs3/433h_data
bpe_model=/data/DB/lrs3/spm1000/spm_unigram1000.model
#visg_avhubert
#1번째 paper with noise
if [ -f "${result}/finish.txt" ]; then
    echo "=== [SKIP train] visg_noise already finished ==="
else
        fairseq-hydra-train \
        --config-dir ${AV_HUBERT}/conf/av-finetune \
        --config-name large_noise_pt_noise_ft_433h.yaml \
        task.data=$data \
        task.label_dir=$data \
        task.tokenizer_bpe_model=$bpe_model \
        model.w2v_path=${PRETRAINED_MODEL_PATH} \
        common.user_dir=${PWD} \
        task.noise_wav=null \
        hydra.run.dir=${result} \
        task.noise_prob=0.0 \
        dataset.num_workers=6 \
        distributed_training.distributed_world_size=1 \
        distributed_training.nprocs_per_node=1 \
        checkpoint.save_interval=1 &&
    echo "finished : $(date '+%Y-%m-%d %H:%M:%S')" > ${result}/finish.txt
fi

if [ -f "${result}/s2s/total_wer.txt" ]; then
    echo "=== [SKIP infer] avhubert already has total_wer.txt ==="
else
    ${AV_HUBERT}/infer_all.sh ${result} mix
fi

result=/data/results/avhubert/noise
ROOT=$(dirname "$(dirname "$(readlink -fm "$0")")")
AV_HUBERT=${ROOT}

export PYTHONPATH="/home/dan/projects/av_hubert/fairseq:$PYTHONPATH"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
data=/data/DB/lrs3/433h_data
bpe_model=/data/DB/lrs3/spm1000/spm_unigram1000.model
#visg_avhubert
#1번째 paper with noise
if [ -f "${result}/finish.txt" ]; then
    echo "=== [SKIP train] avhubert noise already finished ==="
else
        fairseq-hydra-train \
        --config-dir ${AV_HUBERT}/conf/av-finetune \
        --config-name large_noise_pt_noise_ft_433h.yaml \
        task.data=$data \
        task.label_dir=$data \
        task.tokenizer_bpe_model=$bpe_model \
        model.w2v_path=${PRETRAINED_MODEL_PATH} \
        common.user_dir=${PWD} \
        task.noise_wav=/data/DB/musan/tsv/all \
        hydra.run.dir=${result} \
        optimization.clip_norm=10.0 \
        dataset.num_workers=6 \
        distributed_training.distributed_world_size=1 \
        distributed_training.nprocs_per_node=1 \
        checkpoint.save_interval=1 &&
    echo "finished : $(date '+%Y-%m-%d %H:%M:%S')" > ${result}/finish.txt
fi

if [ -f "${result}/s2s/total_wer.txt" ]; then
    echo "=== [SKIP infer] visg_noise already has total_wer.txt ==="
else
    ${AV_HUBERT}/infer_all.sh ${result} mix
fi
