PRETRAINED_MODEL_PATH=/data/DB/large_vox_iter5.pt
result=/data/results/visg/visg_noise_ctcstart40k
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
        --config-name vsm_large_noise_pt_noise_ft_433h.yaml \
        task.data=$data \
        task.label_dir=$data \
        task.tokenizer_bpe_model=$bpe_model \
        model.w2v_path=${PRETRAINED_MODEL_PATH} \
        common.user_dir=${PWD} \
        task.noise_wav=/data/DB/musan/tsv/all \
        hydra.run.dir=${result} \
        task.viseme_dir=$data \
        dataset.num_workers=6 \
        distributed_training.distributed_world_size=1 \
        distributed_training.nprocs_per_node=1 \
        criterion.viseme_start_update=40000 \
        checkpoint.save_interval=1 &&
    echo "finished : $(date '+%Y-%m-%d %H:%M:%S')" > ${result}/finish.txt
fi

if [ -f "${result}/s2s/total_wer.txt" ]; then
    echo "=== [SKIP infer] visg_noise already has total_wer.txt ==="
else
    ${AV_HUBERT}/infer_all.sh ${result} mix
fi



PRETRAINED_MODEL_PATH=/data/DB/large_vox_iter5.pt
result=/data/results/visg/visg_noise_ctcstart0k
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
        --config-name vsm_large_noise_pt_noise_ft_433h.yaml \
        task.data=$data \
        task.label_dir=$data \
        task.tokenizer_bpe_model=$bpe_model \
        model.w2v_path=${PRETRAINED_MODEL_PATH} \
        common.user_dir=${PWD} \
        task.noise_wav=/data/DB/musan/tsv/all \
        hydra.run.dir=${result} \
        task.viseme_dir=$data \
        dataset.num_workers=6 \
        distributed_training.distributed_world_size=1 \
        distributed_training.nprocs_per_node=1 \
        criterion.viseme_start_update=0 \
        checkpoint.save_interval=1 &&
    echo "finished : $(date '+%Y-%m-%d %H:%M:%S')" > ${result}/finish.txt
fi

if [ -f "${result}/s2s/total_wer.txt" ]; then
    echo "=== [SKIP infer] visg_noise already has total_wer.txt ==="
else
    ${AV_HUBERT}/infer_all.sh ${result} mix
fi



# #2번째 paper without noise
# if [ -f "/home/litsub09/projects/visg_avhubert/visg/finish.txt" ]; then
#     echo "=== [SKIP train] visg already finished ==="
# else
#     fairseq-hydra-train \
#         --config-dir ${AV_HUBERT}/conf/av-finetune \
#         --config-name vsm_large_noise_pt_noise_ft_433h.yaml \
#         task.data=$data \
#         task.label_dir=$data \
#         task.tokenizer_bpe_model=$bpe_model \
#         model.w2v_path=${PRETRAINED_MODEL_PATH} \
#         common.user_dir=${PWD} \
#         task.noise_wav=null \
#         task.noise_prob=0.0 \
#         criterion.warmup=true \
#         hydra.run.dir=/home/litsub09/projects/visg_avhubert/visg \
#         task.viseme_dir=$data \
#         dataset.num_workers=12 &&
#     echo "finished : $(date '+%Y-%m-%d %H:%M:%S')" > /home/litsub09/projects/visg_avhubert/visg/finish.txt
# fi

# if [ -f "/home/litsub09/projects/visg_avhubert/visg/s2s/total_wer.txt" ]; then
#     echo "=== [SKIP infer] visg already has total_wer.txt ==="
# else
#     ${AV_HUBERT}/infer_all.sh /home/litsub09/projects/visg_avhubert/visg mix
# fi


# #avhubert

# #3번째 base with noise
# if [ -f "/home/litsub09/projects/visg_avhubert/base_noise/finish.txt" ]; then
#     echo "=== [SKIP train] base_noise already finished ==="
# else
#     fairseq-hydra-train \
#         --config-dir ${AV_HUBERT}/conf/av-finetune \
#         --config-name base_noise_pt_noise_ft_433h.yaml \
#         task.data=$data \
#         task.label_dir=$data \
#         task.tokenizer_bpe_model=$bpe_model \
#         model.w2v_path=${PRETRAINED_MODEL_PATH} \
#         common.user_dir=${PWD} \
#         task.noise_wav=/data/DB/musan/tsv/all \
#         task.noise_prob=0.25 \
#         hydra.run.dir=/home/litsub09/projects/visg_avhubert/base_noise \
#         dataset.num_workers=12 \
#         model.layerdrop=0.0 \
#         task.noise_num=1 \
#         distributed_training.distributed_world_size=1 \
#         distributed_training.nprocs_per_node=1 &&
#     echo "finished : $(date '+%Y-%m-%d %H:%M:%S')" > /home/litsub09/projects/visg_avhubert/base_noise/finish.txt
# fi

# if [ -f "/home/litsub09/projects/visg_avhubert/base_noise/s2s/total_wer.txt" ]; then
#     echo "=== [SKIP infer] base_noise already has total_wer.txt ==="
# else
#     ${AV_HUBERT}/infer_all.sh /home/litsub09/projects/visg_avhubert/base_noise mix
# fi


# #4번째 base without noise
# if [ -f "/home/litsub09/projects/visg_avhubert/base/finish.txt" ]; then
#     echo "=== [SKIP train] base already finished ==="
# else
#     fairseq-hydra-train \
#         --config-dir ${AV_HUBERT}/conf/av-finetune \
#         --config-name base_noise_pt_noise_ft_433h.yaml \
#         task.data=$data \
#         task.label_dir=$data \
#         task.tokenizer_bpe_model=$bpe_model \
#         model.w2v_path=${PRETRAINED_MODEL_PATH} \
#         common.user_dir=${PWD} \
#         task.noise_wav=null \
#         task.noise_prob=0.0 \
#         hydra.run.dir=/home/litsub09/projects/visg_avhubert/base \
#         dataset.num_workers=6 \
#         model.layerdrop=0.0 \
#         task.noise_num=0 \
#         distributed_training.distributed_world_size=4 \
#         distributed_training.nprocs_per_node=4 \
#         distributed_training.distributed_port=0 \
#         dataset.max_tokens=2000 \
#         optimization.update_freq=[1] &&
#     echo "finished : $(date '+%Y-%m-%d %H:%M:%S')" > /home/litsub09/projects/visg_avhubert/base/finish.txt
# fi

# if [ -f "/home/litsub09/projects/visg_avhubert/base/s2s/total_wer.txt" ]; then
#     echo "=== [SKIP infer] base already has total_wer.txt ==="
# else
#     ${AV_HUBERT}/infer_all.sh /home/litsub09/projects/visg_avhubert/base mix
# fi
