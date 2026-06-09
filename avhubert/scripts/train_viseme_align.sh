PRETRAINED_MODEL_PATH=/data/DB/large_vox_iter5.pt
result=${result}
ROOT=$(dirname "$(dirname "$(readlink -fm "$0")")")
AV_HUBERT=${ROOT}

export PYTHONPATH="/home/dan/projects/visg_avhubert/fairseq:$PYTHONPATH"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
data=/data/DB/lrs3/433h_data

if [ -f "${result}/finish.txt" ]; then
    echo "=== [SKIP train] v_only_ctc_vsm already finished ==="
else
    fairseq-hydra-train \
        --config-dir ${AV_HUBERT}/conf/av-finetune \
        --config-name v_only_ctc_vsm_large_433h.yaml \
        common.user_dir=${AV_HUBERT} \
        distributed_training.distributed_world_size=1 \
        distributed_training.nprocs_per_node=1 \
        checkpoint.save_interval=1 &&
    echo "finished : $(date '+%Y-%m-%d %H:%M:%S')" > ${result}/finish.txt
fi
