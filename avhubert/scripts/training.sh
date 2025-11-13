PRETRAINED_MODEL_PATH=/home/aristosp/models/noise_pt_base_vox_iter5.pt
ROOT=$(dirname "$(dirname "$(readlink -fm "$0")")")
AV_HUBERT=${ROOT}/avhubert


export PYTHONPATH="${ROOT}/av_hubert/fairseq:$PYTHONPATH"


fairseq-hydra-train \
    --config-dir ${AV_HUBERT}/conf/av-finetune/ \
    --config-name vsm_base_noise_pt_noise_ft_30h.yaml \
    task.data=/home/aristosp/datasets/LRS3/30h_data/ \
    task.label_dir=/home/aristosp/datasets/LRS3/30h_data \
    task.tokenizer_bpe_model=/home/aristosp/datasets/LRS3/spm1000/spm_unigram1000.model \
    model.w2v_path=${PRETRAINED_MODEL_PATH} \
    common.user_dir=${PWD} \
    task.noise_wav=/home/aristosp/datasets/musan/tsv/all/ \
    hydra.run.dir=ctc_20k_1layernonconvex_0.2-2.5db \
    task.viseme_dir=/home/aristosp/datasets/LRS3/30h_data \
    # common.wandb_project="layer0_warmup_roi_convex_0.1_24k_1layer"
