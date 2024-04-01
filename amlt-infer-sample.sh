data_path=~/workspace/data/imagenet

[ -z "${LAMDA}" ] && LAMDA=1
[ -z "${USE_CLUST}" ] && USE_CLUST=True
[ -z "${INFER_DROPOUT}" ] && INFER_DROPOUT=True

if [ ${USE_CLUST} = "True" ]; then
    USE_CLUST_CMD="--use-clustering"
else
    USE_CLUST_CMD=""
fi

if [ ${INFER_DROPOUT} = "True" ]; then
    INFER_DROPOUT_CMD="--infer-dropout"
else
    INFER_DROPOUT_CMD=""
fi

export NCCL_P2P_LEVEL=NVL

echo "=============== prepare data ==============="    
cd ~
wget https://azcopyvnext.azureedge.net/releases/release-10.20.1-20230809/azcopy_linux_amd64_10.20.1.tar.gz
tar -zxvf azcopy_linux_amd64_10.20.1.tar.gz
mkdir workspace/data && cd workspace/data
../../azcopy_linux_amd64_10.20.1/azcopy copy "https://msralaphilly2.blob.core.windows.net/ml-la/v-yanpeng/others/lhp/imagenet/?sv=2023-01-03&st=2024-03-25T11%3A53%3A08Z&se=2025-03-26T11%3A53%3A00Z&sr=c&sp=racwdlf&sig=e95xFQ
FOaiIHTYF7%2BcmKqyYmBk4JdJBQODrZyI3lDxU%3D" . --recursive

echo "=============== begin inference ==============="
cd ~/workspace/mae_p2p

python -m torch.distributed.launch --nproc_per_node=8 main_pretrain.py \
    --batch_size 64 \
    --accum_iter 4 \
    --model mae_vit_base_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 300 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path ${data_path} \
    --output_dir ./p2p_vitb_${LAMDA}\
    --log_dir ./p2p_vitb_${LAMDA} \
    --lamda 1.5 \
    # --distributed

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 main_finetune.py \
    --accum_iter 2 \
    --batch_size 64 \
    --model vit_base_patch16 \
    --finetune "./p2p_vitb_"${LAMDA}"/checkpoint-299.pth" \
    --epochs 100 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval --data_path ${data_path} \
    --output_dir ./p2p_vitb_${LAMDA}_ft\
    --log_dir ./p2p_vitb_${LAMDA}_ft \