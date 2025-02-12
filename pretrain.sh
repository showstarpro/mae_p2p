source /root/anaconda3/etc/profile.d/conda.sh ;
unset LD_LIBRARY_PATH;

conda activate pami;

cd /lpai ;

git clone -b p2p https://github.com/showstarpro/mae_p2p.git mae_p2p;

cd ./mae_p2p;

data_path=/lpai/dataset/imagenet-1k/0-1-0/

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1  # 可能需要禁用 InfiniBand
export OMP_NUM_THREADS=1  # 避免 CPU 线程冲突

torchrun --nproc_per_node=8 --nnodes=1 main_pretrain.py \
    --batch_size 64 \
    --model mae_vit_base_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 30 \
    --warmup_epochs 5 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path ${data_path} \
    --output_dir /lpai/output/models/p2p_vitb_200_05\
    --log_dir /lpai/output/models/p2p_vitb_200_05 \
    --lamda 0.5 \
    --eps 24 \

torchrun --nproc_per_node=8 --nnodes=1  main_finetune.py \
    --batch_size 128 \
    --model vit_base_patch16 \
    --finetune /lpai/output/models/p2p_vitb_200_05/checkpoint-29.pth \
    --epochs 100 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval --data_path ${data_path} \
    --output_dir /lpai/output/models/p2p_vitb_200_05 _ft\
    --log_dir /lpai/output/models/p2p_vitb_200_05_ft \