data_path=/lpai/dataset/imagenet-1k/0-1-0/
export CUDA_VISIBLE_DEVICES=5,6,7,0

python -m torch.distributed.launch  --nproc_per_node=4 main_pretrain.py \
    --batch_size 64 \
    --model mae_vit_base_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 300 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path ${data_path} \
    --output_dir ./p2p_vitb_100\
    --log_dir ./p2p_vitb_100 \
    --lamda 1.5 \
    # --distributed

# OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 main_finetune.py \
#     --accum_iter 2 \
#     --batch_size 64 \
#     --model vit_base_patch16 \
#     --finetune /home/aiscuser/mae_p2p_3_11/mae/p2p_vitb_15/checkpoint-299.pth \
#     --epochs 100 \
#     --blr 5e-4 --layer_decay 0.65 \
#     --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
#     --dist_eval --data_path ${data_path} \
#     --output_dir ./p2p_vitb_15_ft\
#     --log_dir ./p2p_vitb_15_ft \