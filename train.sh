CUDA_VISIBLE_DEVICES=0,2 torchrun \
  --standalone \
  --nproc_per_node=2 \
  train.py \
  --data-dir /data4/tkp/MPH-Data/voxel_output \
  --vae-checkpoint /data4/tkp/Code/EventVAE/outputs_latent64/best.pt \
  --output-dir outputs_motion_ddp \
  --cond-frames 3 \
  --pred-frames 25 \
  --batch-size 2 \
  --num-workers 4 \
  --epochs 50