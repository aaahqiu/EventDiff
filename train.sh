#!/usr/bin/env bash
set -Eeuo pipefail

DATA_DIR=${EVENTDIFF_DATA_DIR:-data/voxel_output}
VAE_CHECKPOINT=${EVENTDIFF_VAE_CHECKPOINT:-checkpoints/eventvae/best.pt}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/eventdiff}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}

python -m torch.distributed.run \
  --standalone \
  --nproc_per_node="$NPROC_PER_NODE" \
  train.py \
  --data-dir "$DATA_DIR" \
  --vae-checkpoint "$VAE_CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --cond-frames 3 \
  --pred-frames 25 \
  --batch-size 2 \
  --num-workers 0 \
  --epochs 50
