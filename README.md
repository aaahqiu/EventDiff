# EventDiff

EventDiff is a compact event-voxel future-sequence diffusion implementation for
recording folders under `/data4/tkp/MPH-Data/voxel_output`.
Each folder is expected to contain `voxel_grid.npy` and
`laser_power_per_frame.npy`.

It follows the useful parts of E-Motion for this data:

- condition on a short event history and generate future event frames;
- encode the aligned laser-power curve as an additional diffusion condition;
- train with the E-Motion/EDM log-normal sigma objective;
- concatenate a repeated condition latent to noisy target latents;
- support classifier-free guidance and event-guided known-prefix sampling.

It also keeps the EventVAE data specializations:

- memory-mapped `.npy` loading for large recordings;
- recording-level train/val/test split;
- signed-log normalization for heavy-tailed event values;
- default frozen spatial EventVAE latent codec from
  `/data4/tkp/Code/EventVAE/outputs_latent64/best.pt`.

## Files

- `event_diff/data.py`: MPH voxel discovery, splitting, mmap sequence dataset.
- `event_diff/autoencoder.py`: EventVAE architecture for checkpoint loading.
- `event_diff/codec.py`: frozen VAE or raw normalized latent codec.
- `event_diff/model.py`: spatio-temporal 3D UNet denoiser.
- `event_diff/diffusion.py`: E-Motion-style EDM loss and sampler.
- `event_diff/visualize.py`: signed event visualization and AVI export.
- `train.py`: training entry point.
- `test.py`: sampling/evaluation with `.npz`, image, and video outputs.

## Train

```bash
python train.py \
  --data-dir /data4/tkp/MPH-Data/voxel_output \
  --vae-checkpoint /data4/tkp/Code/EventVAE/outputs_latent64/best.pt \
  --output-dir outputs_motion \
  --cond-frames 3 \
  --pred-frames 25 \
  --use-laser-power \
  --laser-power-max 200 \
  --batch-size 1 \
  --epochs 50
```

Single-node multi-GPU training with `torchrun`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --standalone \
  --nproc_per_node=4 \
  train.py \
  --data-dir /data4/tkp/MPH-Data/voxel_output \
  --vae-checkpoint /data4/tkp/Code/EventVAE/outputs_latent64/best.pt \
  --output-dir outputs_motion_ddp \
  --cond-frames 3 \
  --pred-frames 25 \
  --use-laser-power \
  --laser-power-max 200 \
  --batch-size 1 \
  --num-workers 4 \
  --epochs 50
```

`--batch-size` is per process/GPU. The effective batch size is
`batch-size * nproc_per_node`.

For a lighter smoke run:

```bash
python train.py --epochs 1 --max-windows-per-file 8 --batch-size 1 --num-workers 0
```

Laser-power conditioning is enabled by default. Add `--no-use-laser-power`
to train the older event-history-only model. The power values are normalized
from `[--laser-power-min, --laser-power-max]` to `[-1, 1]`; the current MPH
data spans `0` to `200`.

## Test And Visualize

```bash
python test.py \
  --checkpoint outputs_motion/best.pt \
  --output-dir outputs_motion/test \
  --num-samples 8 \
  --num-steps 25 \
  --guidance-scale 1.5
```

Visualization defaults to a light background with automatic percentile clipping,
which is easier to inspect for sparse event frames. To force the old dark fixed
scale style, add:

```bash
--vis-theme dark --vis-clip-mode fixed --vis-clip-value 100
```

Outputs include metrics JSON, compressed `.npz` condition/target/prediction
pairs, comparison PNGs, and side-by-side AVI videos.
