# EventDiff

Official research code for **“Process-Conditioned Diffusion for Forecasting
Melt-Pool Dynamics via Event-based Sensing in Laser Wire Directed Energy
Deposition.”**

EventDiff forecasts future melt-pool event-voxel sequences from recent event
observations and a time-aligned laser-power trajectory. It is designed for
predictive monitoring of laser wire directed energy deposition (LW-DED),
including steady deposition, laser activation/deactivation, and power
transitions.

## Method

EventDiff follows a two-stage latent forecasting framework:

1. **EventVAE** encodes sparse event voxels into compact spatial latent maps.
   The pretrained encoder is frozen during diffusion training.
2. **EventDiff** predicts the future latent sequence with an EDM-style
   process-conditioned 3D U-Net, then decodes it back to event voxels.

The denoiser is conditioned jointly on recent event history and the aligned
laser-power curve. At inference time, optional measured prefix frames can be
inserted during early reverse-diffusion steps to anchor the generated sequence
to the observed process while preserving stochasticity in the remaining
future.

The paper uses 3 historical event voxels to forecast 25 future voxels at 40 ms
intervals, corresponding to a 1 s prediction horizon. Experiments cover 60
independent recordings and 12 constant or time-varying laser-power conditions.

## Repository Structure

```text
event_diff/
  autoencoder.py   EventVAE architecture used to load pretrained weights
  codec.py         frozen EventVAE and raw-latent codecs
  data.py          recording discovery, splitting, and sequence loading
  model.py         process-conditioned EventDiff denoiser
  diffusion.py     EDM training objective and sampler
  utils.py         distributed training and checkpoint utilities
  visualize.py     event-sequence visualization
train.py           training and per-epoch validation
test.py            held-out evaluation, sampling, and visualization
train.sh           configurable distributed-training launcher
```

Datasets, checkpoints, cached arrays, and generated results are intentionally
excluded from version control.

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For GPU training, install a PyTorch build compatible with the local CUDA
installation.

## Data and EventVAE Checkpoint

Each recording should use the following layout:

```text
data/voxel_output/
  recording_001/
    voxel_grid.npy               # (T, C, H, W) or (T, H, W, C)
    laser_power_per_frame.npy    # (T,)
  recording_002/
    ...

checkpoints/eventvae/
  best.pt
```

These relative locations are the defaults. Custom locations can be configured
without modifying the source:

```bash
export EVENTDIFF_DATA_DIR=/path/to/voxel_output
export EVENTDIFF_VAE_CHECKPOINT=/path/to/eventvae/best.pt
```

Recordings are divided into training, validation, and test subsets before
window construction, preventing windows from one deposition track from leaking
across subsets. The selected split is stored in the training output directory.

## Training

```bash
python train.py \
  --data-dir "$EVENTDIFF_DATA_DIR" \
  --vae-checkpoint "$EVENTDIFF_VAE_CHECKPOINT" \
  --output-dir outputs/eventdiff \
  --cond-frames 3 \
  --pred-frames 25 \
  --batch-size 1 \
  --epochs 50
```

Training supports mixed precision, classifier-free guidance, checkpoint resume,
and single-node distributed data parallelism. A validation pass is performed
after every epoch; `last.pt` stores the latest state and `best.pt` stores the
checkpoint with the lowest validation loss.

For multi-GPU training, expose the desired devices and set one process per
visible GPU:

```bash
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 bash train.sh
```

The batch size is per process, so the effective global batch size is
`batch_size × NPROC_PER_NODE`.

## Evaluation

```bash
python test.py \
  --checkpoint outputs/eventdiff/best.pt \
  --output-dir outputs/eventdiff/test \
  --num-samples 8 \
  --num-steps 25 \
  --guidance-scale 1.5
```

Evaluation uses the held-out recordings stored in the checkpoint. It reports
normalized and original-domain errors, activity statistics, and optionally
saves compressed predictions, comparison images, and videos. Prefix-guided
sampling is controlled with `--known-future-frames` and
`--event-guided-steps`.

## Paper Configuration

The main settings used in the paper are:

- event voxels with 3 temporal channels at `512 × 512` resolution;
- 3 observation frames and 25 prediction frames;
- frozen EventVAE posterior means with `16 × 64 × 64` latent maps;
- AdamW, mixed precision, gradient clipping, and validation-based checkpoint
  selection;
- 25-step Euler sampling with a Karras noise schedule and classifier-free
  guidance.

Hardware capacity and dataset locations are environment-specific; adjust batch
size and worker count without editing source files.

## Citation

If this repository is useful in your research, please cite:

```bibtex
@article{tan2026eventdiff,
  title   = {Process-Conditioned Diffusion for Forecasting Melt-Pool Dynamics
             via Event-based Sensing in Laser Wire Directed Energy Deposition},
  author  = {Tan, Kunpeng and Yang, Ao and Zhang, Huaqing and Yuan, Runze and
             Wang, Chenxi and Zhang, Xingwu and Zhao, Zhibin},
  year    = {2026},
  note    = {Preprint}
}
```
