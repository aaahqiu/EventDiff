# EventDiff: **Process-Conditioned Diffusion for Forecasting Melt-Pool Dynamics via Event-based Sensing in Laser Wire Directed Energy Deposition**

Accurate forecasting of melt-pool evolution is an important step from process
monitoring toward predictive control in laser wire directed energy deposition
(LW-DED). EventDiff is a latent diffusion framework for forecasting future
melt-pool dynamics from event-based observations. It combines the recent event
history with the time-aligned laser-power trajectory so that the prediction
reflects both the current process state and the imposed operating condition.

The framework consists of EventVAE for compact representation of sparse event
voxels and EventDiff for latent-space sequence forecasting. It is designed for
steady deposition as well as transient conditions such as laser activation,
deactivation, and power increase or decrease. The predicted representations
can also be used by downstream models, including deposited-track height
prediction.

## Method

The asynchronous event stream is first consolidated into a sequence of fixed
size event voxels. A signed-log transform compresses the heavy-tailed voxel
values before encoding. EventVAE preserves the spatial organization of the
melt-pool response in a compact latent map; after pretraining, its parameters
are frozen and the posterior mean is used as the deterministic event
representation.

EventDiff models the future sequence in this latent space. The laser-power
curve over the observation and prediction interval describes the external
process input, while the recent event latents describe the melt-pool state from
which the forecast begins. A process-conditioned 3D U-Net is trained with an
EDM objective to denoise the perturbed future latents using both sources of
information. The predicted latents are finally decoded by EventVAE to recover
the future event-voxel sequence.

When initial frames inside the prediction interval have already been measured,
they can be incorporated as a prefix constraint during inference to maintain
continuity with the observed evolution. The remaining sequence is still
generated probabilistically to represent unobserved process variability.

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
