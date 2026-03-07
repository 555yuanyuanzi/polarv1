# Dataset And TensorBoard Guide

## 1. GoPro Source

This V1 package uses the GoPro deblurring dataset released by Seungjun Nah et al.

Recommended sources:

- Official dataset page: <https://seungjunnah.github.io/Datasets/gopro.html>
- Official Hugging Face mirror: <https://huggingface.co/datasets/snah/GOPRO_Large/tree/main>

For V1 training, use `GOPRO_Large.zip`.
That archive contains the blurry/sharp pairs needed for supervised image deblurring.

## 2. Why A Preparation Script Is Needed

The original GoPro archive is sequence-based, with directories like:

```text
GOPRO_Large/
  train/
    GOPRxxxx_xx_xx/
      blur/
      sharp/
  test/
    GOPRxxxx_xx_xx/
      blur/
      sharp/
```

The V1 loader expects a flattened pair layout:

```text
<root>/
  train/
    blur/
    sharp/
  test/
    blur/
    sharp/
```

So `prepare_gopro.sh` does two things:

- downloads and extracts `GOPRO_Large.zip`
- rebuilds a V1-ready flat layout with symlinks, not copied image files

That keeps disk usage reasonable while matching the current loader contract.

## 3. Prepare The Dataset

From the `v1/` directory:

```bash
chmod +x scripts/prepare_gopro.sh
bash scripts/prepare_gopro.sh /data/datasets/gopro_v1
```

The script creates:

```text
/data/datasets/gopro_v1/
  downloads/
    GOPRO_Large.zip
  raw/
    GOPRO_Large/...
  flat/
    train/
      blur/
      sharp/
    test/
      blur/
      sharp/
```

After it finishes, set [gopro_v1.yaml](/c:/Users/86155/polar_code/v1/configs/gopro_v1.yaml) `data.root_dir` to:

```text
/data/datasets/gopro_v1/flat
```

## 4. Minimal Dataset Sanity Check

Before training, confirm:

- `train/blur` and `train/sharp` both exist
- `test/blur` and `test/sharp` both exist
- image counts match between blur and sharp for both splits
- the YAML path points to the `flat/` directory, not `raw/`

If the loader throws a missing-path error, the usual cause is pointing `data.root_dir` at the extracted sequence root instead of the flattened root.

## 5. Recommended 4xA30 Training Setup

For the current V1 model, a good starting point is:

```yaml
data:
  train_crop_size: 256
  batch_size: 4
  val_batch_size: 1
  num_workers: 4

runtime:
  amp: true
  distributed: true
```

Important:

- `batch_size` is per GPU in the current DDP implementation
- with 4 GPUs and `batch_size: 4`, total train batch becomes `16`

Launch with:

```bash
cd v1
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=4
torchrun --standalone --nproc_per_node=4 train.py --config configs/gopro_v1.yaml
```

## 6. Where TensorBoard Logs Are Written

Every run writes outputs under:

```text
v1/outputs/<exp_name>/<timestamp>/
```

Inside each run:

```text
tensorboard/
train.log
metrics.jsonl
checkpoints/
resolved_config.yaml
```

The TensorBoard event files are written inside the `tensorboard/` subdirectory.

## 7. Start TensorBoard

### View all runs

From the repository root:

```bash
tensorboard --logdir v1/outputs --port 6006 --bind_all
```

Then open:

```text
http://127.0.0.1:6006
```

### View one experiment family

```bash
tensorboard --logdir v1/outputs/polarformer_v1 --port 6006 --bind_all
```

This is cleaner when you want to compare only the `polarformer_v1` runs.

### View one specific run

```bash
tensorboard --logdir v1/outputs/polarformer_v1/<timestamp>/tensorboard --port 6006 --bind_all
```

Use this when you only want one run and no cross-run comparison.

## 8. Remote Server Usage

If training is running on a remote Linux server, start TensorBoard on the server:

```bash
tensorboard --logdir v1/outputs --port 6006 --bind_all
```

Then on your local machine create an SSH tunnel:

```bash
ssh -L 16006:127.0.0.1:6006 <user>@<server>
```

Open locally:

```text
http://127.0.0.1:16006
```

This is usually better than exposing the server port directly.

## 9. What To Watch In TensorBoard

Core curves:

- `train/loss`
- `train/lr`
- `val/raw_psnr`
- `val/raw_ssim`
- `val/ema_psnr`
- `val/ema_ssim`

Router diagnostics:

- `router/mean_confidence`
- `router/top2_entropy`
- `router/expert_usage_e1`
- `router/expert_usage_e2`
- `router/expert_usage_e3`
- `router/expert_usage_e4`

Recommended reading:

- use `ema_psnr` as the main checkpoint quality signal
- watch whether one expert collapses to near-100% usage
- watch whether `router/mean_confidence` stays near zero for the whole run, which would indicate the local polar prior is not contributing much

## 10. W&B Sync

The current V1 integration keeps TensorBoard as the primary logger and mirrors it to W&B.

Config knobs:

- `logging.wandb`
- `logging.wandb_project`
- `logging.wandb_entity`
- `logging.wandb_mode`

Behavior:

- W&B is initialized in `train.py`
- `sync_tensorboard=True` is used
- existing TensorBoard scalar writes continue unchanged
- the main process is the only one that initializes a W&B run

Recommended usage:

- keep `gopro_v1.yaml` on `wandb: true` for formal training
- keep `debug_v1.yaml` on `wandb: false` or `wandb_mode: offline`

## 11. Common Mistakes

- Pointing `data.root_dir` at `raw/GOPRO_Large` instead of `flat/`
- Starting TensorBoard from the wrong directory and seeing no runs
- Using `python train.py` instead of `torchrun` when `distributed: true`
- Setting validation batch too high and hitting OOM on full-resolution test images
- Comparing only raw-model validation and ignoring EMA
