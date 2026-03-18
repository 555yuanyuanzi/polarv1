# Hybrid Decoder-Heavy Deblurring V1

This directory is a self-contained implementation of the V1 research model:

- Hybrid U-shaped backbone
- `NAFBlock` in encoder and full-resolution decoder
- `RestormerLiteBlock` in bottleneck and mid/high-resolution decoders
- `FBEB` in `decoder3` and `decoder2` for global frequency-band enhancement
- `LocalRefinementBlock` in `decoder3` and `decoder2` for shared local detail repair

## Layout

- `train.py`: training entrypoint
- `eval.py`: checkpoint evaluation entrypoint
- `configs/`: YAML experiment configs
- `scripts/`: operational helpers such as dataset preparation
- `docs/`: setup and usage guides
- `src/config.py`: config schema and YAML loading
- `src/data/`: GoPro dataset
- `src/engine/`: training, evaluation, EMA, checkpoint, optimizer
- `src/models/`: model stack
- `src/utils/`: logging, metrics, seed, experiment, distributed helpers

## Quick Start

Update `data.root_dir` in `configs/gopro_fbeb.yaml`, then run:

```bash
python v1/train.py --config v1/configs/gopro_fbeb.yaml
python v1/eval.py --config v1/configs/gopro_fbeb.yaml --checkpoint path/to/best_psnr.pth --use-ema
```

For dataset preparation, 4-GPU launch recommendations, and TensorBoard usage, see:

- `docs/DATASET_AND_TENSORBOARD.md`

If `logging.wandb=true`, training also mirrors TensorBoard scalars to W&B via `sync_tensorboard=True`.

`configs/debug_fbeb.yaml` is configured as a smoke test: 10 train steps, then one eval pass and checkpoint save.

## Environment

Base Python package requirements are listed in `requirements.txt`.

For a fast Linux setup with `uv`, run:

```bash
cd v1
bash uv_setup.sh cpu
```

Optional arguments:

- first argument: `cpu`, `cu121`, or `cu124`
- second argument: Python version, default `3.11`

Example:

```bash
cd v1
bash uv_setup.sh cu121 3.11
```

## Notes

- This package does not import root-level `dataset.py`, `utils.py`, or `model_polar.py`.
- `FrequencyLoss` is implemented but disabled by default.
- All outputs are written to `v1/outputs/<exp_name>/<timestamp>/`.
- Legacy `polar/router/experts` modules have been removed from the active code path.
