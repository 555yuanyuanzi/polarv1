# V1 Maintenance Context

## Scope

`v1/` is a self-contained single-image motion deblurring package for GoPro.

Current active architecture focus:

- hybrid U-shaped backbone
- `NAFBlock` in encoder and full-resolution decoder
- `RestormerLiteBlock` in bottleneck and mid-resolution decoders
- `FBEB` in `decoder3` and `decoder2`
- `LocalRefinementBlock` after `FBEB` in `decoder3` and `decoder2`

The old `Local Polar -> Router -> Experts` path is no longer part of the active model and has been removed from the code path.

This directory must not depend on root-level `dataset.py`, `utils.py`, or `model_polar.py`.

## Directory Contract

Keep responsibilities split:

- `train.py`: training entrypoint only
- `eval.py`: checkpoint evaluation only
- `configs/`: runtime parameters
- `scripts/`: operational helpers
- `docs/`: setup and architecture notes
- `src/config.py`: schema, YAML loading, validation
- `src/data/`: dataset logic
- `src/models/`: model code
- `src/engine/`: trainer, evaluator, checkpoint, optimizer
- `src/utils/`: logging, metrics, distributed helpers

## Active Model Definition

Input:

- `B x 3 x H x W`
- training crop is typically `256 x 256`
- `H` and `W` must be divisible by `8`

Backbone:

- Patch embed: `3x3 Conv(3 -> 48)`
- Encoder1: `3 x NAFBlock(48)`
- Down1 -> `96`
- Encoder2: `4 x NAFBlock(96)`
- Down2 -> `192`
- Encoder3: `6 x NAFBlock(192)`
- Down3 -> `384`
- Bottleneck: `3 x RestormerLiteBlock(384)`
- Decoder3: `Up3 + Fuse3 + 2 x RestormerLiteBlock(192) + FBEB(192) + LocalRefinementBlock(192)`
- Decoder2: `Up2 + Fuse2 + 2 x RestormerLiteBlock(96) + FBEB(96) + LocalRefinementBlock(96)`
- Decoder1: `Up1 + Fuse1 + 3 x NAFBlock(48)`
- Output: `3x3 Conv(48 -> 3) + global residual`

Current recommended placement:

- `FBEB` in `decoder3` and `decoder2`
- `LocalRefinementBlock` in `decoder3` and `decoder2`
- no frequency-domain skip connection yet

## FBEB Contract

`FBEB` is a global frequency-band enhancement block, not an explicit prior estimator.

Input:

- `X in R^(B x C x H x W)`

Core steps:

1. `LayerNorm2d`
2. `FFT2` in `float32`
3. learnable radial soft masks:
   - low
   - mid
   - high
4. inverse FFT to spatial domain
5. per-band SE modulation
6. `concat`
7. `1x1 Conv`
8. `3x3 Conv`
9. residual add with learnable scaling

The mask parameters are stage-local and learnable:

- `r1`
- `r2`
- `tau`

They must stay bounded through parameterization.

## LocalRefinementBlock Contract

`LocalRefinementBlock` is the shared local detail restoration block.

Structure:

- `LayerNorm2d`
- `3x3 DWConv`
- `1x1 Conv`
- `GELU`
- `5x5 DWConv`
- `1x1 Conv`
- residual add with learnable scaling

Its role is:

- local edge refinement
- local texture repair
- complement `FBEB` global frequency modulation

Do not turn it into a routed expert block unless that research claim is explicitly revived.

## Config Contract

All runtime parameters must come from YAML.

Primary configs currently in use:

- `configs/gopro_fbeb.yaml`
- `configs/debug_fbeb.yaml`

`src/config.py` remains the only place for:

- schema definition
- YAML loading
- validation

Do not hardcode experiment parameters in model or trainer code.

## Training Contract

Current training behavior:

- `CharbonnierLoss` enabled by default
- `FrequencyLoss` implemented but optional
- `AdamW`
- `CosineAnnealingLR`
- optional AMP
- EMA validation path
- structured JSONL metrics
- TensorBoard logging
- checkpoint resume with full state

Checkpoint payload must include:

- `model`
- `ema`
- `optimizer`
- `scheduler`
- `scaler`
- `epoch`
- `global_step`
- `best_psnr`
- `best_ssim`
- `config`

## Logging Contract

Outputs live under:

- `v1/outputs/<exp_name>/<timestamp>/`

Required artifacts:

- `train.log`
- `metrics.jsonl`
- `resolved_config.yaml`
- `tensorboard/`
- `checkpoints/latest.pth`
- `checkpoints/best_psnr.pth`
- `checkpoints/best_ssim.pth`

Current module diagnostics should focus on `FBEB`, not router metrics:

- `fbeb/r1`
- `fbeb/r2`
- `fbeb/tau`
- `fbeb/low_energy`
- `fbeb/mid_energy`
- `fbeb/high_energy`

## Maintenance Rules

- prefer small, reversible changes
- keep model contracts explicit
- keep docs aligned with active architecture
- remove dead code when a research direction is abandoned
- do not silently add back router / experts / local polar
- if a change affects tensor shapes, update docs at the same time
- if a change affects logging names, update trainer and operator docs together

## Validation Checklist

Before considering a change complete:

- run static compile checks over touched files
- run a model forward smoke test on `128x128` and `256x256`
- confirm train/eval/checkpoint loop still works
- confirm checkpoint round-trip restores all states
- if `FBEB` is enabled, confirm `fbeb/*` stats are logged

## Current Non-Goals

Not part of the active architecture unless explicitly requested:

- local polar prior
- top-k router
- expert mixture branch
- full-resolution frequency skip connection
- GAN or perceptual loss by default
- replacing the hybrid backbone family entirely
