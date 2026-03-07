# V1 Maintenance Context

## 1. Scope

This `v1/` directory is a self-contained research package for single-image motion deblurring on GoPro.

Primary goals:

- train and evaluate a Hybrid Decoder-Heavy PolarFormer V1
- keep the V1 research variable focused on `Local Polar -> Router -> Shared + 4 Experts`
- measure image restoration quality with `PSNR / SSIM`
- preserve a clean experiment loop with YAML config, EMA, TensorBoard, checkpoint resume, and structured logs

This directory must not depend on root-level `dataset.py`, `utils.py`, or `model_polar.py`.

## 2. Directory Contract

`v1/` is intentionally split by responsibility:

- `train.py`: training entrypoint only
- `eval.py`: checkpoint evaluation only
- `configs/`: single source of runtime parameters
- `scripts/`: operational helpers such as dataset preparation
- `docs/`: operator-facing setup and usage notes
- `src/config.py`: dataclass schema, YAML loading, validation, resolved config dump
- `src/data/gopro.py`: GoPro dataset and augmentation
- `src/models/`: all model code
- `src/engine/`: trainer, evaluator, EMA, optimizer, checkpoint logic
- `src/utils/`: distributed setup, logging, metrics, seed, experiment directory helpers

Do not collapse these layers unless there is a strong maintenance reason.

## 3. Frozen V1 Model Definition

The V1 architecture is fixed unless an explicit architecture change is requested.

Backbone:

- Input: `B x 3 x H x W`, with `H` and `W` divisible by `8`
- Patch embed: `3x3 Conv(3 -> 48)`
- Encoder1: `3 x NAFBlock(48)`
- Down1 -> `96`
- Encoder2: `4 x NAFBlock(96)`
- Down2 -> `192`
- Encoder3: `6 x NAFBlock(192)`
- Down3 -> `384`
- Bottleneck: `3 x RestormerLiteBlock(384) + LPEB(384)`
- Decoder3: `Up3 + Fuse3 + 2 x RestormerLiteBlock(192) + LPEB(192)`
- Decoder2: `Up2 + Fuse2 + 2 x RestormerLiteBlock(96) + LPEB(96)`
- Decoder1: `Up1 + Fuse1 + 3 x NAFBlock(48)`
- Output: `3x3 Conv(48 -> 3) + global residual`

Do not:

- add global polar branches
- move LPEB into full-resolution decoder
- change `topk`
- change expert count
- silently replace the hybrid backbone

If one of these changes is needed, confirm the new research claim first.

## 4. LPEB Contract

`LPEB` is the core V1 research block.

Input:

- `X in R^(B x C x Hs x Ws)`

Forward order is fixed:

1. `Xn = LayerNorm2d(X)`
2. `F_base = SharedExpert(Xn)`
3. `D_local, c = LocalPolarPrior(Xn)`
4. `alpha, alpha_up, c_up = TopKRouter(D_local, c, Xn)`
5. `F1 = ShortIsoExpert(Xn)`
6. `F2 = LongIsoExpert(Xn)`
7. `F3 = ShortAnisoExpert(Xn, D_local)`
8. `F4 = LongAnisoExpert(Xn, D_local)`
9. `F_local = alpha_up[:,0:1]*F1 + alpha_up[:,1:2]*F2 + alpha_up[:,2:3]*F3 + alpha_up[:,3:4]*F4`
10. `F_mix = F_base + c_up * F_local`
11. `Y1 = X + F_mix`
12. `Y = Y1 + GDFN(LayerNorm2d(Y1))`

Any refactor must preserve this functional contract.

## 5. Local Polar Contract

`LocalPolarPrior` is the only explicit direction prior source in V1.

Fixed hyperparameters:

- `window_size = 8`
- `n_theta = 16`
- `n_r = 8`
- `polar_proj_dim = 32`

Expected shapes:

- input: `Xn in R^(B x C x Hs x Ws)`
- projected: `Fp in R^(B x 32 x Hs x Ws)`
- windowed: `B*Nw x 32 x 8 x 8`
- FFT magnitude: `B*Nw x 32 x 8 x 8`
- polar resample: `B*Nw x 32 x 8 x 16`
- direction map: `D_local in R^(B x 16 x Hg x Wg)`
- confidence map: `c in R^(B x 1 x Hg x Wg)`

Implementation rules:

- use non-overlapping `8x8` windows
- run FFT in `float32`
- use inscribed-circle normalized radius
- aggregate over `channel` and `radius`
- apply `softmax` over `theta`
- compute confidence from normalized entropy

Do not average away the `theta` axis.

## 6. Router Contract

`TopKRouter` converts local polar priors into local expert weights.

Inputs:

- `D_local in R^(B x 16 x Hg x Wg)`
- `c in R^(B x 1 x Hg x Wg)`
- `S = Conv1x1(AvgPool8(Xn)) in R^(B x 16 x Hg x Wg)`

Router input:

- `R_in = concat(D_local, c, S) in R^(B x 33 x Hg x Wg)`

Router body:

- `1x1 Conv 33 -> 32`
- `GELU`
- `3x3 DWConv`
- `1x1 Conv 32 -> 4`

Output:

- `alpha in R^(B x 4 x Hg x Wg)` after top-2 masked softmax
- `alpha_up` and `c_up` are nearest-upsampled to `Hs x Ws`

The router only controls the 4 local experts.
It must not gate the shared expert.

## 7. Expert Contract

Experts are fixed to:

- `SharedExpert = 2 x NAFBlock(C)`
- `ShortIsoExpert = 2 x IsoResidualUnit(k=3)`
- `LongIsoExpert = 2 x IsoResidualUnit(k=7)`
- `ShortAnisoExpert = 2 x DirectionalBasisMixer(k=5)`
- `LongAnisoExpert = 2 x DirectionalBasisMixer(k=9)`

`DirectionalBasisMixer` rules:

- direction source is only `D_local`
- `beta = softmax(Conv1x1(16 -> 4)(D_local), dim=1)`
- four basis branches:
  - horizontal
  - vertical
  - main diagonal
  - anti diagonal
- each branch uses masked depthwise convolution
- output is weighted sum plus residual

Do not rename these experts into unrelated semantics.
The V1 interpretation is:

- short vs long range
- isotropic vs anisotropic

## 8. Config Contract

All runtime parameters must come from YAML.

The canonical config file is `configs/gopro_v1.yaml`.
The smoke config is `configs/debug_v1.yaml`.

`src/config.py` is responsible for:

- schema definition
- YAML load
- V1 validation
- resolved config dump

Do not hardcode new experiment parameters directly into model or trainer code.

## 9. Training Contract

Training behavior is fixed to:

- `CharbonnierLoss` enabled by default
- `FrequencyLoss` implemented but disabled by default
- `AdamW`
- `CosineAnnealingLR`
- optional AMP
- EMA validation path
- raw model and EMA model validation
- structured JSONL metrics
- TensorBoard logging
- resume with optimizer, scheduler, scaler, EMA, best metrics, and global step

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

Only the main process may write:

- checkpoints
- `train.log`
- `metrics.jsonl`
- TensorBoard events

## 10. Logging Contract

Experiment outputs live under:

- `v1/outputs/<exp_name>/<timestamp>/`

Required artifacts:

- `train.log`
- `metrics.jsonl`
- `resolved_config.yaml`
- `tensorboard/`
- `checkpoints/latest.pth`
- `checkpoints/best_psnr.pth`
- `checkpoints/best_ssim.pth`
- optional `checkpoints/epoch_XXXX.pth`

Router debug metrics must stay available:

- `router/mean_confidence`
- `router/top2_entropy`
- `router/expert_usage_e1`
- `router/expert_usage_e2`
- `router/expert_usage_e3`
- `router/expert_usage_e4`

## 11. Maintenance Rules

When modifying `v1/`, keep these rules:

- prefer small, local, reversible changes
- preserve file responsibility boundaries
- do not silently change research claims while refactoring
- do not add new model branches without config and documentation updates
- if an implementation detail is unclear, stop and confirm instead of guessing
- if a change affects tensor shapes, update the shape comments or docs at the same time
- if a change affects training state, update checkpoint load/save together
- if a change affects metrics or logging names, update trainer and docs together

## 12. Code Style Rules

- keep code readable over clever
- use explicit names for tensor roles
- keep tensor shape assumptions close to the relevant module
- avoid broad utility abstractions that hide the model contract
- use ASCII by default unless there is a strong reason otherwise
- add comments only where the logic is not obvious

## 13. Validation Checklist

Before considering a V1 change complete:

- run static compile check over `v1/`
- run module-level shape checks for `LocalPolarPrior`, `TopKRouter`, experts, and `LPEB`
- run full-model forward smoke test on `128x128` and `256x256`
- confirm router top-2 sparsity still holds
- confirm `D_local.sum(dim=1) == 1`
- confirm `c in [0, 1]`
- confirm checkpoint round-trip still restores all states

## 14. Common Non-Goals

These are not part of V1 unless explicitly requested:

- global polar prior
- more than 4 local experts
- full-resolution LPEB
- GAN or perceptual loss by default
- replacing the hybrid backbone with a fully different family
- folding root-level legacy code back into `v1/`
