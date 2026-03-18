# Heavier Backbone Changelog

## Purpose

This note records the follow-up change where the backbone is made slightly heavier while keeping the overall architecture unchanged.

The goal is:

- keep the same stage layout
- keep `dim=48`
- avoid changing the number of encoder/decoder stages
- only increase the internal width of existing blocks a little

## What Changed

### 1. `v1/src/models/network.py`

The network now supports configurable internal width for `NAFBlock`:

- `naf_dw_expand`
- `naf_ffn_expand`

These parameters are used in:

- `encoder1`
- `encoder2`
- `encoder3`
- `decoder1`
- `Fuse` blocks

So the backbone can be made a little heavier without changing stage counts.

### 2. `v1/src/config.py`

New model config fields:

- `naf_dw_expand: int = 2`
- `naf_ffn_expand: int = 2`

Validation was also added:

- both must be positive
- `restormer_ffn_expansion` must stay positive

### 3. `v1/train.py`

`build_model(config)` now forwards:

- `naf_dw_expand`
- `naf_ffn_expand`

### 4. `v1/eval.py`

`build_model(config)` now also forwards:

- `naf_dw_expand`
- `naf_ffn_expand`

This keeps train/eval model construction aligned.

### 5. Legacy Baseline Config

Default backbone width fields were added explicitly:

```yaml
model:
  restormer_ffn_expansion: 2.0
  naf_dw_expand: 2
  naf_ffn_expand: 2
```

This keeps the default V1 behavior unchanged.

### 6. Legacy Debug Config

The same default fields were added for clarity:

```yaml
model:
  restormer_ffn_expansion: 2.0
  naf_dw_expand: 2
  naf_ffn_expand: 2
```

### 7. New heavier configs

Two new configs were added:

- `v1/configs/gopro_fbeb.yaml`
- `v1/configs/debug_fbeb.yaml`

They define a slightly heavier backbone:

```yaml
model:
  restormer_ffn_expansion: 2.5
  naf_dw_expand: 2
  naf_ffn_expand: 3
  fbeb_enabled: true
  fbeb_stages: ["decoder3", "decoder2"]
```

## Design Intent

This is intentionally a small increase, not a redesign.

What stays unchanged:

- stage counts
- `dim=48`
- hybrid backbone layout
- `FBEB` placement logic

What becomes a bit heavier:

- NAF feed-forward width
- RestormerLite feed-forward width

## Recommended Usage

If you want to test the slightly heavier version first with smoke testing:

```bash
python train.py --config configs/debug_fbeb.yaml
```

If smoke testing passes, then use:

```bash
torchrun --standalone --nproc_per_node=4 train.py --config configs/gopro_fbeb.yaml
```
