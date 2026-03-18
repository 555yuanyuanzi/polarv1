# FBEB 代码变更日志

## 目的

本文档记录本轮围绕 `FrequencyBandEnhancementBlock (FBEB)` 做过的代码改动，方便后续维护、回溯和做消融实验。

本轮改动的原则是：

- `FBEB` 作为一个**可选模块**接入
- 默认关闭
- 不静默替换现有 V1 默认训练路径
- 通过 YAML 显式开启后才参与前向

## 改动总览

本轮新增或修改的文件如下：

- `v1/src/models/fbeb.py`
- `v1/src/models/network.py`
- `v1/src/config.py`
- `v1/src/models/__init__.py`
- `v1/train.py`
- `v1/eval.py`
- legacy baseline configs removed in the current cleanup pass
- `v1/docs/FREQUENCY_BAND_ENHANCEMENT_BLOCK.md`

## 1. 新增文件

### `v1/src/models/fbeb.py`

新增 `FrequencyBandEnhancementBlock` 模块，实现了：

- 输入 `B x C x H x W` 的 2D 特征图
- `LayerNorm2d`
- `float32` FFT
- 可学习三频带划分参数：
  - `r1`
  - `r2`
  - `tau`
- soft radial masks：
  - `low`
  - `mid`
  - `high`
- 三路逆 FFT 回空间域
- 三路独立 `SE` 调制
- `concat -> 1x1 conv -> 3x3 conv`
- 残差回注：
  - `identity + gamma * fused`

同时增加了运行时统计接口：

- `get_last_band_stats()`

当前会记录：

- `r1`
- `r2`
- `tau`
- `low_energy`
- `mid_energy`
- `high_energy`

### `v1/docs/FREQUENCY_BAND_ENHANCEMENT_BLOCK.md`

新增中文设计文档，定义了：

- 模块定位
- 输入输出 shape
- `r1 / r2 / tau` 参数化方式
- soft mask 公式
- 插入位置建议
- 推荐消融顺序

## 2. 修改文件

### `v1/configs/gopro_v1_fbeb_dec23.yaml`

新增正式训练配置，专门用于：

- `FBEB` 开启
- 只插入在：
  - `decoder3`
  - `decoder2`

对应设置：

```yaml
model:
  fbeb_enabled: true
  fbeb_stages: ["decoder3", "decoder2"]
```

### `v1/configs/debug_v1_fbeb_dec23.yaml`

新增对应的 debug 配置，用于：

- `10 step` 烟雾测试
- `FBEB(decoder3 + decoder2)` 版本

### `v1/src/models/network.py`

在主网络 `PolarFormer` 中加入了 `FBEB` 的可选接入逻辑。

新增构造参数：

- `fbeb_enabled`
- `fbeb_stages`
- `fbeb_init_r1`
- `fbeb_init_r2`
- `fbeb_init_tau`

新增三个可选模块位：

- `self.bottleneck_fbeb`
- `self.decoder3_fbeb`
- `self.decoder2_fbeb`

当前接入顺序为：

- `stage_base -> FBEB -> LPEB`

也就是说：

- bottleneck:
  - `bottleneck_base -> bottleneck_fbeb -> bottleneck_lpeb`
- decoder3:
  - `decoder3_base -> decoder3_fbeb -> decoder3_lpeb`
- decoder2:
  - `decoder2_base -> decoder2_fbeb -> decoder2_lpeb`

默认情况下：

- `fbeb_enabled = false`
- `fbeb_stages = []`

因此现有默认行为不变。

### `v1/src/config.py`

在 `ModelConfig` 中新增配置项：

- `fbeb_enabled: bool = False`
- `fbeb_stages: list[str] = []`
- `fbeb_init_r1: float = 0.22`
- `fbeb_init_r2: float = 0.58`
- `fbeb_init_tau: float = 0.05`

在 `validate_config()` 中新增校验：

- `fbeb_stages` 只能取：
  - `bottleneck`
  - `decoder3`
  - `decoder2`
- 若 `fbeb_enabled=true`，则 `fbeb_stages` 不能为空

### `v1/src/models/__init__.py`

新增导出：

- `FrequencyBandEnhancementBlock`

以便后续模块级导入或单独测试。

### `v1/train.py`

更新 `build_model(config)`，将新配置项传入 `PolarFormer`：

- `fbeb_enabled`
- `fbeb_stages`
- `fbeb_init_r1`
- `fbeb_init_r2`
- `fbeb_init_tau`

这意味着训练入口已经支持通过 YAML 开关 `FBEB`。

### `v1/eval.py`

更新 `build_model(config)`，和训练入口保持一致，同步支持：

- `fbeb_enabled`
- `fbeb_stages`
- `fbeb_init_r1`
- `fbeb_init_r2`
- `fbeb_init_tau`

这样评估时能够正确重建带 `FBEB` 的模型结构。

### Legacy Baseline Config

在 `model:` 下新增默认配置：

```yaml
fbeb_enabled: false
fbeb_stages: []
fbeb_init_r1: 0.22
fbeb_init_r2: 0.58
fbeb_init_tau: 0.05
```

默认关闭，不影响现有正式训练配置。

### Legacy Debug Config

同步新增与正式配置相同的 `FBEB` 配置项，默认同样关闭。

这样 debug 配置也能直接切换到 `FBEB` 版本做冒烟测试。

## 3. 当前实现与最初原型的差异

当前 `FBEB` 和最早的 `fft_fenjie.py` 原型不完全一样，主要差异如下：

- 原型输入更像 `B x C x N`
- 当前实现直接处理 `B x C x H x W`
- 原型使用硬阈值频带切分
- 当前实现使用可学习 `r1 / r2 / tau` 的 soft mask
- 原型只做频带分解
- 当前实现补齐了：
  - 三路 `SE`
  - 融合卷积
  - 残差回注

也就是说，当前版本已经从“实验原型”变成了“可接入主模型训练的模块”。

## 4. 默认行为说明

当前所有改动都遵循：

- 配置默认关闭
- 不影响现有默认 V1 训练
- 只有显式打开时才会启用 `FBEB`

因此：

- 旧实验可以继续按原配置复现
- 新实验可以在 YAML 中单独开启 `FBEB` 做消融

## 5. 当前推荐试验方式

第一轮建议只在 YAML 中这样开启：

```yaml
model:
  fbeb_enabled: true
  fbeb_stages: ["bottleneck"]
  fbeb_init_r1: 0.22
  fbeb_init_r2: 0.58
  fbeb_init_tau: 0.05
```

第二轮再尝试：

```yaml
model:
  fbeb_enabled: true
  fbeb_stages: ["bottleneck", "decoder3"]
```

不建议第一轮就启用：

- `decoder2`
- full-resolution stage

## 6. 后续待做事项

当前这轮只完成了：

- 模块接入
- 配置接入
- 文档沉淀

后续建议继续补：

- 训练日志中记录 `r1 / r2 / tau`
- 记录 `low / mid / high` 三路能量统计
- 做当前 `debug_fbeb.yaml` 前向与训练烟雾测试
- 做 `baseline vs bottleneck FBEB vs bottleneck+decoder3 FBEB` 对照实验
