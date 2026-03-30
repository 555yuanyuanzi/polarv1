# 当前模型架构总览

本文档描述 `v1` 当前 active 模型的**真实实现**，不包含已经废弃的 `polar/router/experts` 路线。  
核心代码文件如下：

- [network.py](/c:/Users/86155/polar_code/v1/src/models/network.py)
- [common.py](/c:/Users/86155/polar_code/v1/src/models/common.py)
- [naf.py](/c:/Users/86155/polar_code/v1/src/models/naf.py)
- [restormer_lite.py](/c:/Users/86155/polar_code/v1/src/models/restormer_lite.py)
- [fbeb.py](/c:/Users/86155/polar_code/v1/src/models/fbeb.py)
- [importance.py](/c:/Users/86155/polar_code/v1/src/models/importance.py)
- [local_refine.py](/c:/Users/86155/polar_code/v1/src/models/local_refine.py)

当前主线可以概括为：

**Hybrid U-Net + FBEB + Raw-Guided Importance + Local Refinement**

对应职责：

- 编码器：`NAFBlock`，负责高分辨率局部特征提取
- bottleneck：`RestormerLiteBlock + FBEB`，负责中低分辨率全局建模与频带增强
- decoder3 / decoder2：`RestormerLiteBlock + FBEB + ImportanceHead + LocalRefinement`
- decoder1：`NAFBlock`，负责最终高分辨率细节恢复

---

## 1. 结构一句话

输入模糊图先经过一个 `3x3` 卷积映射到特征空间，然后进入三层编码器；  
中间通过 `RestormerLite` 做上下文恢复，并在 bottleneck / decoder3 / decoder2 上用 `FBEB` 做低中高频增强；  
在 decoder3 / decoder2 上，再通过 `RawGuidancePyramid + RestorationImportanceHead` 预测局部恢复重要性图，用它调制 `LocalRefinementBlock` 的局部修复强度；  
最后通过 `3x3` 输出头恢复 RGB 残差并加回输入图。

---

## 2. 当前默认结构配置

当前 active 默认规模为：

| 项目 | 设置 |
|---|---|
| 基础宽度 | `dim = 48` |
| 编码器层数 | `enc_blocks = [3, 4, 6]` |
| bottleneck 层数 | `bottleneck_base_blocks = 3` |
| decoder3 层数 | `dec3_base_blocks = 3` |
| decoder2 层数 | `dec2_base_blocks = 3` |
| decoder1 层数 | `dec1_base_blocks = 4` |
| Restormer FFN 扩展率 | `restormer_ffn_expansion = 2.0 ~ 2.5` |
| NAF DW 扩展率 | `naf_dw_expand = 2` |
| NAF FFN 扩展率 | `naf_ffn_expand = 2 ~ 3` |

当前 active 功能开关通常为：

| 开关 | 当前主线 |
|---|---|
| `fbeb_enabled` | `true` |
| `fbeb_stages` | `["bottleneck", "decoder3", "decoder2"]` |
| `local_refine_enabled` | `true` |
| `local_refine_stages` | `["decoder3", "decoder2"]` |

因此当前推荐结构是：

- bottleneck：`RestormerLite + FBEB`
- decoder3：`RestormerLite + FBEB + ImportanceHead + LocalRefinement`
- decoder2：`RestormerLite + FBEB + ImportanceHead + LocalRefinement`
- decoder1：`NAFBlock`

---

## 3. 整体张量尺寸

以下以 `256 x 256` 输入为例。

### 输入与输出

| 阶段 | Tensor Shape |
|---|---|
| 输入模糊图 | `B x 3 x 256 x 256` |
| 输出残差图 | `B x 3 x 256 x 256` |
| 最终恢复图 | `output + input` |

### 主干特征尺寸

| 位置 | 结构 | 输出 Shape |
|---|---|---|
| Patch Embed | `3x3 Conv, 3 -> 48` | `B x 48 x 256 x 256` |
| Encoder1 | `NAFBlock x 3` | `B x 48 x 256 x 256` |
| Down1 | `3x3 Conv, stride=2, 48 -> 96` | `B x 96 x 128 x 128` |
| Encoder2 | `NAFBlock x 4` | `B x 96 x 128 x 128` |
| Down2 | `3x3 Conv, stride=2, 96 -> 192` | `B x 192 x 64 x 64` |
| Encoder3 | `NAFBlock x 6` | `B x 192 x 64 x 64` |
| Down3 | `3x3 Conv, stride=2, 192 -> 384` | `B x 384 x 32 x 32` |
| Bottleneck | `RestormerLite x 3 + FBEB` | `B x 384 x 32 x 32` |
| Up3 | `1x1 Conv + PixelShuffle, 384 -> 192` | `B x 192 x 64 x 64` |
| Decoder3 | `Fuse + RestormerLite x 3 + FBEB + Importance + LocalRefine` | `B x 192 x 64 x 64` |
| Up2 | `1x1 Conv + PixelShuffle, 192 -> 96` | `B x 96 x 128 x 128` |
| Decoder2 | `Fuse + RestormerLite x 3 + FBEB + Importance + LocalRefine` | `B x 96 x 128 x 128` |
| Up1 | `1x1 Conv + PixelShuffle, 96 -> 48` | `B x 48 x 256 x 256` |
| Decoder1 | `Fuse + NAFBlock x 4` | `B x 48 x 256 x 256` |
| Output Head | `3x3 Conv, 48 -> 3` | `B x 3 x 256 x 256` |

---

## 4. 总体前向流程

### 4.1 主干拓扑

```text
Input RGB
  -> PatchEmbed(3x3 Conv)
  -> Encoder1 (NAF x3)
  -> Down1
  -> Encoder2 (NAF x4)
  -> Down2
  -> Encoder3 (NAF x6)
  -> Down3
  -> BottleneckBase (RestormerLite x3)
  -> BottleneckFBEB
  -> Up3
  -> Fuse3(enc3, up3)
  -> Decoder3Base (RestormerLite x3)
  -> Decoder3FBEB
  -> Decoder3Importance(raw-guided)
  -> Decoder3LocalRefine
  -> Up2
  -> Fuse2(enc2, up2)
  -> Decoder2Base (RestormerLite x3)
  -> Decoder2FBEB
  -> Decoder2Importance(raw-guided)
  -> Decoder2LocalRefine
  -> Up1
  -> Fuse1(enc1, up1)
  -> Decoder1 (NAF x4)
  -> OutputConv(3x3)
  -> Residual Add Input
```

### 4.2 数学形式

\[
x_0 = \text{PatchEmbed}(I)
\]

\[
enc1 = E_1(x_0), \quad enc2 = E_2(Down_1(enc1)), \quad enc3 = E_3(Down_2(enc2))
\]

\[
b = FBEB_b(Bottleneck(Down_3(enc3)))
\]

\[
dec3_{stage} = D_3(Fuse_3(enc3, Up_3(b)))
\]

\[
dec3_{fbeb} = FBEB_3(dec3_{stage})
\]

\[
D_3 = Importance_3(dec3_{stage}, dec3_{fbeb}, g_3)
\]

\[
dec3 = LocalRefine_3(dec3_{fbeb}, D_3)
\]

\[
dec2_{stage} = D_2(Fuse_2(enc2, Up_2(dec3)))
\]

\[
dec2_{fbeb} = FBEB_2(dec2_{stage})
\]

\[
D_2 = Importance_2(dec2_{stage}, dec2_{fbeb}, g_2)
\]

\[
dec2 = LocalRefine_2(dec2_{fbeb}, D_2)
\]

\[
dec1 = D_1(Fuse_1(enc1, Up_1(dec2)))
\]

\[
\hat{S} = Output(dec1) + I
\]

其中：

- `g2, g3` 由 `RawGuidancePyramid` 从原始模糊图提取
- `D2, D3` 是 importance map

---

## 5. 编码器、bottleneck、解码器的精确定义

## 5.1 Encoder

编码器共三层，全部使用 `NAFBlock`。

### Encoder1

- 输入：`B x 48 x 256 x 256`
- 结构：`NAFBlock x 3`
- 输出：`B x 48 x 256 x 256`

### Encoder2

- 输入前先 `Down1`
- `Down1 = 3x3 Conv(stride=2), 48 -> 96`
- 结构：`NAFBlock x 4`
- 输出：`B x 96 x 128 x 128`

### Encoder3

- 输入前先 `Down2`
- `Down2 = 3x3 Conv(stride=2), 96 -> 192`
- 结构：`NAFBlock x 6`
- 输出：`B x 192 x 64 x 64`

### 编码器职责

- 提供稳定的多尺度空间表征
- 在高分辨率下保留纹理和边缘基础信息
- 为 decoder skip connection 提供恢复所需细节

---

## 5.2 Bottleneck

### 结构

- 输入：`B x 192 x 64 x 64`
- `Down3 = 3x3 Conv(stride=2), 192 -> 384`
- `RestormerLiteBlock x 3`
- `FBEB`

### 输出

\[
B \times 384 \times 32 \times 32
\]

### bottleneck 职责

- 最低分辨率、最大感受野
- 负责全局上下文整合
- 用 `FBEB` 做全局 low / mid / high 频带重整

### 当前不在 bottleneck 上做的事

- 不接 `ImportanceHead`
- 不接 `LocalRefinementBlock`

原因是 bottleneck 分辨率较低，更适合全局频带建模，不适合直接做局部精修。

---

## 5.3 Decoder3

### 结构顺序

1. `Up3`
2. `Fuse3`
3. `RestormerLiteBlock x 3`
4. `FBEB`
5. `RestorationImportanceHead`
6. `LocalRefinementBlock`

### 详细输入输出

- `Up3`: `1x1 Conv + PixelShuffle`, `384 -> 192`, `32 -> 64`
- `Fuse3(enc3, up3)`: 输出 `B x 192 x 64 x 64`
- `Decoder3Base`: `RestormerLiteBlock x 3`
- `Decoder3FBEB`: `B x 192 x 64 x 64 -> B x 192 x 64 x 64`
- `Decoder3Importance`: 输出 `B x 1 x 64 x 64`
- `Decoder3LocalRefine`: 输出 `B x 192 x 64 x 64`

### 职责

- 在中等分辨率下恢复主要结构
- 引入频带增强
- 初步做差异化局部修复

---

## 5.4 Decoder2

### 结构顺序

1. `Up2`
2. `Fuse2`
3. `RestormerLiteBlock x 3`
4. `FBEB`
5. `RestorationImportanceHead`
6. `LocalRefinementBlock`

### 详细输入输出

- `Up2`: `1x1 Conv + PixelShuffle`, `192 -> 96`, `64 -> 128`
- `Fuse2(enc2, up2)`: 输出 `B x 96 x 128 x 128`
- `Decoder2Base`: `RestormerLiteBlock x 3`
- `Decoder2FBEB`: `B x 96 x 128 x 128 -> B x 96 x 128 x 128`
- `Decoder2Importance`: 输出 `B x 1 x 128 x 128`
- `Decoder2LocalRefine`: 输出 `B x 96 x 128 x 128`

### 职责

- 更高分辨率地细化结构
- 放大难区域局部修复
- 为最终 decoder1 输送更干净的恢复特征

---

## 5.5 Decoder1

### 结构顺序

1. `Up1`
2. `Fuse1`
3. `NAFBlock x 4`
4. `3x3 Conv`

### 详细输入输出

- `Up1`: `1x1 Conv + PixelShuffle`, `96 -> 48`, `128 -> 256`
- `Fuse1(enc1, up1)`: 输出 `B x 48 x 256 x 256`
- `Decoder1`: `NAFBlock x 4`
- `Output Head`: `3x3 Conv, 48 -> 3`

### 职责

- 最终高分辨率纹理整理
- 生成 RGB 残差图

---

## 6. 每个模块的“实现细节”

## 6.1 `LayerNorm2d`

文件：
- [common.py](/c:/Users/86155/polar_code/v1/src/models/common.py)

### 实现

自定义 `LayerNormFunction`，在每个空间位置对通道维做归一化：

1. 计算该位置通道均值 `mu`
2. 计算通道方差 `var`
3. 归一化
4. 乘可学习 `weight`
5. 加可学习 `bias`

### 输入输出

- 输入：`B x C x H x W`
- 输出：`B x C x H x W`

---

## 6.2 `SimpleGate`

文件：
- [common.py](/c:/Users/86155/polar_code/v1/src/models/common.py)

### 实现

1. 沿通道把输入一分为二
2. 两半逐元素相乘

如果输入是 `B x 2C x H x W`，输出是 `B x C x H x W`。

---

## 6.3 `NAFBlock`

文件：
- [naf.py](/c:/Users/86155/polar_code/v1/src/models/naf.py)

### 第一段：局部卷积 + 通道注意力

```text
LN
-> 1x1 Conv
-> 3x3 DWConv
-> SimpleGate
-> SCA
-> 1x1 Conv
-> Dropout
-> residual(beta)
```

### 第二段：轻量 FFN

```text
LN
-> 1x1 Conv
-> SimpleGate
-> 1x1 Conv
-> Dropout
-> residual(gamma)
```

### 关键算子

- `conv1`: `1x1`, `C -> C*dw_expand`
- `conv2`: `3x3 DWConv`
- `sca`: `AdaptiveAvgPool2d(1) + 1x1 Conv`
- `conv3`: `1x1`, 压回 `C`
- `conv4`: `1x1`, `C -> C*ffn_expand`
- `conv5`: `1x1`, 压回 `C`

### 残差形式

\[
x = x + \beta \cdot y_1
\]

\[
x = x + \gamma \cdot y_2
\]

其中 `beta, gamma` 为可学习参数，初始为 0。

---

## 6.4 `GDFN`

文件：
- [restormer_lite.py](/c:/Users/86155/polar_code/v1/src/models/restormer_lite.py)

### 实现顺序

```text
1x1 Conv
-> 3x3 DWConv
-> channel split
-> GELU(x1) * x2
-> 1x1 Conv
```

### 关键点

- `project_in`: `C -> 2 * hidden`
- `dwconv`: 对 `2 * hidden` 做 depthwise conv
- 再拆成两半做门控
- `project_out`: `hidden -> C`

---

## 6.5 `MDTA`

文件：
- [restormer_lite.py](/c:/Users/86155/polar_code/v1/src/models/restormer_lite.py)

### 实现顺序

```text
1x1 Conv(qkv)
-> 3x3 DWConv(qkv)
-> split q/k/v
-> reshape heads
-> normalize q/k
-> attention
-> attn @ v
-> reshape back
-> 1x1 Conv
```

### 关键点

- `qkv`: `C -> 3C`
- `qkv_dwconv`: `3x3 depthwise conv`
- 多头数固定依赖通道：
  - `96 -> 2`
  - `192 -> 4`
  - `384 -> 8`

注意这里是 `Restormer` 风格的 `MDTA`，不是标准 ViT 的 token-token 注意力写法。

---

## 6.6 `RestormerLiteBlock`

文件：
- [restormer_lite.py](/c:/Users/86155/polar_code/v1/src/models/restormer_lite.py)

### 实现顺序

```text
x = x + MDTA(LN(x))
x = x + GDFN(LN(x))
```

### 作用

- 中低分辨率下的上下文建模
- 比单纯卷积更强
- 但结构比完整 Restormer 更轻

---

## 6.7 `Downsample`

文件：
- [network.py](/c:/Users/86155/polar_code/v1/src/models/network.py)

### 实现

```text
3x3 Conv, stride=2, padding=1
```

### 作用

- 空间尺寸减半
- 通道数提升到下一层宽度

---

## 6.8 `Upsample`

文件：
- [network.py](/c:/Users/86155/polar_code/v1/src/models/network.py)

### 实现

```text
1x1 Conv
-> PixelShuffle(2)
```

### 作用

- 空间尺寸扩大 2 倍
- 通道映射到 decoder 对应宽度

---

## 6.9 `Fuse`

文件：
- [network.py](/c:/Users/86155/polar_code/v1/src/models/network.py)

### 实现顺序

```text
concat(enc, dec)
-> 1x1 Conv
-> NAFBlock
-> 1x1 Conv
-> channel split
-> left + right
```

### 作用

- 融合 skip 和 decoder 特征
- 避免直接拼接造成通道冲突

---

## 6.10 `SEBranch`

文件：
- [fbeb.py](/c:/Users/86155/polar_code/v1/src/models/fbeb.py)

### 实现顺序

```text
AdaptiveAvgPool2d(1)
-> 1x1 Conv
-> GELU
-> 1x1 Conv
-> sigmoid
-> x * scale
```

### 输入输出

- 输入：单一路频带特征 `B x C x H x W`
- 输出：同 shape

### 作用

- 对 low / mid / high 各自做通道重加权

---

## 6.11 `FBEB`

Update note:
- Current implementation is `FBEB-v2`.
- It keeps the learnable radial `r1 / r2 / tau` split.
- After inverse FFT, each band now passes through a band-specific compensation block.
- The block also predicts sample-level redistribution weights `alpha_low / alpha_mid / alpha_high` before fusion.
- Existing output shape and residual interface stay unchanged.

文件：
- [fbeb.py](/c:/Users/86155/polar_code/v1/src/models/fbeb.py)

### 模块定位

`FBEB` 是一个**频带增强块**，不是显式 blur type 分类器。  
它负责：

- 在频域里把特征分成 low / mid / high 三路
- 分别增强后再融合回主干

### 详细实现顺序

```text
LN
-> FFT2
-> fftshift
-> radial mask build
-> low/mid/high split
-> IFFT(low)
-> IFFT(mid)
-> IFFT(high)
-> SE_low / SE_mid / SE_high
-> concat
-> 1x1 Conv
-> 3x3 Conv
-> residual(gamma)
```

### 频带参数

可学习参数：

- `p1`
- `p2`
- `p3`

映射成：

\[
r_1 = 0.10 + 0.25 \cdot \sigma(p_1)
\]

\[
r_2 = 0.50 + 0.25 \cdot \sigma(p_2)
\]

\[
\tau = 0.03 + 0.07 \cdot \sigma(p_3)
\]

### 频率半径图

在正方形频率平面上构造：

\[
r(i,j) = \sqrt{x^2 + y^2}
\]

再归一化到 `[0,1]`。

### 三路软掩码

\[
M_{low} = \sigma\left(\frac{r_1-r}{\tau}\right)
\]

\[
M_{high} = \sigma\left(\frac{r-r_2}{\tau}\right)
\]

\[
M_{mid} = \sigma\left(\frac{r-r_1}{\tau}\right)\cdot\sigma\left(\frac{r_2-r}{\tau}\right)
\]

然后：

\[
M_{low}, M_{mid}, M_{high}
\]

再按三路和归一化。

### 最终残差形式

\[
y = x + \gamma_f \cdot \Delta_f
\]

其中：

- `Δf` 为三路频带融合输出
- `gamma_f` 为可学习缩放，初始为 0

### 训练期缓存

- `r1`
- `r2`
- `tau`
- `low_energy`
- `mid_energy`
- `high_energy`
- `alpha_low`
- `alpha_mid`
- `alpha_high`
- `low / mid / high` 可视化图

---

## 6.12 `RawGuidancePyramid`

文件：
- [importance.py](/c:/Users/86155/polar_code/v1/src/models/importance.py)

### 实现顺序

```text
Input RGB
-> 3x3 Conv (stem)
-> GELU
-> 3x3 Conv(stride=2)  -> g2
-> GELU
-> 3x3 Conv(stride=2)  -> g3
-> GELU
```

### 输出

- `g2`: 给 decoder2，用于 `128 x 128`
- `g3`: 给 decoder3，用于 `64 x 64`

### 作用

- 让 importance 估计显式看到原始模糊图中的退化线索

---

## 6.13 `RestorationImportanceHead`

文件：
- [importance.py](/c:/Users/86155/polar_code/v1/src/models/importance.py)

### 模块定位

它不是 blur kernel estimator，也不是显式真值图。  
它是一个**局部修复强度控制图预测头**。

输出：

\[
D \in \mathbb{R}^{B \times 1 \times H \times W}
\]

其中：

- `D` 高：该区域更值得加强局部修复
- `D` 低：该区域可以轻修

### 输入

```text
stage_feat : B x C x H x W
fbeb_feat  : B x C x H x W
raw_guidance : B x G x H x W
```

### 实现顺序

```text
concat(stage_feat, fbeb_feat)
-> LayerNorm2d
-> 3x3 Conv
-> GELU
-> raw-guided gating
-> 3x3 DWConv
-> 1x1 Conv -> 1 channel
-> sigmoid
```

### Raw-guided gating

先从 guidance 生成单通道门控图：

\[
A = \sigma(\text{Conv}(g))
\]

然后调制中间特征：

\[
h' = h \odot (1 + A)
\]

### 最终输出

\[
D = \sigma(\text{Conv}_{1 \times 1}(\text{DWConv}(h')))
\]

### 训练期统计

当前记录：

- `importance/mean`
- `importance/std`
- `importance/high_ratio`
- `importance/raw_gate_mean`
- `importance_sup/map_loss`
- `importance_sup/aux_loss`

其中：

\[
importance\_high\_ratio = \text{mean}(D > 0.6)
\]

这只是诊断指标，不是监督标签。

当前实现还支持一个训练期可选的轻量 importance supervision：
- 仅在 `decoder3 / decoder2` 上启用
- 仅增加一个 `3x3 Conv -> RGB residual` 的 stage 预测头
- 用当前 stage 粗恢复残差和少量 `blur-sharp` 先验构造 supervision target
- 推理时不会走这条辅助监督路径

### 训练期可视化

当前缓存：

- `importance`
- `raw_gate`

---

## 6.14 `LocalRefinementBlock`

文件：
- [local_refine.py](/c:/Users/86155/polar_code/v1/src/models/local_refine.py)

### 模块定位

用于在 `FBEB` 之后做共享局部细节精修。

### 实现顺序

```text
LN
-> 3x3 DWConv
-> 1x1 Conv
-> GELU
-> 5x5 DWConv
-> 1x1 Conv
-> residual(gamma)
```

### 局部残差

模块内部先预测：

\[
\Delta_{local} = \gamma_l \cdot f(x)
\]

其中 `gamma_l` 为可学习参数，初始为 0。

### importance 调制

如果提供 importance map：

\[
\Delta_{local}' = \Delta_{local} \odot (1 + \lambda D)
\]

最终输出：

\[
y = x + \Delta_{local}'
\]

### 意义

- `FBEB` 给全局频带增强
- `LocalRefinement` 负责真正做局部边缘/纹理修补
- `ImportanceHead` 决定在哪些位置放大这个局部修补

---

## 7. 三个新增核心模块之间的关系

当前新结构最核心的是这三块：

### 1. `FBEB`

负责：

- 频域 low / mid / high 分解
- 全局频带增强

### 2. `ImportanceHead`

负责：

- 预测空间上的恢复重要性图
- 决定哪里更该重点修

### 3. `LocalRefinementBlock`

负责：

- 执行共享的局部细节精修

### 它们在 decoder 中的顺序

```text
RestormerLite stage
-> FBEB
-> ImportanceHead(raw-guided)
-> LocalRefinement
```

即：

- 先增强频带
- 再判断哪些位置更值得强修
- 最后执行局部修复

---

## 8. 当前训练和可视化里能看到什么

## 8.1 标量统计

### FBEB

- `fbeb/r1`
- `fbeb/r2`
- `fbeb/tau`
- `fbeb/low_energy`
- `fbeb/mid_energy`
- `fbeb/high_energy`
- `fbeb/alpha_low`
- `fbeb/alpha_mid`
- `fbeb/alpha_high`

### Importance

- `importance/mean`
- `importance/std`
- `importance/high_ratio`
- `importance/raw_gate_mean`

## 8.2 TensorBoard 图像

### FBEB 频带图

- `visuals/fbeb/bottleneck_low`
- `visuals/fbeb/bottleneck_mid`
- `visuals/fbeb/bottleneck_high`
- `visuals/fbeb/decoder3_low`
- `visuals/fbeb/decoder3_mid`
- `visuals/fbeb/decoder3_high`
- `visuals/fbeb/decoder2_low`
- `visuals/fbeb/decoder2_mid`
- `visuals/fbeb/decoder2_high`

### Importance 图

- `visuals/importance/decoder3_importance`
- `visuals/importance/decoder3_raw_gate`
- `visuals/importance/decoder2_importance`
- `visuals/importance/decoder2_raw_gate`

---

## 9. 与旧结构的区别

当前 active 模型已经**不再使用**：

- `polar.py`
- `router.py`
- `experts.py`
- `lpeb.py`

因此当前模型不再是：

**Polar Router + Experts**

而是：

**FBEB + Raw-Guided Importance + Shared Local Refinement**

论文/汇报层面的表述也应该相应更新，不要再沿用旧的“polar routing”叙事。

---

## 10. 最终概括

当前这版模型可以概括为：

> 一个 hybrid U-Net 去模糊网络。编码器和最后一层解码器使用 `NAFBlock` 做轻量局部建模；bottleneck、decoder3、decoder2 使用 `RestormerLiteBlock` 做更强的上下文恢复；在 bottleneck / decoder3 / decoder2 上引入 `FBEB` 做低中高频重加权；在 decoder3 / decoder2 上再通过 `RawGuidancePyramid + RestorationImportanceHead` 生成局部恢复重要性图，用它调制 `LocalRefinementBlock` 的局部修复强度，最后输出 RGB 残差并加回输入图。
