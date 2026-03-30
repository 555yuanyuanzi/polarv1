# 频带增强块设计文档

## 目的

本文档定义一个新的频域增强模块，用于当前图像去模糊项目：

- 模块名：`FrequencyBandEnhancementBlock`
- 简称：`FBEB`

Current implementation note:
- The live code has moved to `FBEB-v2`.
- It still uses learnable radial soft masks parameterized by `r1 / r2 / tau`.
- After inverse FFT, each band now goes through a band-specific compensation block.
- The three bands are then reweighted by sample-level `alpha_low / alpha_mid / alpha_high` before fusion.

这个模块的定位是一个**全局频带增强块**，而不是显式的方向先验估计器。

它的目标是：

- 将某一层特征图分解成 `low / mid / high` 三个频带
- 对三路频带分别做轻量通道注意力调制
- 再把三路频带融合回主干特征流
- 在不依赖脆弱角度估计的前提下提升恢复效果

这条路线应该被视为一个**V2 / 消融实验方向**，而不是静默替换现有冻结版 V1 `LPEB` 的实现。

## 为什么考虑这个模块

前面的 local polar 路线存在三个结构性问题：

- 深层 stage 特征图在多次下采样后已经很小
- 再做局部 patch FFT 会让有效空间支持进一步变小
- 显式角度和置信度估计都不稳定

`FBEB` 避开了这些问题：

- 它直接在整张 stage 特征图上工作
- 它不需要显式估计方向角
- 它把频带作为增强线索，而不是作为硬路由信号

## 模块角色

`FBEB` 应该被理解为：

- 一个残差式特征增强块
- 一个全局频率条件化模块

它**不应该**被理解为：

- 模糊类型分类器
- 方向先验提取器
- 专家路由模块的替代物

## 建议的插入位置

在当前 `256 x 256` 训练 crop 和已有 hybrid U 形骨干下，各 stage 大致为：

- bottleneck: `B x 384 x 32 x 32`
- decoder3: `B x 192 x 64 x 64`
- decoder2: `B x 96 x 128 x 128`

推荐插入顺序：

1. 第一版先只在 bottleneck 上使用
2. 第二版再尝试 bottleneck + decoder3
3. 第一版不要上 decoder2

原因：

- bottleneck 的全局结构最干净，计算也最省
- decoder3 仍然适合做 FFT
- decoder2 纹理更重、计算更高，也更容易被内容干扰

## 输入输出约定

输入：

\[
x \in \mathbb{R}^{B \times C \times H \times W}
\]

输出：

\[
y \in \mathbb{R}^{B \times C \times H \times W}
\]

模块保持空间尺寸和通道数不变。

这里不需要：

- flatten
- token 化
- `reshape_to_square()`

因为你当前模型本来就是规则的二维特征图。

## 高层前向流程

模块前向顺序如下：

1. 对输入做归一化
2. 在 `float32` 上执行 2D FFT
3. 构造三张软径向 mask：
   - low
   - mid
   - high
4. 用三张 mask 对频谱做三路分解
5. 对每一路做逆 FFT 回到空间域
6. 每一路分别做一个轻量 `SE` 分支
7. 将三路结果拼接
8. 用 `1x1 conv -> 3x3 conv` 融合
9. 通过残差门控加回输入

形式化表达：

\[
X = \mathcal{F}(Norm(x))
\]

\[
X_l = X \odot m_{low}, \quad X_m = X \odot m_{mid}, \quad X_h = X \odot m_{high}
\]

\[
x_l = \mathcal{F}^{-1}(X_l), \quad x_m = \mathcal{F}^{-1}(X_m), \quad x_h = \mathcal{F}^{-1}(X_h)
\]

\[
z = Concat(SE_l(x_l), SE_m(x_m), SE_h(x_h))
\]

\[
fused = Conv_{3 \times 3}(Conv_{1 \times 1}(z))
\]

\[
y = x + \gamma \cdot fused
\]

其中 `gamma` 是一个可学习的残差缩放参数。

## 软径向频带划分

### 核心思想

不要使用这种硬阈值圆形划分：

- `distance <= 0.3`
- `0.3 < distance < 0.7`
- `distance >= 0.7`

而是使用：

- 两个可学习半径：`r1`, `r2`
- 一个可学习平滑参数：`tau`

这三个参数共同定义归一化半径 `R` 上的平滑频带划分。

### 归一化半径网格

对每个特征尺寸 `H x W`，构造一张半径图：

\[
R(i,j) \in [0,1]
\]

其中：

- `R = 0` 表示 FFT 中心
- `R = 1` 表示到最远角点的归一化半径

### 可学习参数

每个块只学习三个标量参数：

- `p1`
- `p2`
- `p3`

再映射成有界物理参数：

\[
r1 = 0.10 + 0.25 \cdot \sigma(p1)
\]

\[
r2 = 0.50 + 0.25 \cdot \sigma(p2)
\]

\[
\tau = 0.03 + 0.07 \cdot \sigma(p3)
\]

这样可以保证：

- `r1` 始终落在低频区间
- `r2` 始终落在高频分界附近
- `tau` 始终为正且范围稳定

推荐初始化：

- `r1 ≈ 0.22`
- `r2 ≈ 0.58`
- `tau ≈ 0.05`

### 软 mask 公式

定义：

\[
m_{low}(R)=\sigma\left(\frac{r1-R}{\tau}\right)
\]

\[
m_{high}(R)=\sigma\left(\frac{R-r2}{\tau}\right)
\]

\[
m_{mid}(R)=\sigma\left(\frac{R-r1}{\tau}\right)\cdot \sigma\left(\frac{r2-R}{\tau}\right)
\]

然后做归一化：

\[
\hat m_k = \frac{m_k}{m_{low}+m_{mid}+m_{high}+\epsilon}
\]

其中 `k ∈ {low, mid, high}`。

这样做的好处是：

- 频带边界平滑过渡
- 不会出现硬切造成的频率不连续
- 不同 stage 可以学出不同的分界位置

## 三路注意力

三条频带分支分别使用独立的轻量 SE 模块：

- `SE_low`
- `SE_mid`
- `SE_high`

推荐结构：

1. global average pooling
2. `1x1 conv`
3. `GELU`
4. `1x1 conv`
5. `sigmoid`
6. 乘回各自分支

这样网络可以自己决定：

- 是否更强调低频结构
- 是否更强调中频轮廓
- 是否更强调高频细节恢复

## 融合路径

逆 FFT 和 SE 调制后：

- 将三路结果在通道维拼接
- 用 `1x1 conv` 融合跨频带信息
- 用 `3x3 conv` 恢复局部空间耦合

推荐 shape：

- 单路分支输出：`B x C x H x W`
- 拼接后：`B x 3C x H x W`
- `1x1` 后：`B x C x H x W`
- `3x3` 后：`B x C x H x W`

## 残差缩放

建议使用可学习残差系数：

\[
y = x + \gamma \cdot fused
\]

其中：

- `gamma` 初值设为 `0` 或很小，例如 `1e-3`

理由：

- 模块初始状态接近 identity
- 训练更稳定
- 只有当模块真的学到有用频带增强时，残差才会被逐步放大

## 推荐模块定义

建议类名：

```python
class FrequencyBandEnhancementBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        ...
```

内部组件建议包括：

- `norm: LayerNorm2d`
- 可学习参数 `p1, p2, p3`
- `se_low`
- `se_mid`
- `se_high`
- `fuse_in: Conv2d(3C -> C, kernel_size=1)`
- `fuse_out: Conv2d(C -> C, kernel_size=3, padding=1)`
- `gamma: 可学习标量或通道级张量`

## Forward 伪代码

```python
def forward(self, x):
    identity = x
    x_norm = self.norm(x)

    fft_x = torch.fft.fftshift(torch.fft.fft2(x_norm.float(), dim=(-2, -1)), dim=(-2, -1))

    radius = self._build_radius_map(x.shape[-2], x.shape[-1], x.device, x.dtype)
    r1, r2, tau = self._get_band_params()
    m_low, m_mid, m_high = self._build_soft_masks(radius, r1, r2, tau)

    x_low_fft = fft_x * m_low
    x_mid_fft = fft_x * m_mid
    x_high_fft = fft_x * m_high

    x_low = torch.fft.ifft2(torch.fft.ifftshift(x_low_fft, dim=(-2, -1)), dim=(-2, -1)).real
    x_mid = torch.fft.ifft2(torch.fft.ifftshift(x_mid_fft, dim=(-2, -1)), dim=(-2, -1)).real
    x_high = torch.fft.ifft2(torch.fft.ifftshift(x_high_fft, dim=(-2, -1)), dim=(-2, -1)).real

    x_low = self.se_low(x_low)
    x_mid = self.se_mid(x_mid)
    x_high = self.se_high(x_high)

    fused = torch.cat([x_low, x_mid, x_high], dim=1)
    fused = self.fuse_in(fused)
    fused = self.fuse_out(fused)
    return identity + self.gamma * fused
```

## 接入当前骨干的方式

### 方案 A：只放 bottleneck

当前：

```python
bottleneck = self.bottleneck_base(bottleneck)
bottleneck = self.bottleneck_lpeb(bottleneck)
```

新的消融版本：

```python
bottleneck = self.bottleneck_base(bottleneck)
bottleneck = self.bottleneck_fbeb(bottleneck)
```

这是最干净的第一版测试方式。

### 方案 B：bottleneck + decoder3

```python
bottleneck = self.bottleneck_base(bottleneck)
bottleneck = self.bottleneck_fbeb(bottleneck)

dec3 = self.decoder3_base(dec3)
dec3 = self.decoder3_fbeb(dec3)
```

### 第一版不要这样做

- 不要先放 decoder2
- 不要先放 full-resolution decoder1
- 不要每个 block 都插

## 建议的消融顺序

1. hybrid backbone baseline
2. baseline + bottleneck FBEB
3. baseline + bottleneck + decoder3 FBEB
4. 和原始 `LPEB` 版本对照

主要看：

- PSNR
- SSIM
- 训练稳定性
- 显存与计算量

## 建议记录的诊断量

为了让这个块更可解释，建议记录：

- 学到的 `r1`
- 学到的 `r2`
- 学到的 `tau`
- low 分支 SE 平均权重
- mid 分支 SE 平均权重
- high 分支 SE 平均权重

这些量可以帮助判断：

- 模型是否更偏好低频增强
- 高频修复是否被大量使用
- 不同 stage 是否学出了不同的频带边界

## 为什么这比当前 polar 路线更稳

相比之前的 local polar 方案：

- 不需要再切 patch
- 不需要显式角度估计
- 不需要脆弱的熵型 confidence
- 不依赖 experts 和 router 才能发挥作用
- 它直接增强主干特征流

因此 `FBEB` 更适合作为当前项目下一阶段的低风险可验证方向。

## 最终建议

下一轮对照实验建议如下：

- 单独实现 `FBEB` 为一个新模块
- 使用可学习 `r1 / r2 / tau`
- 第一版先只放 bottleneck
- 将它明确定位成频带增强块，而不是先验估计器
- 不要静默替换冻结版 V1 主线，而是作为明确消融实验运行
