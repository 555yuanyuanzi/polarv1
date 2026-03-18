# FBEB + LocalRefinementBlock 新架构设计

## 1. 目的

本文档定义一版新的去模糊结构方向，用于替代当前 `LPEB -> Router -> Experts` 路线。

核心原则：

- 去掉显式方向先验、路由器和专家模型
- 保留当前 hybrid backbone
- 用 `FBEB` 负责全局频带增强
- 用 `LocalRefinementBlock` 负责局部边缘与纹理修复
- 在解码阶段同时保留
  - 空间域跳跃连接
  - 频域跳跃连接

这条路线的目标是：

- 降低结构复杂度
- 避免 experts / router 塌缩
- 保留频域增强的研究主线
- 让全局与局部恢复职责更清楚

## 2. 当前损失函数

当前 `v1` 已实现的损失在 [losses.py](/c:/Users/86155/polar_code/v1/src/losses.py)：

- `CharbonnierLoss`
- `FrequencyLoss`

具体定义：

- `CharbonnierLoss`
  - 当前默认主损失
  - 用于像素级恢复监督
- `FrequencyLoss`
  - 当前实现为预测图和 GT 图幅值谱的 L1
  - 默认关闭

当前推荐的下一版损失组合：

\[
L = L_{char} + \lambda_f L_{freq}
\]

建议初始权重：

- `L_char = 1.0`
- `\lambda_f = 0.05 ~ 0.10`

说明：

- 第一版不要再引入感知损失或 GAN
- 先只加频率损失，观察高频细节恢复是否改善

## 3. 新架构总览

新的结构不再包含：

- `LocalPolarPrior`
- `TopKRouter`
- `SharedExpert`
- `Short/Long Iso/Aniso Experts`
- `LPEB`

新的恢复主线为：

```text
Input
  -> Patch Embed
  -> Encoder1
  -> Down1
  -> Encoder2
  -> Down2
  -> Encoder3
  -> Down3
  -> BottleneckBase
  -> Up3
  -> DualSkipFuse3
  -> Decoder3Base
  -> FBEB_3
  -> LocalRefine_3
  -> Up2
  -> DualSkipFuse2
  -> Decoder2Base
  -> FBEB_2
  -> LocalRefine_2
  -> Up1
  -> Fuse1
  -> Decoder1
  -> Output
```

其中：

- `FBEB_3` 对应 `decoder3`
- `FBEB_2` 对应 `decoder2`
- `LocalRefine_3` 和 `LocalRefine_2` 为共享结构但参数独立的局部恢复块

## 4. Backbone 保留部分

Backbone 仍然沿用当前 hybrid U 形结构：

- `Patch Embed: 3x3 Conv(3 -> 48)`
- `Encoder1: 3 x NAFBlock(48)`
- `Encoder2: 4 x NAFBlock(96)`
- `Encoder3: 6 x NAFBlock(192)`
- `Bottleneck: 3 x RestormerLiteBlock(384)`
- `Decoder3: 2 x RestormerLiteBlock(192)`
- `Decoder2: 2 x RestormerLiteBlock(96)`
- `Decoder1: 3 x NAFBlock(48)`

说明：

- `FBEB` 不放 bottleneck
- `FBEB` 只放 `decoder3` 和 `decoder2`
- bottleneck 保持纯 backbone 表征

## 5. FBEB 的职责

`FBEB` 的作用是：

- 对整张 stage feature 做频带分解
- 学习 low / mid / high 的软划分
- 对三路频带做 SE 调制
- 再融合回主干

它负责的是：

- 全局频带重加权
- 结构频率与细节频率的重新分配

它不负责：

- 局部 patch 路由
- 显式 blur-type 分类
- 专家选择

因此在新架构里，`FBEB` 被定位为：

**Global Frequency Enhancement Module**

## 6. LocalRefinementBlock 定义

`LocalRefinementBlock` 用来承担局部恢复职责。

### 输入输出

输入：

\[
X \in \mathbb{R}^{B \times C \times H \times W}
\]

输出：

\[
Y \in \mathbb{R}^{B \times C \times H \times W}
\]

### 结构

推荐第一版结构：

1. `3x3 DWConv`
2. `1x1 Conv`
3. `GELU`
4. `5x5 DWConv`
5. `1x1 Conv`
6. residual add

形式化写法：

\[
Y = X + PW_2(DW_{5\times5}(GELU(PW_1(DW_{3\times3}(X)))))
\]

### 作用

它负责：

- 局部边缘修复
- 局部纹理整理
- 高频细节补偿

它不负责：

- 全局频带重加权
- 长距离依赖建模

因此在新架构里，`LocalRefinementBlock` 被定位为：

**Local Detail Recovery Module**

## 7. 双跳跃连接设计

新的解码阶段使用两类 skip：

- 空间域跳跃连接
- 频域跳跃连接

### 7.1 空间域跳跃连接

这部分保留当前 U 形结构的常规 skip。

例如：

- `enc3 -> decoder3`
- `enc2 -> decoder2`

即直接把 encoder feature 送到 decoder fuse。

其职责是：

- 保留空间布局
- 保留结构细节
- 弥补上采样后的空间信息损失

### 7.2 频域跳跃连接

频域跳跃连接不直接传递 FFT 复数张量，而是传递：

**encoder 特征经过频带适配后的频域增强表示**

定义一个轻量模块：

`FrequencySkipAdapter`

输入：

\[
E_s \in \mathbb{R}^{B \times C_s \times H_s \times W_s}
\]

处理：

1. `LayerNorm2d`
2. `FFT`
3. `low / mid / high` soft mask
4. `IFFT`
5. `1x1 Conv`

输出：

\[
F_s \in \mathbb{R}^{B \times C_s \times H_s \times W_s}
\]

它的职责是：

- 把 encoder skip 中的频带信息显式整理后送给 decoder
- 给 decoder 提供额外的频率结构提示

### 7.3 融合方式

对 `decoder3`：

- `S3 = spatial skip from enc3`
- `F3 = frequency skip from enc3`
- `D3 = upsampled decoder feature`

推荐融合方式：

\[
U_3 = concat(S_3, F_3, D_3)
\]

然后：

1. `1x1 Conv`
2. `NAFBlock`
3. `1x1 Conv`

同理对 `decoder2` 做一套：

\[
U_2 = concat(S_2, F_2, D_2)
\]

说明：

- 空间 skip 和频域 skip 不是二选一
- 二者同时作为 decoder 的辅助输入
- 频域 skip 是辅助，不应比空间 skip 更强

## 8. 解码阶段完整流程

### Decoder3

\[
D_3 = Up(bottleneck)
\]

\[
S_3 = enc3
\]

\[
F_3 = FrequencySkipAdapter(enc3)
\]

\[
Z_3 = DualSkipFuse(D_3, S_3, F_3)
\]

\[
Z_3 = Decoder3Base(Z_3)
\]

\[
Z_3 = FBEB_3(Z_3)
\]

\[
Z_3 = LocalRefine_3(Z_3)
\]

### Decoder2

\[
D_2 = Up(Z_3)
\]

\[
S_2 = enc2
\]

\[
F_2 = FrequencySkipAdapter(enc2)
\]

\[
Z_2 = DualSkipFuse(D_2, S_2, F_2)
\]

\[
Z_2 = Decoder2Base(Z_2)
\]

\[
Z_2 = FBEB_2(Z_2)
\]

\[
Z_2 = LocalRefine_2(Z_2)
\]

### Decoder1

`Decoder1` 保持简单：

- 空间 skip 保留
- 不加频域 skip
- 不加 FBEB
- 不加 LocalRefine

这样可以控制复杂度。

## 9. 为什么不再使用专家模型

这次去掉 experts / router / polar 的原因是：

1. 当前显式先验不稳定
2. router 很容易塌缩或平均化
3. expert 分工很难证明对恢复真正有效
4. 结构过长，不利于归因

新的职责分工更清楚：

- `RestormerLite`: 中尺度上下文恢复
- `FBEB`: 全局频带增强
- `LocalRefinementBlock`: 局部细节修复

这比：

- `prior -> router -> experts`

更稳，也更容易做对照实验。

## 10. 建议的损失函数

### 主损失

\[
L_{char}
\]

### 辅助频率损失

\[
L_{freq} = \| |\mathcal{F}(I_{pred})| - |\mathcal{F}(I_{gt})| \|_1
\]

### 总损失

\[
L = L_{char} + \lambda_f L_{freq}
\]

建议：

- `\lambda_f = 0.05` 起步
- 如果高频恢复有帮助，再试 `0.10`

## 11. 推荐实验顺序

### 版本 A

- 去掉 `LPEB`
- 只保留 backbone

### 版本 B

- backbone + `FBEB_3`

### 版本 C

- backbone + `FBEB_3 + FBEB_2`

### 版本 D

- backbone + `FBEB_3 + FBEB_2`
- 再加 `LocalRefine_3 + LocalRefine_2`

### 版本 E

- 在版本 D 基础上
- 加 `FrequencyLoss`

### 版本 F

- 在版本 D 或 E 基础上
- 再加入双跳跃连接中的频域 skip

不要一开始就把所有部件一起加上，否则难以归因。

## 12. 最终建议

当前最值得优先验证的新主线是：

**Hybrid Backbone + FBEB(decoder3, decoder2) + LocalRefinementBlock(decoder3, decoder2)**

并在后续版本里逐步验证：

- `FrequencyLoss`
- 频域跳跃连接

这条线相比原来的 `LPEB -> Router -> Experts`：

- 更短
- 更稳
- 更符合当前实验现象
- 更容易得到可靠结论
