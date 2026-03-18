import cv2
import numpy as np
import matplotlib.pyplot as plt

img_path = "pic/scene002_gt_11.png"

# 读灰度图
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise ValueError(f"Cannot read image: {img_path}")

# 随机切分四个256x256的patch
import random
h, w = img.shape
crop_size = 256
num_patches = 4
if h < crop_size or w < crop_size:
    raise ValueError(f"Image size too small for random crops: {img.shape}")

random.seed(12)  # 固定随机种子，保证可复现
blocks = []
coords = []
for _ in range(num_patches):
    top = random.randint(0, h - crop_size)
    left = random.randint(0, w - crop_size)
    blocks.append(img[top:top+crop_size, left:left+crop_size])
    coords.append((top, left))

magnitudes = []
phases = []
for blk in blocks:
    F = np.fft.fft2(blk)
    F_shift = np.fft.fftshift(F)
    magnitude = np.abs(F_shift)
    magnitude_log = np.log1p(magnitude)
    phase = np.angle(F_shift)
    phase_vis = (phase + np.pi) / (2 * np.pi)
    magnitudes.append(magnitude_log)
    phases.append(phase_vis)

# 可视化：3行4列
plt.figure(figsize=(16, 10))
for i in range(num_patches):
    top, left = coords[i]
    plt.subplot(3, num_patches, i+1)
    plt.imshow(blocks[i], cmap="gray")
    plt.title(f"patch{i+1} ({top},{left})")
    plt.axis("off")
    plt.subplot(3, num_patches, i+1+num_patches)
    plt.imshow(magnitudes[i], cmap="viridis")
    plt.title(f"patch{i+1} 频谱")
    plt.axis("off")
    plt.subplot(3, num_patches, i+1+2*num_patches)
    plt.imshow(phases[i], cmap="viridis")
    plt.title(f"patch{i+1} 相位")
    plt.axis("off")

plt.tight_layout()
plt.savefig("pic_phase_spectrum_gt.png", dpi=200)
plt.show()