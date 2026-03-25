import numpy as np
import cv2
import matplotlib.pyplot as plt


def hann2d(h: int, w: int) -> np.ndarray:
    """2D Hann window."""
    wy = np.hanning(h)
    wx = np.hanning(w)
    return np.outer(wy, wx).astype(np.float32)


def fft_amplitude_phase(patch: np.ndarray, use_hann: bool = True):
    """
    Compute centered FFT amplitude/phase for a grayscale patch.
    patch: float32, range [0,1] preferred
    """
    patch = patch.astype(np.float32)
    if use_hann:
        patch = patch * hann2d(*patch.shape)

    F = np.fft.fft2(patch)
    F = np.fft.fftshift(F)

    amp = np.abs(F)
    phase = np.angle(F)

    # log amplitude is usually easier to use
    log_amp = np.log1p(amp)
    return log_amp, phase


def build_polar_grids(h: int, w: int):
    """
    Return radius rho and angle theta grids for a centered spectrum.
    theta is mapped to [0, pi), i.e. direction not orientation.
    """
    cy, cx = h // 2, w // 2
    yy, xx = np.indices((h, w))
    x = xx - cx
    y = yy - cy

    rho = np.sqrt(x ** 2 + y ** 2)

    # angle in [0, 2pi)
    theta = np.arctan2(y, x)
    theta = np.mod(theta, np.pi)  # collapse theta and theta+pi into same direction
    return rho, theta


def sector_direction_prior(
    patch: np.ndarray,
    num_sectors: int = 12,
    r_min_ratio: float = 0.08,
    r_max_ratio: float = 0.45,
    radial_power: float = 1.0,
    use_soft_sector: bool = False,
    soft_sigma_deg: float = 10.0,
):
    """
    Compute sector-wise directional prior P_k from a single patch.

    Returns:
        result: dict with
            - log_amp
            - phase
            - sector_energy
            - sector_prior
            - dir_confidence
            - blur_score
            - theta_centers_deg
    """
    if patch.ndim == 3:
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

    patch = patch.astype(np.float32)
    if patch.max() > 1.0:
        patch = patch / 255.0

    h, w = patch.shape
    log_amp, phase = fft_amplitude_phase(patch, use_hann=True)
    rho, theta = build_polar_grids(h, w)

    r_max = min(h, w) / 2.0
    r_min = r_min_ratio * r_max
    r_high = r_max_ratio * r_max

    # Use only a mid-frequency annulus
    radial_mask = (rho >= r_min) & (rho <= r_high)

    # Optional: suppress the strong horizontal/vertical cross a bit
    # remove very near x/y axes in frequency domain if you want
    # Uncomment if needed:
    # axis_band = 2
    # cy, cx = h // 2, w // 2
    # yy, xx = np.indices((h, w))
    # axis_mask = (np.abs(xx - cx) > axis_band) & (np.abs(yy - cy) > axis_band)
    # radial_mask = radial_mask & axis_mask

    A2 = log_amp ** 2
    weights = np.power(np.maximum(rho, 1e-6), radial_power)

    theta_edges = np.linspace(0.0, np.pi, num_sectors + 1)
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    sector_energy = np.zeros(num_sectors, dtype=np.float64)

    if not use_soft_sector:
        for k in range(num_sectors):
            ang_mask = (theta >= theta_edges[k]) & (theta < theta_edges[k + 1])
            mask = radial_mask & ang_mask
            sector_energy[k] = np.sum(weights[mask] * A2[mask])
    else:
        # Soft angular binning
        sigma = np.deg2rad(soft_sigma_deg)
        for k, tc in enumerate(theta_centers):
            # shortest distance on [0, pi)
            d = np.abs(theta - tc)
            d = np.minimum(d, np.pi - d)
            ang_w = np.exp(-(d ** 2) / (2 * sigma ** 2))
            mask = radial_mask
            sector_energy[k] = np.sum(weights[mask] * A2[mask] * ang_w[mask])

    sector_prior = sector_energy / (np.sum(sector_energy) + 1e-8)

    # Direction confidence via normalized entropy
    entropy = -np.sum(sector_prior * np.log(sector_prior + 1e-12))
    dir_confidence = 1.0 - entropy / np.log(num_sectors)

    # Simple blur score: less high-frequency energy => blurrier
    # Use a second wider band to estimate retained HF ratio
    r_mid = 0.20 * r_max
    r_hi2 = 0.48 * r_max
    all_mask = (rho >= r_mid) & (rho <= r_hi2)
    hf_mask = (rho >= 0.32 * r_max) & (rho <= r_hi2)

    all_energy = np.sum(A2[all_mask]) + 1e-8
    hf_energy = np.sum(A2[hf_mask])
    hf_ratio = hf_energy / all_energy
    blur_score = 1.0 - hf_ratio

    return {
        "log_amp": log_amp,
        "phase": phase,
        "sector_energy": sector_energy,
        "sector_prior": sector_prior,
        "dir_confidence": float(dir_confidence),
        "blur_score": float(blur_score),
        "theta_centers_deg": np.rad2deg(theta_centers),
    }


def sliding_direction_prior_map(
    image: np.ndarray,
    patch_size: int = 128,
    stride: int = 32,
    num_sectors: int = 12,
):
    """
    Sliding-window directional prior on a whole image.

    Returns:
        maps: dict with
            - dir_prior_map: (H, W, K)
            - dir_conf_map: (H, W)
            - blur_map: (H, W)
            - count_map: (H, W)
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    gray = gray.astype(np.float32)
    if gray.max() > 1.0:
        gray = gray / 255.0

    H, W = gray.shape
    K = num_sectors

    dir_prior_map = np.zeros((H, W, K), dtype=np.float32)
    dir_conf_map = np.zeros((H, W), dtype=np.float32)
    blur_map = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patch = gray[y:y + patch_size, x:x + patch_size]
            res = sector_direction_prior(
                patch,
                num_sectors=num_sectors,
                r_min_ratio=0.08,
                r_max_ratio=0.45,
                radial_power=1.0,
                use_soft_sector=True,
                soft_sigma_deg=10.0,
            )

            p = res["sector_prior"].astype(np.float32)
            c = np.float32(res["dir_confidence"])
            b = np.float32(res["blur_score"])

            # Fill center region or whole patch; here use whole patch averaging
            dir_prior_map[y:y + patch_size, x:x + patch_size, :] += p[None, None, :]
            dir_conf_map[y:y + patch_size, x:x + patch_size] += c
            blur_map[y:y + patch_size, x:x + patch_size] += b
            count_map[y:y + patch_size, x:x + patch_size] += 1.0

    valid = count_map > 0
    dir_prior_map[valid] /= count_map[valid][:, None]
    dir_conf_map[valid] /= count_map[valid]
    blur_map[valid] /= count_map[valid]

    return {
        "dir_prior_map": dir_prior_map,
        "dir_conf_map": dir_conf_map,
        "blur_map": blur_map,
        "count_map": count_map,
    }


def visualize_patch_result(patch: np.ndarray, res: dict, title: str = "patch", save_path: str = None):
    """Show patch, log amplitude, phase, and sector prior bar plot. Optionally save to file."""
    log_amp = res["log_amp"]
    phase = res["phase"]
    sector_prior = res["sector_prior"]
    theta_deg = res["theta_centers_deg"]

    plt.figure(figsize=(12, 3.5))

    plt.subplot(1, 4, 1)
    plt.imshow(patch, cmap="gray")
    plt.title(title)
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(log_amp, cmap="viridis")
    plt.title("log amplitude")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(phase, cmap="viridis")
    plt.title("phase")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.bar(theta_deg, sector_prior, width=180 / len(theta_deg) * 0.8)
    plt.title(f"sector prior\nconf={res['dir_confidence']:.4f}, blur={res['blur_score']:.4f}")
    plt.xlabel("direction (deg)")
    plt.ylabel("prior")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200)
    plt.show()


def visualize_maps(gray: np.ndarray, maps: dict, sector_idx: int = None, save_path: str = None):
    """
    Visualize image, blur map, direction confidence map,
    and optionally one sector prior map. Optionally save to file.
    """
    dir_conf_map = maps["dir_conf_map"]
    blur_map = maps["blur_map"]
    dir_prior_map = maps["dir_prior_map"]

    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3 if sector_idx is None else 4, 1)
    plt.imshow(gray, cmap="gray")
    plt.title("input")
    plt.axis("off")

    plt.subplot(1, 3 if sector_idx is None else 4, 2)
    plt.imshow(blur_map, cmap="jet")
    plt.title("blur map")
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(1, 3 if sector_idx is None else 4, 3)
    plt.imshow(dir_conf_map, cmap="jet")
    plt.title("direction confidence")
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04)

    if sector_idx is not None:
        plt.subplot(1, 4, 4)
        plt.imshow(dir_prior_map[:, :, sector_idx], cmap="jet")
        plt.title(f"sector prior k={sector_idx}")
        plt.axis("off")
        plt.colorbar(fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200)
    plt.show()


if __name__ == "__main__":
    img_path = "pic/GOPR0372_07_00_000047.png"   # 改成你的路径
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    y, x = 200, 200
    patch_size = 128
    patch = img[y:y + patch_size, x:x + patch_size]

    res = sector_direction_prior(
        patch,
        num_sectors=12,
        r_min_ratio=0.08,
        r_max_ratio=0.45,
        radial_power=1.0,
        use_soft_sector=True,
        soft_sigma_deg=10.0,
    )
    patch_fig_path = f"pic/patch_sector_prior_{x}_{y}.png"
    visualize_patch_result(patch, res, title=f"patch ({x},{y})", save_path=patch_fig_path)
    print(f"Patch sector prior figure saved to: {patch_fig_path}")

    print("Sector prior:", np.round(res["sector_prior"], 4))
    print("Direction confidence:", res["dir_confidence"])
    print("Blur score:", res["blur_score"])

    maps = sliding_direction_prior_map(
        img,
        patch_size=128,
        stride=32,
        num_sectors=12,
    )

    map_fig_path = "pic/whole_image_direction_map.png"
    visualize_maps(img, maps, sector_idx=2, save_path=map_fig_path)
    print(f"Whole image direction map figure saved to: {map_fig_path}")