from __future__ import annotations

import numpy as np


def _coast_tangent_field(land_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    基于陆地梯度估算海岸法线，再旋转得到切线方向场。
    """
    land = (land_mask > 127).astype(np.float32)
    gy, gx = np.gradient(land)
    tx = -gy
    ty = gx
    norm = np.sqrt(tx * tx + ty * ty) + 1e-6
    return tx / norm, ty / norm


def _paint_disk(mask: np.ndarray, cx: int, cy: int, radius: int) -> None:
    h, w = mask.shape
    y0 = max(0, cy - radius)
    y1 = min(h, cy + radius + 1)
    x0 = max(0, cx - radius)
    x1 = min(w, cx + radius + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
    mask[y0:y1, x0:x1][circle] = 255


def add_islands_from_bathymetry(
    land_mask: np.ndarray,
    bathymetry: np.ndarray,
    *,
    seed: int,
    max_islands: int,
    min_distance: int,
    min_radius: int,
    max_radius: int,
    shallow_min: float,
    shallow_max: float,
) -> np.ndarray:
    if land_mask.shape != bathymetry.shape:
        raise ValueError("land_mask 与 bathymetry 尺寸必须一致")
    if min_radius > max_radius:
        raise ValueError("min_radius 不能大于 max_radius")

    out = land_mask.copy()
    sea = out < 128

    # 岛链偏好：以浅海带中值作为“最佳深度”，越靠近该深度分数越高。
    band_center = (shallow_min + shallow_max) / 2.0
    band_half = max(0.03, (shallow_max - shallow_min) / 2.0)
    band_score = 1.0 - np.abs(bathymetry - band_center) / band_half
    band_score = np.clip(band_score, 0.0, 1.0)

    # 低频噪声场用于把候选区切成“岛链状带”，而非均匀撒点。
    h, w = out.shape
    rng = np.random.default_rng(seed)
    ch = max(8, h // 20)
    cw = max(8, w // 20)
    coarse = rng.random((ch, cw), dtype=np.float32)
    from PIL import Image

    noise_img = Image.fromarray((coarse * 255).astype(np.uint8), mode="L").resize(
        (w, h), Image.Resampling.BILINEAR
    )
    low_freq = np.asarray(noise_img, dtype=np.float32) / 255.0

    candidate = (
        sea
        & (bathymetry >= shallow_min)
        & (bathymetry <= shallow_max)
        & (band_score > 0.35)
        & (low_freq > 0.45)
    )
    ys, xs = np.where(candidate)
    if ys.size == 0:
        return out

    # hybrid 密度：自动给建议上限，并允许手动值覆盖（取两者中的较合理值）。
    auto_suggest = max(8, int(np.sqrt(ys.size) / max(1.0, min_distance * 0.7)))
    if max_islands <= 0:
        target_islands = auto_suggest
    else:
        target_islands = min(max_islands, int(auto_suggest * 1.8))
    target_islands = max(1, target_islands)

    order = rng.permutation(ys.size)
    chosen: list[tuple[int, int]] = []
    min_dist2 = float(min_distance * min_distance)

    for idx in order:
        y = int(ys[idx])
        x = int(xs[idx])
        ok = True
        for cy, cx in chosen:
            if (x - cx) ** 2 + (y - cy) ** 2 < min_dist2:
                ok = False
                break
        if not ok:
            continue
        chosen.append((y, x))
        if len(chosen) >= target_islands:
            break

    # 每个锚点沿海岸切线方向绘制短链，方向更贴合海岸线走势。
    tangent_x, tangent_y = _coast_tangent_field(land_mask)
    chain_len_min = 2
    chain_len_max = 5
    step = max(2, min_distance // 2)
    for y, x in chosen:
        chain_len = int(rng.integers(chain_len_min, chain_len_max + 1))
        tx = float(tangent_x[y, x])
        ty = float(tangent_y[y, x])
        if abs(tx) + abs(ty) < 1e-4:
            angle = float(rng.uniform(0.0, np.pi * 2.0))
            dx = int(round(np.cos(angle) * step))
            dy = int(round(np.sin(angle) * step))
        else:
            sign = 1.0 if rng.random() < 0.5 else -1.0
            dx = int(round(tx * step * sign))
            dy = int(round(ty * step * sign))
        if dx == 0 and dy == 0:
            dx = step

        cy, cx = y, x
        for _ in range(chain_len):
            if not (0 <= cy < h and 0 <= cx < w):
                break
            if out[cy, cx] > 127:
                break
            if not (shallow_min <= bathymetry[cy, cx] <= shallow_max):
                break
            radius = int(rng.integers(min_radius, max_radius + 1))
            _paint_disk(out, cx, cy, radius)
            cy += dy
            cx += dx
    return out
