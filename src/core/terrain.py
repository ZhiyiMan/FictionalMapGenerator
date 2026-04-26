from __future__ import annotations

import numpy as np


def init_height_from_mask(mask: np.ndarray, *, base_elevation: float = 0.35) -> np.ndarray:
    """陆地初始海拔 [0,1]，海域为 NaN。"""
    h, w = mask.shape
    out = np.full((h, w), np.nan, dtype=np.float32)
    land = mask > 127
    out[land] = np.float32(base_elevation)
    return out


def _iter_line_points(x0: int, y0: int, x1: int, y1: int, step: float) -> list[tuple[int, int]]:
    pts: list[tuple[int, int]] = []
    dx = x1 - x0
    dy = y1 - y0
    length = float(np.hypot(dx, dy))
    if length < 1e-6:
        return [(x0, y0)]
    n = max(1, int(length / step) + 1)
    for i in range(n + 1):
        t = i / n
        x = int(round(x0 + dx * t))
        y = int(round(y0 + dy * t))
        pts.append((x, y))
    # 去重相邻重复点
    dedup: list[tuple[int, int]] = []
    for p in pts:
        if not dedup or dedup[-1] != p:
            dedup.append(p)
    return dedup


def stamp_raise_lower(
    height: np.ndarray,
    land_mask: np.ndarray,
    cy: int,
    cx: int,
    radius: int,
    delta: float,
    *,
    clamp01: bool = True,
) -> None:
    """圆形软笔刷：delta>0 隆起，delta<0 压低。仅在陆地上修改。"""
    h, w = height.shape
    r = max(1, radius)
    y0 = max(0, cy - r)
    y1 = min(h, cy + r + 1)
    x0 = max(0, cx - r)
    x1 = min(w, cx + r + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    dist = np.sqrt((yy - cy).astype(np.float64) ** 2 + (xx - cx).astype(np.float64) ** 2)
    falloff = np.clip(1.0 - dist / float(r), 0.0, 1.0).astype(np.float32) ** 2
    patch_land = land_mask[y0:y1, x0:x1]
    patch_ok = patch_land & np.isfinite(height[y0:y1, x0:x1])
    delta_map = delta * falloff * patch_ok.astype(np.float32)
    height[y0:y1, x0:x1] = np.where(
        patch_ok,
        height[y0:y1, x0:x1] + delta_map,
        height[y0:y1, x0:x1],
    )
    if clamp01:
        hi = height[y0:y1, x0:x1]
        hi[patch_land & np.isfinite(hi)] = np.clip(hi[patch_land & np.isfinite(hi)], 0.0, 1.0)


def stamp_smooth(
    height: np.ndarray,
    land_mask: np.ndarray,
    cy: int,
    cx: int,
    radius: int,
) -> None:
    """局部平滑：圆形邻域内对陆地像素做 3×3 邻域均值。"""
    h, w = height.shape
    r = max(2, radius)
    y0 = max(1, cy - r)
    y1 = min(h - 1, cy + r + 1)
    x0 = max(1, cx - r)
    x1 = min(w - 1, cx + r + 1)
    if y1 <= y0 or x1 <= x0:
        return

    patch_h = height[y0:y1, x0:x1].copy()
    patch_land = land_mask[y0:y1, x0:x1]
    yy, xx = np.ogrid[y0:y1, x0:x1]
    dist = np.sqrt((yy - cy).astype(np.float64) ** 2 + (xx - cx).astype(np.float64) ** 2)
    circle = dist <= float(r)

    smoothed = patch_h.copy()
    ph, pw = patch_h.shape
    for iy in range(1, ph - 1):
        for ix in range(1, pw - 1):
            if not patch_land[iy, ix] or not circle[iy, ix]:
                continue
            window = patch_h[iy - 1 : iy + 2, ix - 1 : ix + 2]
            wm = patch_land[iy - 1 : iy + 2, ix - 1 : ix + 2]
            finite = wm & np.isfinite(window)
            if int(finite.sum()) < 1:
                continue
            smoothed[iy, ix] = float(np.nansum(np.where(finite, window, 0.0)) / float(finite.sum()))

    height[y0:y1, x0:x1] = np.where(patch_land & circle, smoothed, patch_h)


def brush_along_segment(
    height: np.ndarray,
    land_mask: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    radius: int,
    *,
    mode: str,
    strength: float,
) -> None:
    """沿线插值盖章；mode: raise | lower | smooth"""
    if mode == "smooth":
        step = max(2.0, radius * 0.55)
    else:
        step = max(1.0, radius * 0.35)
    pts = _iter_line_points(x0, y0, x1, y1, step)
    delta = float(strength) * 0.08
    for cx, cy in pts:
        if mode == "raise":
            stamp_raise_lower(height, land_mask, cy, cx, radius, delta)
        elif mode == "lower":
            stamp_raise_lower(height, land_mask, cy, cx, radius, -delta)
        elif mode == "smooth":
            stamp_smooth(height, land_mask, cy, cx, radius)
        else:
            raise ValueError(f"未知 mode: {mode}")
