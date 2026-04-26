from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScaleInfo:
    km_per_px_x: float
    km_per_px_y: float
    px_per_100km_x: float
    px_per_100km_y: float


def compute_scale_info(
    *,
    world_width_km: float,
    world_height_km: float,
    image_width_px: int,
    image_height_px: int,
) -> ScaleInfo:
    if world_width_km <= 0 or world_height_km <= 0:
        raise ValueError("world_width_km 和 world_height_km 必须大于 0")
    if image_width_px <= 0 or image_height_px <= 0:
        raise ValueError("image_width_px 和 image_height_px 必须大于 0")

    km_per_px_x = world_width_km / image_width_px
    km_per_px_y = world_height_km / image_height_px

    return ScaleInfo(
        km_per_px_x=km_per_px_x,
        km_per_px_y=km_per_px_y,
        px_per_100km_x=100.0 / km_per_px_x,
        px_per_100km_y=100.0 / km_per_px_y,
    )
