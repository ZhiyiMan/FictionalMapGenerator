from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class ScaleConfig:
    world_width_km: float = 4000.0
    world_height_km: float = 4000.0


@dataclass
class MapConfig:
    width: int = 512
    height: int = 512
    seed: int = 0
    sea_level: float = 0.48
    coarse_scale: int = 8
    brush_size: int = 14
    terrain_strength: float = 1.0
    island_count: int = 40
    island_min_distance: int = 20
    island_min_radius: int = 2
    island_max_radius: int = 5
    shallow_min: float = 0.10
    shallow_max: float = 0.35


@dataclass
class GenerationResult:
    config: MapConfig
    scale: ScaleConfig
    draft_exists: bool
    refined_exists: bool
    final_exists: bool
    bathymetry_exists: bool
    terrain_exists: bool = False

    def to_dict(self) -> dict:
        data = asdict(self)
        data["config"] = asdict(self.config)
        data["scale"] = asdict(self.scale)
        return data
