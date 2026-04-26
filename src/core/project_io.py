from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from core.models import GenerationResult, MapConfig, ScaleConfig


def export_assets(
    export_dir: Path,
    *,
    final_mask: np.ndarray | None,
    bathymetry: np.ndarray | None,
    terrain_height: np.ndarray | None = None,
    config: MapConfig,
    scale: ScaleConfig,
) -> Path:
    export_dir.mkdir(parents=True, exist_ok=True)
    if final_mask is not None:
        from PIL import Image

        Image.fromarray(final_mask, mode="L").save(export_dir / "final_mask.png")
    if bathymetry is not None:
        np.save(export_dir / "bathymetry.npy", bathymetry.astype(np.float32))
    if terrain_height is not None:
        np.save(export_dir / "terrain_height.npy", terrain_height.astype(np.float32))

    result = GenerationResult(
        config=config,
        scale=scale,
        draft_exists=False,
        refined_exists=False,
        final_exists=final_mask is not None,
        bathymetry_exists=bathymetry is not None,
        terrain_exists=terrain_height is not None,
    )
    with (export_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
    return export_dir


def save_project(
    project_dir: Path,
    *,
    draft_mask: np.ndarray | None,
    refined_mask: np.ndarray | None,
    final_mask: np.ndarray | None,
    bathymetry: np.ndarray | None,
    terrain_height: np.ndarray | None = None,
    config: MapConfig,
    scale: ScaleConfig,
) -> Path:
    project_dir.mkdir(parents=True, exist_ok=True)
    if draft_mask is not None:
        np.save(project_dir / "draft_mask.npy", draft_mask)
    if refined_mask is not None:
        np.save(project_dir / "refined_mask.npy", refined_mask)
    if final_mask is not None:
        np.save(project_dir / "final_mask.npy", final_mask)
    if bathymetry is not None:
        np.save(project_dir / "bathymetry.npy", bathymetry.astype(np.float32))
    if terrain_height is not None:
        np.save(project_dir / "terrain_height.npy", terrain_height.astype(np.float32))

    result = GenerationResult(
        config=config,
        scale=scale,
        draft_exists=draft_mask is not None,
        refined_exists=refined_mask is not None,
        final_exists=final_mask is not None,
        bathymetry_exists=bathymetry is not None,
        terrain_exists=terrain_height is not None,
    )
    with (project_dir / "project.json").open("w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
    return project_dir


def load_project(project_dir: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for key in ("draft_mask", "refined_mask", "final_mask", "bathymetry", "terrain_height"):
        p = project_dir / f"{key}.npy"
        data[key] = np.load(p) if p.exists() else None

    meta_path = project_dir / "project.json"
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            data["meta"] = json.load(f)
    else:
        data["meta"] = None
    return data
