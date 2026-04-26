from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtGui import QAction, QKeySequence, QShortcut

from core.bathymetry import build_bathymetry
from core.coastline import refine_coastline_from_draft
from core.islands import add_islands_from_bathymetry
from core.models import MapConfig, ScaleConfig
from core.project_io import export_assets, load_project, save_project
from core.scale import compute_scale_info
from ui.canvas_widget import CanvasWidget


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Fantasy Map Studio (MVP)")
        self.resize(1000, 680)

        self.config = MapConfig()
        self.scale = ScaleConfig()

        self.draft_mask: np.ndarray | None = None
        self.refined_mask: np.ndarray | None = None
        self.final_mask: np.ndarray | None = None
        self.bathymetry: np.ndarray | None = None

        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)

        self.canvas = CanvasWidget(self.config.width, self.config.height)
        self.canvas.mask_changed.connect(self._on_mask_changed)
        layout.addWidget(self.canvas, stretch=3)

        panel = self._build_panel()
        panel.setMaximumWidth(320)
        layout.addWidget(panel, stretch=0)
        self._build_menu_bar()
        self._sync_widgets_to_canvas()
        self._update_scale_label()

    def _build_panel(self) -> QWidget:
        panel = QWidget()
        v = QVBoxLayout(panel)
        form = QFormLayout()

        self.w_spin = QSpinBox()
        self.w_spin.setRange(128, 4096)
        self.w_spin.setValue(self.config.width)
        form.addRow("宽度(px)", self.w_spin)

        self.h_spin = QSpinBox()
        self.h_spin.setRange(128, 4096)
        self.h_spin.setValue(self.config.height)
        form.addRow("高度(px)", self.h_spin)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(self.config.seed)
        form.addRow("Seed", self.seed_spin)

        self.sea_spin = QDoubleSpinBox()
        self.sea_spin.setRange(0.01, 0.99)
        self.sea_spin.setSingleStep(0.01)
        self.sea_spin.setValue(self.config.sea_level)
        form.addRow("SeaLevel", self.sea_spin)

        self.coarse_spin = QSpinBox()
        self.coarse_spin.setRange(1, 64)
        self.coarse_spin.setValue(self.config.coarse_scale)
        form.addRow("CoarseScale", self.coarse_spin)

        self.world_w_spin = QDoubleSpinBox()
        self.world_w_spin.setRange(1, 100000)
        self.world_w_spin.setValue(self.scale.world_width_km)
        form.addRow("世界宽(km)", self.world_w_spin)

        self.world_h_spin = QDoubleSpinBox()
        self.world_h_spin.setRange(1, 100000)
        self.world_h_spin.setValue(self.scale.world_height_km)
        form.addRow("世界高(km)", self.world_h_spin)

        self.brush_slider = QSlider(Qt.Orientation.Horizontal)
        self.brush_slider.setRange(1, 80)
        self.brush_slider.setValue(self.config.brush_size)
        form.addRow("笔刷", self.brush_slider)

        self.island_count = QSpinBox()
        self.island_count.setRange(0, 500)
        self.island_count.setValue(self.config.island_count)
        form.addRow("群岛数量", self.island_count)

        self.island_dist = QSpinBox()
        self.island_dist.setRange(1, 200)
        self.island_dist.setValue(self.config.island_min_distance)
        form.addRow("最小间距", self.island_dist)

        self.shallow_min = QDoubleSpinBox()
        self.shallow_min.setRange(0, 1)
        self.shallow_min.setSingleStep(0.01)
        self.shallow_min.setValue(self.config.shallow_min)
        form.addRow("浅海下限", self.shallow_min)

        self.shallow_max = QDoubleSpinBox()
        self.shallow_max.setRange(0, 1)
        self.shallow_max.setSingleStep(0.01)
        self.shallow_max.setValue(self.config.shallow_max)
        form.addRow("浅海上限", self.shallow_max)

        self.terrain_strength_spin = QDoubleSpinBox()
        self.terrain_strength_spin.setRange(0.2, 5.0)
        self.terrain_strength_spin.setSingleStep(0.1)
        self.terrain_strength_spin.setValue(self.config.terrain_strength)
        form.addRow("地形笔刷强度", self.terrain_strength_spin)

        v.addLayout(form)

        self.scale_label = QLabel()
        self.scale_label.setWordWrap(True)
        v.addWidget(self.scale_label)

        v.addStretch(1)

        self.w_spin.valueChanged.connect(self._resize_canvas)
        self.h_spin.valueChanged.connect(self._resize_canvas)
        self.world_w_spin.valueChanged.connect(self._update_scale_label)
        self.world_h_spin.valueChanged.connect(self._update_scale_label)
        self.brush_slider.valueChanged.connect(self.canvas.set_brush_size)
        self.terrain_strength_spin.valueChanged.connect(self.canvas.set_terrain_strength)
        QShortcut(QKeySequence.StandardKey.Undo, self, activated=self.canvas.undo_one)
        return panel

    def _build_menu_bar(self) -> None:
        bar = self.menuBar()

        m_file = bar.addMenu("文件(&F)")
        m_file.addAction(self._act("导出资源…", lambda: self._export(), QKeySequence("Ctrl+Shift+E")))
        m_file.addAction(self._act("保存工程…", lambda: self._save_project(), QKeySequence.StandardKey.Save))
        m_file.addAction(self._act("加载工程…", lambda: self._load_project(), QKeySequence.StandardKey.Open))

        m_draft = bar.addMenu("底稿(&D)")
        m_draft.addAction(self._act("画陆地", lambda: self.canvas.set_tool_land(True)))
        m_draft.addAction(self._act("擦除（海洋）", lambda: self.canvas.set_tool_land(False)))
        m_draft.addSeparator()
        m_draft.addAction(self._act("清空画布", lambda: self.canvas.clear(), QKeySequence("Ctrl+Shift+Delete")))

        m_view = bar.addMenu("视图(&V)")
        m_view.addAction(self._act("显示陆海图", lambda: self.canvas.set_display_mode("mask"), QKeySequence("Ctrl+Alt+1")))
        m_view.addAction(self._act("显示海底图", lambda: self.canvas.set_display_mode("bathy"), QKeySequence("Ctrl+Alt+2")))
        m_view.addAction(self._act("显示地形图", lambda: self._show_terrain_view(), QKeySequence("Ctrl+Alt+3")))

        m_gen = bar.addMenu("生成(&G)")
        m_gen.addAction(self._act("生成海岸线", lambda: self._generate_coastline(), QKeySequence("Ctrl+Shift+1")))
        m_gen.addAction(self._act("生成海底矩阵", lambda: self._generate_bathymetry(), QKeySequence("Ctrl+Shift+2")))
        m_gen.addAction(self._act("生成群岛", lambda: self._generate_islands(), QKeySequence("Ctrl+Shift+3")))

        m_terrain = bar.addMenu("地形(&T)")
        m_terrain.addAction(self._act("进入地形编辑", lambda: self._enter_terrain_edit(), QKeySequence("Ctrl+Shift+T")))
        m_terrain.addAction(self._act("返回底稿编辑", lambda: self._leave_terrain_edit(), QKeySequence("Ctrl+Shift+D")))
        m_terrain.addSeparator()
        m_terrain.addAction(self._act("笔刷：隆起", lambda: self.canvas.set_terrain_op("raise"), QKeySequence("Ctrl+Alt+Up")))
        m_terrain.addAction(self._act("笔刷：压低", lambda: self.canvas.set_terrain_op("lower"), QKeySequence("Ctrl+Alt+Down")))
        m_terrain.addAction(self._act("笔刷：平滑", lambda: self.canvas.set_terrain_op("smooth"), QKeySequence("Ctrl+Alt+S")))

        m_canvas = bar.addMenu("画布(&C)")
        a_in = QAction("放大", self)
        a_in.setShortcut(QKeySequence.StandardKey.ZoomIn)
        a_in.triggered.connect(lambda: self.canvas.set_zoom(self.canvas._zoom * 1.2))
        m_canvas.addAction(a_in)
        a_out = QAction("缩小", self)
        a_out.setShortcut(QKeySequence.StandardKey.ZoomOut)
        a_out.triggered.connect(lambda: self.canvas.set_zoom(self.canvas._zoom / 1.2))
        m_canvas.addAction(a_out)

    def _act(
        self,
        text: str,
        slot,
        shortcut: QKeySequence | QKeySequence.StandardKey | str | None = None,
    ) -> QAction:
        a = QAction(text, self)
        a.triggered.connect(slot)
        if shortcut is not None:
            if isinstance(shortcut, QKeySequence):
                a.setShortcut(shortcut)
            else:
                a.setShortcut(QKeySequence(shortcut))
        return a

    def _sync_widgets_to_canvas(self) -> None:
        self.canvas.set_brush_size(self.config.brush_size)
        self.canvas.set_terrain_strength(self.config.terrain_strength)
        self.canvas.set_mask(np.zeros((self.config.height, self.config.width), dtype=np.uint8))

    def _collect_config(self) -> None:
        self.config.width = int(self.w_spin.value())
        self.config.height = int(self.h_spin.value())
        self.config.seed = int(self.seed_spin.value())
        self.config.sea_level = float(self.sea_spin.value())
        self.config.coarse_scale = int(self.coarse_spin.value())
        self.config.brush_size = int(self.brush_slider.value())
        self.config.island_count = int(self.island_count.value())
        self.config.island_min_distance = int(self.island_dist.value())
        self.config.shallow_min = float(self.shallow_min.value())
        self.config.shallow_max = float(self.shallow_max.value())
        self.config.terrain_strength = float(self.terrain_strength_spin.value())
        self.scale.world_width_km = float(self.world_w_spin.value())
        self.scale.world_height_km = float(self.world_h_spin.value())

    def _on_mask_changed(self) -> None:
        self.draft_mask = self.canvas.mask.copy()

    def _enter_terrain_edit(self) -> None:
        self._collect_config()
        if not np.any(self.canvas.mask > 127):
            QMessageBox.information(self, "提示", "当前没有陆地像素，请先绘制底稿（陆海边界）。")
            return
        self.canvas.set_edit_mode("terrain")

    def _leave_terrain_edit(self) -> None:
        self.canvas.set_edit_mode("land")

    def _show_terrain_view(self) -> None:
        if self.canvas.terrain_height is None:
            QMessageBox.information(self, "提示", "还没有地形数据，请先点击「进入地形编辑」。")
            return
        self.canvas.set_display_mode("terrain")

    def _resize_canvas(self) -> None:
        self._collect_config()
        self.draft_mask = np.zeros((self.config.height, self.config.width), dtype=np.uint8)
        self.refined_mask = None
        self.final_mask = None
        self.bathymetry = None
        self.canvas.set_bathymetry(None)
        self.canvas.set_terrain_height(None)
        self.canvas.set_display_mode("mask")
        self.canvas.set_mask(self.draft_mask)
        self._update_scale_label()

    def _generate_coastline(self) -> None:
        self._collect_config()
        draft = self.canvas.mask.copy()
        self.canvas.set_terrain_height(None)
        self.canvas.set_edit_mode("land")
        self.refined_mask = refine_coastline_from_draft(
            draft,
            seed=self.config.seed,
            sea_level=self.config.sea_level,
            coarse_scale=self.config.coarse_scale,
        )
        self.final_mask = self.refined_mask.copy()
        self.canvas.set_mask(self.final_mask)
        self.canvas.set_display_mode("mask")

    def _generate_bathymetry(self) -> None:
        if self.refined_mask is None:
            QMessageBox.information(self, "提示", "请先点击“生成海岸线”。")
            return
        self._collect_config()
        self.bathymetry = build_bathymetry(self.refined_mask, seed=self.config.seed)
        self.canvas.set_bathymetry(self.bathymetry)
        self.canvas.set_display_mode("bathy")

    def _generate_islands(self) -> None:
        if self.bathymetry is None:
            QMessageBox.information(self, "提示", "请先生成海底矩阵。")
            return
        self._collect_config()
        if self.config.shallow_min > self.config.shallow_max:
            QMessageBox.warning(
                self,
                "参数错误",
                "浅海下限不能大于浅海上限，请调整后再生成群岛。",
            )
            return
        base = self.final_mask if self.final_mask is not None else self.refined_mask
        if base is None:
            return
        self.canvas.set_terrain_height(None)
        self.canvas.set_edit_mode("land")
        self.final_mask = add_islands_from_bathymetry(
            base,
            self.bathymetry,
            seed=self.config.seed,
            max_islands=self.config.island_count,
            min_distance=self.config.island_min_distance,
            min_radius=self.config.island_min_radius,
            max_radius=self.config.island_max_radius,
            shallow_min=self.config.shallow_min,
            shallow_max=self.config.shallow_max,
        )
        self.canvas.set_mask(self.final_mask)
        self.canvas.set_bathymetry(self.bathymetry)
        self.canvas.set_display_mode("mask")

    def _update_scale_label(self) -> None:
        self._collect_config()
        scale = compute_scale_info(
            world_width_km=self.scale.world_width_km,
            world_height_km=self.scale.world_height_km,
            image_width_px=max(1, self.config.width),
            image_height_px=max(1, self.config.height),
        )
        self.scale_label.setText(
            f"比例尺: X 1px={scale.km_per_px_x:.3f}km, "
            f"Y 1px={scale.km_per_px_y:.3f}km; 100km≈{scale.px_per_100km_x:.1f}px"
        )

    def _export(self) -> None:
        self._collect_config()
        path = QFileDialog.getExistingDirectory(self, "选择导出目录")
        if not path:
            return
        export_assets(
            Path(path),
            final_mask=self.final_mask,
            bathymetry=self.bathymetry,
            terrain_height=self.canvas.terrain_height,
            config=self.config,
            scale=self.scale,
        )
        QMessageBox.information(self, "完成", "导出完成。")

    def _save_project(self) -> None:
        self._collect_config()
        path = QFileDialog.getExistingDirectory(self, "选择工程目录")
        if not path:
            return
        save_project(
            Path(path),
            draft_mask=self.draft_mask,
            refined_mask=self.refined_mask,
            final_mask=self.final_mask,
            bathymetry=self.bathymetry,
            terrain_height=self.canvas.terrain_height,
            config=self.config,
            scale=self.scale,
        )
        QMessageBox.information(self, "完成", "工程已保存。")

    def _load_project(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择工程目录")
        if not path:
            return
        data = load_project(Path(path))
        self.draft_mask = data["draft_mask"]
        self.refined_mask = data["refined_mask"]
        self.final_mask = data["final_mask"]
        self.bathymetry = data["bathymetry"]
        terrain_h = data.get("terrain_height")
        self.canvas.set_terrain_height(terrain_h)
        meta = data.get("meta")
        if isinstance(meta, dict):
            cfg = meta.get("config", {})
            scl = meta.get("scale", {})
            if isinstance(cfg, dict):
                self.seed_spin.setValue(int(cfg.get("seed", self.seed_spin.value())))
                self.sea_spin.setValue(float(cfg.get("sea_level", self.sea_spin.value())))
                self.coarse_spin.setValue(int(cfg.get("coarse_scale", self.coarse_spin.value())))
                self.brush_slider.setValue(int(cfg.get("brush_size", self.brush_slider.value())))
                self.island_count.setValue(int(cfg.get("island_count", self.island_count.value())))
                self.island_dist.setValue(int(cfg.get("island_min_distance", self.island_dist.value())))
                self.shallow_min.setValue(float(cfg.get("shallow_min", self.shallow_min.value())))
                self.shallow_max.setValue(float(cfg.get("shallow_max", self.shallow_max.value())))
                self.terrain_strength_spin.setValue(
                    float(cfg.get("terrain_strength", self.terrain_strength_spin.value()))
                )
                self.canvas.set_terrain_strength(float(self.terrain_strength_spin.value()))
            if isinstance(scl, dict):
                self.world_w_spin.setValue(float(scl.get("world_width_km", self.world_w_spin.value())))
                self.world_h_spin.setValue(float(scl.get("world_height_km", self.world_h_spin.value())))
        show = None
        for arr in (self.final_mask, self.refined_mask, self.draft_mask):
            if arr is not None:
                show = arr
                break
        if show is not None:
            h, w = show.shape
            self.w_spin.blockSignals(True)
            self.h_spin.blockSignals(True)
            try:
                self.w_spin.setValue(int(w))
                self.h_spin.setValue(int(h))
            finally:
                self.w_spin.blockSignals(False)
                self.h_spin.blockSignals(False)
            self.config.width = int(w)
            self.config.height = int(h)
            self.canvas.set_mask(show)
        self.canvas.set_bathymetry(self.bathymetry)
        self.canvas.set_display_mode("mask")
        self._update_scale_label()
        QMessageBox.information(self, "完成", "工程已加载。")
