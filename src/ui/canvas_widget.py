from __future__ import annotations

from collections import deque
import time

import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPaintEvent, QPen, QPixmap
from PySide6.QtWidgets import QMenu, QMessageBox, QWidget

from core.terrain import brush_along_segment, init_height_from_mask


class CanvasWidget(QWidget):
    mask_changed = Signal()
    terrain_changed = Signal()

    def __init__(self, width: int = 512, height: int = 512) -> None:
        super().__init__()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(420, 420)
        self._mask = np.zeros((height, width), dtype=np.uint8)
        self._display_mode = "mask"
        self._bathymetry: np.ndarray | None = None
        self._zoom = 1.0
        self._brush_size = 14
        self._draw_land = True
        self._last_pos: QPoint | None = None
        self._last_snapshot: np.ndarray | None = None
        self._line_mode = False
        self._line_start: QPoint | None = None
        self._pan_mode = False
        self._pan_last: QPoint | None = None
        self._space_pressed = False
        self._pan_offset = QPoint(0, 0)
        self._prev_space_press_mono: float | None = None
        self._double_space_seconds = 0.35
        self._edit_mode: str = "land"
        self._terrain_op: str = "raise"
        self._terrain_strength: float = 1.0
        self._height: np.ndarray | None = None
        self._last_height_snapshot: np.ndarray | None = None

    @property
    def mask(self) -> np.ndarray:
        return self._mask

    @property
    def terrain_height(self) -> np.ndarray | None:
        return self._height

    def set_mask(self, arr: np.ndarray) -> None:
        new_mask = arr.copy().astype(np.uint8)
        if self._height is not None and self._height.shape != new_mask.shape:
            self._height = None
            self._last_height_snapshot = None
        self._mask = new_mask
        self.update()
        self.mask_changed.emit()

    def set_terrain_height(self, arr: np.ndarray | None) -> None:
        if arr is None:
            self._height = None
        else:
            self._height = arr.astype(np.float32).copy()
        self.update()
        self.terrain_changed.emit()

    def set_edit_mode(self, mode: str) -> None:
        if mode not in ("land", "terrain"):
            raise ValueError(mode)
        self._edit_mode = mode
        if mode == "terrain":
            if self._height is None:
                self._height = init_height_from_mask(self._mask)
            self._display_mode = "terrain"
        else:
            self._display_mode = "mask"
        self.update()

    def set_terrain_op(self, op: str) -> None:
        if op not in ("raise", "lower", "smooth"):
            raise ValueError(op)
        self._terrain_op = op

    def set_terrain_strength(self, value: float) -> None:
        self._terrain_strength = max(0.1, min(5.0, float(value)))

    def set_bathymetry(self, bathymetry: np.ndarray | None) -> None:
        self._bathymetry = bathymetry
        self.update()

    def set_display_mode(self, mode: str) -> None:
        self._display_mode = mode
        self.update()

    def set_brush_size(self, size: int) -> None:
        self._brush_size = max(1, int(size))

    def set_tool_land(self, is_land: bool) -> None:
        self._draw_land = is_land

    def set_zoom(self, zoom: float) -> None:
        self._zoom = min(6.0, max(0.3, zoom))
        self.update()

    def _reset_view(self) -> None:
        """缩放与平移恢复默认（双击空格触发）。"""
        self._zoom = 1.0
        self._pan_offset = QPoint(0, 0)
        self._pan_mode = False
        self._pan_last = None
        self.update()

    def clear(self) -> None:
        self._save_undo()
        self._mask.fill(0)
        self._height = None
        self._last_height_snapshot = None
        self.update()
        self.mask_changed.emit()
        self.terrain_changed.emit()

    def undo_one(self) -> None:
        if self._edit_mode == "terrain":
            if self._last_height_snapshot is None or self._height is None:
                return
            self._height = self._last_height_snapshot.copy()
            self._last_height_snapshot = None
            self.update()
            self.terrain_changed.emit()
            return
        if self._last_snapshot is None:
            return
        self._mask = self._last_snapshot
        self._last_snapshot = None
        self.update()
        self.mask_changed.emit()

    def _save_undo(self) -> None:
        if self._edit_mode == "terrain" and self._height is not None:
            self._last_height_snapshot = self._height.copy()
        else:
            self._last_snapshot = self._mask.copy()

    def _qimage_from_display(self) -> QImage:
        h, w = self._mask.shape
        if self._display_mode == "terrain" and self._height is not None:
            land = self._mask > 127
            elev = self._height
            t = np.clip(np.nan_to_num(elev, nan=0.0), 0.0, 1.0)
            r = np.zeros((h, w), dtype=np.float32)
            g = np.zeros((h, w), dtype=np.float32)
            b = np.zeros((h, w), dtype=np.float32)
            r[land] = 40.0 + 120.0 * t[land]
            g[land] = 80.0 + 100.0 * t[land]
            b[land] = 50.0 + 40.0 * t[land]
            r[~land] = 15.0
            g[~land] = 40.0
            b[~land] = 80.0
            rgb = np.dstack([r, g, b]).astype(np.uint8).copy()
            return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()

        if self._display_mode == "bathy" and self._bathymetry is not None:
            depth = np.clip(self._bathymetry, 0.0, 1.0)
            r = (20 + depth * 50).astype(np.uint8)
            g = (40 + depth * 90).astype(np.uint8)
            b = (110 + depth * 130).astype(np.uint8)
            land = self._mask > 127
            r[land], g[land], b[land] = 200, 190, 150
            rgb = np.dstack([r, g, b]).copy()
            return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()

        gray = self._mask.copy()
        return QImage(gray.data, w, h, w, QImage.Format.Format_Grayscale8).copy()

    def _to_mask_point(self, p: QPoint) -> tuple[int, int]:
        h, w = self._mask.shape
        draw_w = int(w * self._zoom)
        draw_h = int(h * self._zoom)
        left = (self.width() - draw_w) // 2 + self._pan_offset.x()
        top = (self.height() - draw_h) // 2 + self._pan_offset.y()
        x = int((p.x() - left) / self._zoom)
        y = int((p.y() - top) / self._zoom)
        return max(0, min(w - 1, x)), max(0, min(h - 1, y))

    def _draw_to_mask(self, p1: QPoint, p2: QPoint) -> None:
        x1, y1 = self._to_mask_point(p1)
        x2, y2 = self._to_mask_point(p2)
        color = 255 if self._draw_land else 0
        img = QImage(
            self._mask.data,
            self._mask.shape[1],
            self._mask.shape[0],
            self._mask.shape[1],
            QImage.Format.Format_Grayscale8,
        )
        painter = QPainter(img)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pen = QPen(QColor(color, color, color), self._brush_size, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawLine(x1, y1, x2, y2)
        painter.end()

    def _draw_terrain_segment(self, p1: QPoint, p2: QPoint) -> None:
        if self._height is None:
            return
        land_mask = self._mask > 127
        x0, y0 = self._to_mask_point(p1)
        x1, y1 = self._to_mask_point(p2)
        brush_along_segment(
            self._height,
            land_mask,
            x0,
            y0,
            x1,
            y1,
            max(2, self._brush_size // 2),
            mode=self._terrain_op,
            strength=self._terrain_strength,
        )

    def _fill_enclosed_region(self, click_pos: QPoint) -> tuple[bool, str]:
        x, y = self._to_mask_point(click_pos)
        if self._mask[y, x] > 127:
            return False, "请在轮廓内部的海域点击右键填充。"

        h, w = self._mask.shape
        visited = np.zeros((h, w), dtype=np.uint8)
        q: deque[tuple[int, int]] = deque()
        q.append((y, x))
        visited[y, x] = 1
        region: list[tuple[int, int]] = []
        touches_border = False

        while q:
            cy, cx = q.popleft()
            region.append((cy, cx))
            if cy == 0 or cx == 0 or cy == h - 1 or cx == w - 1:
                touches_border = True

            if cy > 0 and not visited[cy - 1, cx] and self._mask[cy - 1, cx] <= 127:
                visited[cy - 1, cx] = 1
                q.append((cy - 1, cx))
            if cy < h - 1 and not visited[cy + 1, cx] and self._mask[cy + 1, cx] <= 127:
                visited[cy + 1, cx] = 1
                q.append((cy + 1, cx))
            if cx > 0 and not visited[cy, cx - 1] and self._mask[cy, cx - 1] <= 127:
                visited[cy, cx - 1] = 1
                q.append((cy, cx - 1))
            if cx < w - 1 and not visited[cy, cx + 1] and self._mask[cy, cx + 1] <= 127:
                visited[cy, cx + 1] = 1
                q.append((cy, cx + 1))

        if touches_border:
            return False, "轮廓可能未封闭，填充区域与边界连通。"

        self._save_undo()
        for cy, cx in region:
            self._mask[cy, cx] = 255
        self.update()
        self.mask_changed.emit()
        return True, f"已自动填充封闭区域（{len(region)} 像素）。"

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        self.setFocus()
        if event.button() == Qt.MouseButton.RightButton:
            if self._edit_mode == "terrain":
                return
            menu = QMenu(self)
            act = menu.addAction("自动填充封闭区域")
            chosen = menu.exec(event.globalPosition().toPoint())
            if chosen is act:
                ok, msg = self._fill_enclosed_region(event.position().toPoint())
                if not ok:
                    QMessageBox.information(self, "自动填充", msg)
            return

        if event.button() not in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            return
        if event.button() == Qt.MouseButton.MiddleButton or (
            event.button() == Qt.MouseButton.LeftButton and self._space_pressed
        ):
            self._pan_mode = True
            self._pan_last = event.position().toPoint()
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            return

        pos = event.position().toPoint()
        if self._edit_mode == "terrain":
            if bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
                self._save_undo()
                self._line_mode = True
                self._line_start = pos
                self._last_pos = None
                return
            self._line_mode = False
            self._line_start = None
            self._save_undo()
            self._last_pos = pos
            self._draw_terrain_segment(pos, pos)
            self.update()
            self.terrain_changed.emit()
            return

        if bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            self._save_undo()
            self._line_mode = True
            self._line_start = pos
            self._last_pos = None
            return

        self._line_mode = False
        self._line_start = None
        self._save_undo()
        self._last_pos = pos
        self._draw_to_mask(self._last_pos, self._last_pos)
        self.update()
        self.mask_changed.emit()

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if self._pan_mode and self._pan_last is not None:
            cur = event.position().toPoint()
            delta = cur - self._pan_last
            self._pan_offset += delta
            self._pan_last = cur
            self.update()
            return
        if self._line_mode:
            return
        if self._last_pos is None:
            return
        cur = event.position().toPoint()
        if self._edit_mode == "terrain":
            self._draw_terrain_segment(self._last_pos, cur)
            self._last_pos = cur
            self.update()
            self.terrain_changed.emit()
            return
        self._draw_to_mask(self._last_pos, cur)
        self._last_pos = cur
        self.update()
        self.mask_changed.emit()

    def _sync_cursor_after_pan(self) -> None:
        """平移结束后：若仍按住空格则保持手掌光标，否则恢复箭头。"""
        if self._space_pressed:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            if self._pan_mode:
                self._pan_mode = False
                self._pan_last = None
                self._sync_cursor_after_pan()
                return
        if event.button() == Qt.MouseButton.LeftButton:
            if self._line_mode and self._line_start is not None:
                end = event.position().toPoint()
                if self._edit_mode == "terrain":
                    self._draw_terrain_segment(self._line_start, end)
                    self.update()
                    self.terrain_changed.emit()
                else:
                    self._draw_to_mask(self._line_start, end)
                    self.update()
                    self.mask_changed.emit()
            self._line_mode = False
            self._line_start = None
            self._last_pos = None

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        if event.key() == Qt.Key.Key_Space:
            if not event.isAutoRepeat():
                now = time.monotonic()
                if self._prev_space_press_mono is not None and (
                    now - self._prev_space_press_mono
                ) < self._double_space_seconds:
                    self._reset_view()
                    self._prev_space_press_mono = None
                    self._space_pressed = False
                    self.setCursor(Qt.CursorShape.ArrowCursor)
                    event.accept()
                    return
                self._prev_space_press_mono = now
            self._space_pressed = True
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:  # type: ignore[override]
        if event.key() == Qt.Key.Key_Space:
            if not event.isAutoRepeat():
                self._space_pressed = False
                if not self._pan_mode:
                    self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        super().keyReleaseEvent(event)

    def paintEvent(self, _event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(28, 28, 28))
        img = self._qimage_from_display()
        pix = QPixmap.fromImage(img)
        h, w = self._mask.shape
        draw_w = int(w * self._zoom)
        draw_h = int(h * self._zoom)
        target = QRect(
            (self.width() - draw_w) // 2 + self._pan_offset.x(),
            (self.height() - draw_h) // 2 + self._pan_offset.y(),
            draw_w,
            draw_h,
        )
        painter.drawPixmap(target, pix)
