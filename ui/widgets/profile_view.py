"""
海底地层剖面可视化
==================
将测线反射强度数据渲染为 2D 海底地层剖面图，
类似浅地层剖面仪 (Sub-Bottom Profiler) 专业显示。

- 2D 成像: 沿航迹距离 × 深度, 反射强度映射为颜色
- 滚轮横向平移, 固定窗口显示
- 地震色标 (灰度/红蓝)
"""

import traceback
import numpy as np
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QCheckBox, QSlider, QSpinBox, QPushButton,
)
from PyQt5.QtCore import Qt, pyqtSignal

from config.settings import FIGURE_DPI


# 专业地震色标: 黑-蓝-白-红 (正振幅=红, 负=蓝, 零=白)
SEISMIC_CMAP = LinearSegmentedColormap.from_list("seismic_pro", [
    (0.0,  "#000033"),
    (0.15, "#0033AA"),
    (0.35, "#3366DD"),
    (0.45, "#AACCF0"),
    (0.5,  "#FFFFFF"),
    (0.55, "#F0CCAA"),
    (0.65, "#DD6633"),
    (0.85, "#AA3300"),
    (1.0,  "#330000"),
])

# 海底沉积物色标: 深蓝(软泥) → 白 → 棕(砂砾)
SEABED_CMAP = LinearSegmentedColormap.from_list("seabed", [
    (0.0,  "#0D47A1"),
    (0.2,  "#1565C0"),
    (0.4,  "#42A5F5"),
    (0.5,  "#FFFFFF"),
    (0.6,  "#FFCC80"),
    (0.8,  "#E65100"),
    (1.0,  "#BF360C"),
])


def _distance_along_track(lon, lat):
    """Haversine: 计算沿测线累积距离 (米)"""
    R = 6371000.0
    rlon = np.radians(np.asarray(lon, float))
    rlat = np.radians(np.asarray(lat, float))
    dlon = np.diff(rlon); dlat = np.diff(rlat)
    a = np.sin(dlat/2)**2 + np.cos(rlat[:-1]) * np.cos(rlat[1:]) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a + 1e-15))
    d = np.zeros(len(lon))
    d[1:] = np.cumsum(R * c)
    return d


class ProfileView(QWidget):
    """
    2D 海底地层剖面视图。

    从测线数据 (GPS + 深度 + 反射强度) 渲染为:
      - 2D 成像剖面 (距离 × 深度, 反射强度=颜色)
      - 底部导航条

    交互:
      - 鼠标滚轮: 横向平移
      - 底部滑块: 快速跳转
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._survey_lines = []
        self._current_name = None
        self._win_width = 500.0   # 显示窗口宽度 (m)
        self._scroll = 0.0
        self._cmap_name = "seismic"
        self._total_dist = 0.0
        self._updating = False

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # ---- 控制栏 ----
        ctrl = QHBoxLayout()
        ctrl.setSpacing(5)

        ctrl.addWidget(QLabel("测线:"))
        self.cmb_line = QComboBox()
        self.cmb_line.setMinimumWidth(130)
        self.cmb_line.setMaximumWidth(180)
        self.cmb_line.currentTextChanged.connect(self._on_change)
        ctrl.addWidget(self.cmb_line)

        ctrl.addWidget(QLabel("窗口(m):"))
        self.spin_win = QSpinBox()
        self.spin_win.setRange(50, 20000)
        self.spin_win.setValue(500)
        self.spin_win.setSingleStep(100)
        self.spin_win.setMaximumWidth(80)
        self.spin_win.valueChanged.connect(self._on_win_changed)
        ctrl.addWidget(self.spin_win)

        ctrl.addWidget(QLabel("色标:"))
        self.cmb_cmap = QComboBox()
        self.cmb_cmap.addItems(["seismic", "seabed", "gray", "viridis", "RdYlBu"])
        self.cmb_cmap.setMaximumWidth(90)
        self.cmb_cmap.currentTextChanged.connect(self._on_cmap)
        ctrl.addWidget(self.cmb_cmap)

        self.lbl_info = QLabel("")
        self.lbl_info.setStyleSheet("color: #666; font-size: 10px;")
        ctrl.addWidget(self.lbl_info)
        ctrl.addStretch()

        layout.addLayout(ctrl)

        # ---- 主剖面画布 ----
        self.figure = Figure(figsize=(12, 5.5), dpi=FIGURE_DPI)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(300)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        layout.addWidget(self.canvas, 1)

        # ---- 导航条 ----
        nav = QHBoxLayout()
        nav.setSpacing(2)
        self.scrollbar = QSlider(Qt.Horizontal)
        self.scrollbar.setMinimum(0)
        self.scrollbar.setMaximum(1000)
        self.scrollbar.valueChanged.connect(self._on_slider)
        nav.addWidget(self.scrollbar)
        layout.addLayout(nav)

    # ==================== 数据 ====================

    def set_survey_lines(self, survey_lines):
        self._survey_lines = survey_lines
        self.cmb_line.blockSignals(True)
        self.cmb_line.clear()
        self.cmb_line.addItems([sl.name for sl in survey_lines])
        if survey_lines:
            self.cmb_line.setCurrentIndex(0)
            self._current_name = survey_lines[0].name
        self.cmb_line.blockSignals(False)
        self._scroll = 0.0
        self._render()

    def clear(self):
        self._survey_lines = []
        self.cmb_line.clear()
        self.figure.clear()
        self.canvas.draw()

    # ==================== 交互 ====================

    def _on_change(self, *_):
        self._current_name = self.cmb_line.currentText()
        self._scroll = 0.0
        self.scrollbar.setValue(0)
        self._render()

    def _on_win_changed(self, val):
        self._win_width = float(val)
        self._render()

    def _on_cmap(self, name):
        self._cmap_name = name
        self._render()

    def _on_scroll(self, event):
        step = self._win_width * 0.12
        if event.button == 'up':
            self._scroll = max(0, self._scroll - step)
        else:
            self._scroll = min(max(0, self._total_dist - self._win_width), self._scroll + step)
        self._sync_slider()
        self._render()

    def _on_slider(self, v):
        if self._updating or self._total_dist <= self._win_width:
            return
        self._scroll = (v / 1000.0) * (self._total_dist - self._win_width)
        self._render()

    def _sync_slider(self):
        self._updating = True
        if self._total_dist > self._win_width > 0:
            self.scrollbar.setValue(int(self._scroll / (self._total_dist - self._win_width) * 1000))
        else:
            self.scrollbar.setValue(0)
        self._updating = False

    # ==================== 渲染 ====================

    def _render(self):
        self.figure.clear()

        sl = None
        for s in self._survey_lines:
            if s.name == self._current_name:
                sl = s; break

        if sl is None or len(sl.longitude) < 2:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "No survey data.\nImport TXT with GPS + Depth + Intensity.",
                    transform=ax.transAxes, ha="center", va="center", fontsize=12, color="gray")
            self.canvas.draw(); return

        try:
            self._render_section(sl)
        except Exception:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, traceback.format_exc()[-600:],
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=7, color="red", family="monospace")
            self.canvas.draw()

    def _render_section(self, sl):
        # ---- 使用道集数据 ----
        lon = np.asarray(sl.longitude, np.float64)
        lat = np.asarray(sl.latitude, np.float64)
        seq = np.asarray(sl.sequence_number, np.float64)
        intensity = np.asarray(sl.reflection_intensity, np.float64)

        dist = _distance_along_track(lon, lat)
        self._total_dist = dist[-1]

        # 窗口裁剪
        w0, w1 = self._scroll, self._scroll + self._win_width
        m = (dist >= w0) & (dist <= w1)
        if m.sum() < 2:
            m[:] = True; w0, w1 = dist[0], dist[-1]

        d_sub = dist[m]
        dp_sub = seq[m]  # 序号作为深度代理
        int_sub = intensity[m]
        n_pts = len(d_sub)

        # ---- 色标 ----
        cmap_lookup = {"seismic": SEISMIC_CMAP, "seabed": SEABED_CMAP,
                       "gray": "gray", "viridis": "viridis", "RdYlBu": "RdYlBu_r"}
        cmap = cmap_lookup.get(self._cmap_name, "gray")
        vmin = np.percentile(intensity, 2)
        vmax = np.percentile(intensity, 98)
        if vmax - vmin < 0.01:
            vmin, vmax = intensity.min() - 1, intensity.max() + 1

        # ---- 纯原始数据渲染: 每个点按 (距离, 深度, 强度) 画散点 ----
        ax = self.figure.add_subplot(111)

        # 点大小根据数据密度自适应
        if n_pts > 5000:
            s = 4
        elif n_pts > 1000:
            s = 10
        else:
            s = 25

        sc = ax.scatter(d_sub, dp_sub, c=int_sub, s=s, cmap=cmap,
                        vmin=vmin, vmax=vmax, edgecolors='none',
                        alpha=0.85, marker='o')

        # 颜色条
        cbar = self.figure.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
        cbar.set_label("Reflection Intensity (dB)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

        # 坐标轴
        ax.set_xlim(w0, w1)
        ax.invert_yaxis()
        ax.set_xlabel("Distance along track (m)", fontsize=9)
        ax.set_ylabel("Depth (m)", fontsize=9)
        ax.set_title(
            f"Sub-bottom Profile: {sl.name}  "
            f"[{w0:.0f} - {w1:.0f} m / {self._total_dist:.0f}m]  "
            f"({n_pts} raw points)",
            fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.12, linestyle='--')

        # 标注
        i_lo = np.argmin(dp_sub)
        i_hi = np.argmax(dp_sub)
        for idx, label, color, yoff in [
            (i_lo, f"Shallowest\n{dp_sub[i_lo]:.1f}m", "#1565C0", -25),
            (i_hi, f"Deepest\n{dp_sub[i_hi]:.1f}m", "#E65100", 25),
        ]:
            ax.annotate(label, (d_sub[idx], dp_sub[idx]),
                        xytext=(0, yoff), textcoords="offset points",
                        fontsize=7, color=color, fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

        self.figure.tight_layout()
        self.canvas.draw()

        self.lbl_info.setText(
            f"Dist: [{dist.min():.0f}, {dist.max():.0f}]m | "
            f"Seq: [{seq.min():.0f}, {seq.max():.0f}] | "
            f"Intensity: [{intensity.min():.1f}, {intensity.max():.1f}]dB | "
            f"Raw pts: {len(lon)} | Scroll=pan"
        )
