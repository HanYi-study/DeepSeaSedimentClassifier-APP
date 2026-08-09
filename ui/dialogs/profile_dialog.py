"""
Backscatter Profile Viewer
===========================
海底声学剖面展示窗口。

显示:
  上图: 2D 声学剖面 (距离 × 序号深度, 颜色 = 反射强度)
        Y轴 = 真实采样序号 (深度代理, 越大越深)
        每点在其实际 (距离, 序号) 位置着色
  下图: 1D 强度-距离曲线
"""

import traceback
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QSpinBox, QPushButton, QCheckBox,
)
from PyQt5.QtCore import Qt


def _haversine_distance(lon, lat):
    R = 6371000.0
    rlon = np.radians(np.asarray(lon, float))
    rlat = np.radians(np.asarray(lat, float))
    dlon = np.diff(rlon); dlat = np.diff(rlat)
    a = np.sin(dlat/2)**2 + np.cos(rlat[:-1])*np.cos(rlat[1:])*np.sin(dlon/2)**2
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a+1e-15))
    d = np.zeros(len(lon))
    d[1:] = np.cumsum(R*c)
    return d


class ProfileDialog(QDialog):
    """
    Backscatter Profile 窗口

    上图: 2D 剖面 —— 每点位于真实 (距离, 序号) 坐标, 颜色 = 反射强度
         序号越大 = 越深, Y轴反转, 海底曲线自然呈现
    下图: 1D 强度-距离曲线
    """

    def __init__(self, survey_line, click_distance, tif_path=None, parent=None):
        super().__init__(parent)
        self._sl = survey_line
        self._click_dist = click_distance
        self._tif_path = tif_path
        self._half_span = 500.0
        self._dot_size = 12
        self._sub_data = None

        self._setup_ui()
        self._recompute()

    def _setup_ui(self):
        self.setWindowTitle("Backscatter Profile")
        self.setMinimumSize(950, 650); self.resize(1200, 800)
        layout = QVBoxLayout(self); layout.setSpacing(4)

        ctrl = QHBoxLayout(); ctrl.setSpacing(6)
        ctrl.addWidget(QLabel("Span (m):"))
        self.spin_span = QSpinBox()
        self.spin_span.setRange(50, 50000); self.spin_span.setValue(1000)
        self.spin_span.setSingleStep(100); self.spin_span.setMaximumWidth(85)
        self.spin_span.valueChanged.connect(self._on_params); ctrl.addWidget(self.spin_span)

        ctrl.addWidget(QLabel("Dot Size:"))
        self.spin_dot = QSpinBox()
        self.spin_dot.setRange(2, 60); self.spin_dot.setValue(12)
        self.spin_dot.setMaximumWidth(65)
        self.spin_dot.valueChanged.connect(self._on_params); ctrl.addWidget(self.spin_dot)

        self.lbl_info = QLabel("")
        self.lbl_info.setStyleSheet("color: #555; font-size: 10px;")
        ctrl.addWidget(self.lbl_info); ctrl.addStretch()

        btn = QPushButton("Close"); btn.clicked.connect(self.close); ctrl.addWidget(btn)
        layout.addLayout(ctrl)

        self.figure = Figure(figsize=(13, 8), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar); layout.addWidget(self.canvas)

    def _on_params(self, *_):
        self._half_span = self.spin_span.value() / 2.0
        self._dot_size = self.spin_dot.value()
        self._recompute()

    def _recompute(self):
        try:
            lon = np.asarray(self._sl.longitude, np.float64)
            lat = np.asarray(self._sl.latitude, np.float64)
            intensity = np.asarray(self._sl.reflection_intensity, np.float64)
            seq = np.asarray(self._sl.sequence_number, np.float64)

            dist = _haversine_distance(lon, lat)
            d0 = self._click_dist - self._half_span
            d1 = self._click_dist + self._half_span
            mask = (dist >= d0) & (dist <= d1)

            if mask.sum() < 2:
                self.lbl_info.setText("No data. Increase Span.")
                self.figure.clear(); self.canvas.draw(); return

            self._sub_data = (dist[mask], seq[mask], intensity[mask])
            n = mask.sum()
            self.lbl_info.setText(
                f"Points: {n} | Dist: [{dist[mask].min():.0f}, {dist[mask].max():.0f}]m | "
                f"Seq: [{seq[mask].min():.0f}, {seq[mask].max():.0f}] | "
                f"Intensity: [{intensity[mask].min():.3f}, {intensity[mask].max():.3f}]"
            )
            self._render()
        except Exception as e:
            self.lbl_info.setText(f"ERROR: {e}")
            self.figure.clear(); ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, f"Error:\n{traceback.format_exc()[-400:]}",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=8, color="red", family="monospace")
            self.canvas.draw()

    def _render(self):
        self.figure.clear()
        if self._sub_data is None: self.canvas.draw(); return
        try: self._do_render()
        except Exception as e:
            self.figure.clear(); ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, f"Render Error:\n{traceback.format_exc()[-400:]}",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=8, color="red", family="monospace")
            self.canvas.draw()

    def _do_render(self):
        dist, seq, intensity = self._sub_data
        n = len(dist)

        # 颜色范围
        vmin = np.nanpercentile(intensity, 2)
        vmax = np.nanpercentile(intensity, 98)
        if vmax - vmin < 0.01:
            vmin, vmax = intensity.min() - 0.01, intensity.max() + 0.01

        # ---- 上图: 2D 剖面 ----
        # 每个原始数据点在其真实 (distance, seq) 位置着色
        # Y轴 = 序号 (深度代理, 数值大=深), 反转Y轴使"深"在下方
        # 海底曲线 = 同一GPS位置的多个序号点, 自然呈现
        ax1 = self.figure.add_subplot(211)

        sc = ax1.scatter(dist, seq, c=intensity, s=self._dot_size,
                         cmap="viridis", vmin=vmin, vmax=vmax,
                         edgecolors='none', alpha=0.85, marker='o')

        ax1.axvline(x=self._click_dist, color='red', linestyle='--',
                    linewidth=2, alpha=0.9, zorder=10,
                    label=f"Click @ {self._click_dist:.0f}m")

        ax1.invert_yaxis()  # 序号大=深, 在下方
        ax1.set_ylabel("Sequence Number (depth proxy)", fontsize=10)
        ax1.set_title(
            f"Backscatter Profile: {self._sl.name}  "
            f"({n} pts, Y=seq number as depth)",
            fontsize=12, fontweight="bold")
        ax1.legend(fontsize=8, loc='upper right')
        ax1.grid(True, alpha=0.15, linestyle='--')
        cbar = self.figure.colorbar(sc, ax=ax1, shrink=0.8, pad=0.02)
        cbar.set_label("Reflection Intensity", fontsize=9)

        # ---- 下图: 1D 强度-距离曲线 ----
        ax2 = self.figure.add_subplot(212, sharex=ax1)

        ax2.plot(dist, intensity, '-', color="#1E88E5", linewidth=1.2, alpha=0.8)
        ax2.scatter(dist, intensity, s=max(2, self._dot_size//3),
                    color="#1E88E5", alpha=0.4, edgecolors='none')

        mean_i = intensity.mean()
        ax2.axhline(y=mean_i, color='orange', linestyle=':', linewidth=1.2,
                    alpha=0.8, label=f"Mean = {mean_i:.4f}")
        ax2.axvline(x=self._click_dist, color='red', linestyle='--',
                    linewidth=2, alpha=0.8, zorder=10)

        ax2.set_xlabel("Distance along track (m)", fontsize=10)
        ax2.set_ylabel("Backscatter Intensity", fontsize=10, color="#1E88E5")
        ax2.tick_params(axis='y', labelcolor="#1E88E5")
        ax2.legend(fontsize=8, loc='upper right')
        ax2.grid(True, alpha=0.2, linestyle='--')

        self.figure.tight_layout()
        self.canvas.draw()
