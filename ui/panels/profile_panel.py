"""
Backscatter 海底切面 + 去噪预处理
================================
展示每条航带的最大反射强度, 支持去噪前后对比。

去噪方法:
  - 中值滤波: 去除尖峰噪声
  - 高斯平滑: 去除高频随机噪声
"""

import numpy as np
from scipy.ndimage import median_filter, gaussian_filter1d
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QCheckBox, QComboBox, QHBoxLayout, QPushButton


def _denoise(intensity, method="median", strength=3):
    """去噪处理"""
    if method == "median":
        return median_filter(intensity, size=strength)
    elif method == "gaussian":
        return gaussian_filter1d(intensity.astype(float), sigma=strength)
    elif method == "both":
        tmp = median_filter(intensity, size=3)
        return gaussian_filter1d(tmp.astype(float), sigma=strength)
    return intensity


class ProfilePanel(QWidget):
    """海底声学切面面板 (含去噪)"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._survey_lines = []
        self._denoise_method = "median"
        self._denoise_strength = 3
        self._show_raw = True
        self._show_denoised = True
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # 去噪控制
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("去噪:"))
        self.cmb_denoise = QComboBox()
        self.cmb_denoise.addItems(["无", "中值滤波", "高斯平滑", "中值+高斯"])
        self.cmb_denoise.setCurrentIndex(0)  # 默认不处理
        ctrl.addWidget(self.cmb_denoise)
        self.btn_apply = QPushButton("应用去噪")
        self.btn_apply.clicked.connect(self._render)
        self.btn_apply.setStyleSheet(
            "QPushButton { background-color: #FF9800; color: white; padding: 4px 8px; }")
        ctrl.addWidget(self.btn_apply)
        ctrl.addStretch()

        self.chk_raw = QCheckBox("原始")
        self.chk_raw.setChecked(True)
        ctrl.addWidget(self.chk_raw)
        self.chk_denoised = QCheckBox("去噪后")
        self.chk_denoised.setChecked(True)
        ctrl.addWidget(self.chk_denoised)
        layout.addLayout(ctrl)

        self.figure = Figure(figsize=(7.5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(240)
        layout.addWidget(self.canvas)

        self.lbl_stats = QLabel("No data")
        self.lbl_stats.setStyleSheet("color: #555; font-size: 10px;")
        layout.addWidget(self.lbl_stats)

    def set_survey_lines(self, lines):
        self._survey_lines = lines
        self._render()

    def _render(self, *_):
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        if not self._survey_lines:
            ax.text(0.5, 0.5, "No survey data", ha="center", va="center", color="gray")
            self.canvas.draw(); return

        # 解析去噪方法
        method_map = {"无": None, "中值滤波": "median", "高斯平滑": "gaussian", "中值+高斯": "both"}
        self._denoise_method = method_map.get(self.cmb_denoise.currentText(), "median")

        n = len(self._survey_lines)
        max_raw = np.zeros(n)
        max_denoised = np.zeros(n)
        n_pts = np.zeros(n, dtype=int)

        for i, sl in enumerate(self._survey_lines):
            intensity = np.asarray(sl.reflection_intensity, float)
            max_raw[i] = intensity.max()
            n_pts[i] = len(intensity)
            if self._denoise_method:
                denoised = _denoise(intensity, self._denoise_method, 3)
                max_denoised[i] = denoised.max()
            else:
                max_denoised[i] = max_raw[i]

        x = np.arange(1, n + 1)

        # 原始曲线 (灰色, 细线)
        if self.chk_raw.isChecked():
            ax.plot(x, max_raw, '-', color="gray", linewidth=0.8, alpha=0.6, label="Raw")
            ax.scatter(x, max_raw, s=8, color="gray", edgecolors='none', alpha=0.4)

        # 去噪后曲线 (红色, 粗线)
        if self.chk_denoised.isChecked() and self._denoise_method:
            ax.plot(x, max_denoised, '-', color="#E53935", linewidth=1.5, alpha=0.9,
                    label=f"Denoised ({self.cmb_denoise.currentText()})")

        ax.set_xlabel("Survey Line Index", fontsize=9)
        ax.set_ylabel("Backscatter Intensity", fontsize=9)
        ax.set_title(f"Seafloor Backscatter ({n} lines)", fontsize=11, fontweight="bold")
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.2, linestyle='--')

        self.figure.tight_layout()
        self.canvas.draw()

        method_name = self.cmb_denoise.currentText()
        self.lbl_stats.setText(
            f"Lines: {n} | Raw max: [{max_raw.min():.4f}, {max_raw.max():.4f}] | "
            f"Denoised max: [{max_denoised.min():.4f}, {max_denoised.max():.4f}] | "
            f"Method: {method_name} | Total pts: {n_pts.sum()}"
        )
