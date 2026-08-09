"""
SEG 浅剖可视化 + 预处理去噪
=========================
展示 SEG-Y 剖面, 支持去噪前后对比。

预处理选项:
  - 去均值 (remove DC)
  - 中值滤波 (spike removal)
  - 高斯平滑 (high-freq noise)
  - AGC (gain balance)
  - 道间均衡 (trace balance)
"""

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QCheckBox, QPushButton,
)
from PyQt5.QtCore import Qt

from core.seg_preprocess import preprocess_pipeline


class SegView(QWidget):
    """SEG 多剖面可视化 + 预处理"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._seg_list = []
        self._processed = {}      # name -> processed data (full)
        self._processed_disp = {} # name -> processed data (display-sampled)
        self._raw_disp = {}       # name -> raw data (display-sampled)
        self._last_steps = None   # track steps to avoid re-processing
        self._cmap = "gray"
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # 预处理控制
        pp = QHBoxLayout()
        pp.addWidget(QLabel("去噪:"))
        pp.addWidget(QLabel("(不勾=自动)"))
        self.chk_demean = QCheckBox("去均值")
        self.chk_demean.setChecked(True); self.chk_demean.toggled.connect(self._render)
        pp.addWidget(self.chk_demean)
        self.chk_median = QCheckBox("中值滤波")
        self.chk_median.setChecked(True); self.chk_median.toggled.connect(self._render)
        pp.addWidget(self.chk_median)
        self.chk_gauss = QCheckBox("高斯")
        self.chk_gauss.toggled.connect(self._render)
        pp.addWidget(self.chk_gauss)
        self.chk_agc = QCheckBox("AGC")
        self.chk_agc.setChecked(True); self.chk_agc.toggled.connect(self._render)
        pp.addWidget(self.chk_agc)
        self.chk_balance = QCheckBox("道间均衡")
        pp.addWidget(self.chk_balance)
        pp.addStretch()
        self.btn_apply_pp = QPushButton("应用预处理")
        self.btn_apply_pp.clicked.connect(self._apply_preprocess)
        self.btn_apply_pp.setStyleSheet(
            "QPushButton { background-color: #FF9800; color: white; padding: 4px 10px; }")
        pp.addWidget(self.btn_apply_pp)
        layout.addLayout(pp)

        # 文件选择 + 色标
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("文件:"))
        self.cmb_file = QComboBox()
        self.cmb_file.currentIndexChanged.connect(self._render)
        ctrl.addWidget(self.cmb_file)
        ctrl.addWidget(QLabel("色标:"))
        self.cmb_cmap = QComboBox()
        self.cmb_cmap.addItems(["gray", "seismic", "viridis", "inferno", "RdYlBu"])
        self.cmb_cmap.currentTextChanged.connect(self._on_cmap)
        ctrl.addWidget(self.cmb_cmap)
        self.chk_raw = QCheckBox("原始")
        self.chk_raw.setChecked(False); self.chk_raw.toggled.connect(self._render)
        ctrl.addWidget(self.chk_raw)
        self.chk_proc = QCheckBox("去噪后")
        self.chk_proc.setChecked(True); self.chk_proc.toggled.connect(self._render)
        ctrl.addWidget(self.chk_proc)
        ctrl.addStretch()
        self.lbl_info = QLabel("")
        self.lbl_info.setStyleSheet("color: #555; font-size: 10px;")
        ctrl.addWidget(self.lbl_info)
        layout.addLayout(ctrl)

        self.figure = Figure(figsize=(12, 8), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

    def set_data(self, seg_list):
        self._seg_list = seg_list
        self._processed_disp = {}
        self._raw_disp = {}
        # 只抽样, 不预处理
        for seg in seg_list:
            step = max(1, seg.n_traces // 600)
            ds = max(1, seg.n_samples // 400)
            self._raw_disp[seg.name] = seg.data[::step, ::ds]
        self.cmb_file.blockSignals(True)
        self.cmb_file.clear()
        self.cmb_file.addItems([s.name for s in seg_list])
        self.cmb_file.blockSignals(False)
        self._render()

    def _apply_preprocess(self):
        """手动触发预处理"""
        if not self._seg_list: return
        steps = self._get_steps()
        if not steps:
            steps = None  # auto
        for seg in self._seg_list:
            sampled = self._raw_disp.get(seg.name)
            if sampled is not None:
                result, used = preprocess_pipeline(sampled, steps)
                self._processed_disp[seg.name] = result
        self.lbl_info.setText(f"预处理完成: {' → '.join(used) if steps else 'auto'}")
        self._render()

    def _get_steps(self):
        steps = []
        if self.chk_demean.isChecked(): steps.append("demean")
        if self.chk_median.isChecked(): steps.append("median")
        if self.chk_gauss.isChecked(): steps.append("gaussian")
        if self.chk_agc.isChecked(): steps.append("agc")
        if self.chk_balance.isChecked(): steps.append("balance")
        return steps

    def _on_cmap(self, name):
        self._cmap = name
        self._render()

    def _render(self, *_):
        # 只在勾选预处理时执行
        steps = tuple(self._get_steps())
        if self._seg_list and steps and steps != self._last_steps:
            self._last_steps = steps
            for seg in self._seg_list:
                sampled = self._raw_disp.get(seg.name)
                if sampled is not None:
                    result, _ = preprocess_pipeline(sampled, list(steps))
                    self._processed_disp[seg.name] = result
        elif not steps:
            self._last_steps = None

        self.figure.clear()
        if not self._seg_list:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "No SEG data. Import from left panel.",
                    ha="center", va="center", color="gray", fontsize=12,
                    transform=ax.transAxes)
            self.canvas.draw(); return

        # 只渲染下拉选中的文件
        name = self.cmb_file.currentText()
        seg = next((s for s in self._seg_list if s.name == name), None)
        if seg is None:
            seg = self._seg_list[0]

        show_raw = self.chk_raw.isChecked()
        show_proc = self.chk_proc.isChecked()
        raw_d = self._raw_disp.get(seg.name)
        proc_d = self._processed_disp.get(seg.name)

        # 单图模式
        ax = self.figure.add_subplot(111)
        if show_proc and proc_d is not None:
            self._plot_seg(ax, proc_d, seg.name, f"Denoised ({'+'.join(steps) if steps else 'none'})")
        else:
            self._plot_seg(ax, raw_d, seg.name, "Raw")

        self.figure.tight_layout()
        self.canvas.draw()

        steps = self._get_steps()
        if not steps:
            # 显示自动检测的方法
            _, steps = preprocess_pipeline(self._seg_list[0].data)
        total_t = sum(s.n_traces for s in self._seg_list)
        self.lbl_info.setText(
            f"{len(self._seg_list)} files, {total_t} traces | "
            f"{' → '.join(steps) if steps else 'none'} | "
            f"勾选=手动, 不勾选=自动"
        )

    def _plot_seg(self, ax, data, name, tag):
        vmin = np.percentile(data, 2)
        vmax = np.percentile(data, 98)
        if vmax - vmin < 0.001:
            vmin, vmax = -1, 1
        ax.imshow(data.T, aspect='auto', cmap=self._cmap,
                  vmin=vmin, vmax=vmax, origin='upper',
                  interpolation='bilinear')
        ax.set_ylabel(f"{name}", fontsize=7)
        ax.set_title(f"{tag}  ({data.shape[0]} traces)", fontsize=8)
        ax.tick_params(labelsize=6)
        if "Denoised" in tag or not self.chk_raw.isChecked():
            ax.set_xlabel("Trace #", fontsize=7)
