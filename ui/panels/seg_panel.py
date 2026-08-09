"""
SEG 数据导入与选择面板
====================
选择文件夹 → 列出所有 SEG 文件 → 勾选 → 可视化选中剖面
"""

import os
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel, QLineEdit, QFileDialog,
    QListWidget, QListWidgetItem, QCheckBox, QMessageBox,
)
from PyQt5.QtCore import Qt, pyqtSignal

from core.seg_reader import read_seg_folder


class SegPanel(QWidget):
    """SEG 文件导入面板"""

    seg_loaded = pyqtSignal(object)     # list of SegData (selected)
    seg_all_loaded = pyqtSignal(list)   # list of SegData (all)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._all_seg_data = []  # 所有加载的 SEG
        self._checkboxes = []    # 复选框列表
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        grp = QGroupBox("SEG 浅剖数据")
        seg_layout = QVBoxLayout(grp)

        # 文件夹选择
        path_row = QHBoxLayout()
        self.edit_path = QLineEdit()
        self.edit_path.setPlaceholderText("选择 SEG 文件夹...")
        self.edit_path.setReadOnly(True)
        path_row.addWidget(self.edit_path)
        btn_browse = QPushButton("浏览")
        btn_browse.clicked.connect(self._browse_folder)
        path_row.addWidget(btn_browse)
        seg_layout.addLayout(path_row)

        # 文件列表 (带复选框)
        self.list_files = QListWidget()
        self.list_files.setMaximumHeight(180)
        seg_layout.addWidget(self.list_files)

        # 全选/取消
        sel_row = QHBoxLayout()
        btn_all = QPushButton("全选")
        btn_all.clicked.connect(lambda: self._toggle_all(True))
        sel_row.addWidget(btn_all)
        btn_none = QPushButton("取消")
        btn_none.clicked.connect(lambda: self._toggle_all(False))
        sel_row.addWidget(btn_none)
        sel_row.addStretch()
        seg_layout.addLayout(sel_row)

        # 可视化按钮
        btn_vis = QPushButton("可视化选中SEG")
        btn_vis.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; padding: 6px; font-weight: bold; }")
        btn_vis.clicked.connect(self._visualize)
        seg_layout.addWidget(btn_vis)

        self.lbl_info = QLabel("未加载 SEG 数据")
        self.lbl_info.setStyleSheet("color: gray; font-size: 10px;")
        seg_layout.addWidget(self.lbl_info)

        layout.addWidget(grp)
        layout.addStretch()

    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择 SEG 文件夹")
        if not folder:
            return
        self.edit_path.setText(folder)
        self._load_folder(folder)

    def _load_folder(self, folder):
        self.lbl_info.setText("加载中...")
        self.list_files.clear()
        self._checkboxes.clear()

        try:
            self._all_seg_data = read_seg_folder(folder)
        except Exception as e:
            QMessageBox.warning(self, "加载失败", str(e))
            self.lbl_info.setText("加载失败")
            return

        for seg in self._all_seg_data:
            item = QListWidgetItem()
            cb = QCheckBox(f"{seg.name}  ({seg.n_traces}道 × {seg.n_samples}采样)")
            cb.setChecked(True)
            self.list_files.addItem(item)
            self.list_files.setItemWidget(item, cb)
            self._checkboxes.append(cb)

        self.lbl_info.setText(f"已加载 {len(self._all_seg_data)} 个 SEG 文件")
        self.seg_all_loaded.emit(self._all_seg_data)

    def _toggle_all(self, checked):
        for cb in self._checkboxes:
            cb.setChecked(checked)

    def _visualize(self):
        selected = [self._all_seg_data[i] for i, cb in enumerate(self._checkboxes) if cb.isChecked()]
        if not selected:
            QMessageBox.information(self, "提示", "请至少勾选一个 SEG 文件")
            return
        self.lbl_info.setText(f"可视化 {len(selected)} 个文件")
        self.seg_loaded.emit(selected)
