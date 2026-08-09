"""
数据导入面板
==========
四类数据导入:
  ① SEG-Y 浅剖数据 (原始数据, 可提取TXT)
  ② TXT 测线数据 (底质分类核心输入)
  ③ Depth 深度底图 (GeoTIFF, 辅助训练)
  ④ 底质分类底图 (GeoTIFF, 分类标签)
"""

import os

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel, QLineEdit, QFileDialog,
    QCheckBox, QSpinBox, QComboBox, QMessageBox,
    QTextEdit,
)
from PyQt5.QtCore import Qt, pyqtSignal

from core.data_loader import DataLoader
from utils.logger import logger


class ImportPanel(QWidget):
    """数据导入面板"""

    dem_loaded = pyqtSignal(object)
    survey_loaded = pyqtSignal(list)
    seg_loaded = pyqtSignal(list)
    data_cleared = pyqtSignal()

    def __init__(self, data_loader, parent=None):
        super().__init__(parent)
        self.data_loader = data_loader
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # ======== ① SEG-Y 浅剖数据 ========
        grp_seg = QGroupBox("① SEG-Y 浅剖数据（原始数据）")
        seg_layout = QVBoxLayout(grp_seg)
        seg_row = QHBoxLayout()
        self.seg_path_edit = QLineEdit()
        self.seg_path_edit.setPlaceholderText("选择 SEG-Y 文件夹...")
        self.seg_path_edit.setReadOnly(True)
        seg_row.addWidget(self.seg_path_edit)
        self.btn_browse_seg = QPushButton("浏览")
        self.btn_browse_seg.clicked.connect(self._browse_seg)
        seg_row.addWidget(self.btn_browse_seg)
        self.btn_clear_seg = QPushButton("清除")
        self.btn_clear_seg.clicked.connect(self._clear_seg)
        seg_row.addWidget(self.btn_clear_seg)
        seg_layout.addLayout(seg_row)
        self.lbl_seg_info = QLabel("未加载 SEG-Y 数据")
        self.lbl_seg_info.setStyleSheet("color: gray; font-size: 10px;")
        seg_layout.addWidget(self.lbl_seg_info)
        layout.addWidget(grp_seg)

        # ======== ② TXT 测线数据（核心） ========
        grp_survey = QGroupBox("② TXT 测线数据（底质分类核心输入）")
        survey_layout = QVBoxLayout(grp_survey)

        h1 = QHBoxLayout()
        self.survey_path_edit = QLineEdit()
        self.survey_path_edit.setPlaceholderText("选择 TXT 文件...")
        self.survey_path_edit.setReadOnly(True)
        self.btn_browse_survey = QPushButton("浏览")
        self.btn_browse_survey.clicked.connect(self._browse_survey)
        self.btn_clear_survey = QPushButton("清除")
        self.btn_clear_survey.clicked.connect(self._clear_survey)
        h1.addWidget(self.survey_path_edit); h1.addWidget(self.btn_browse_survey); h1.addWidget(self.btn_clear_survey)
        survey_layout.addLayout(h1)

        fmt_row = QHBoxLayout()
        fmt_row.addWidget(QLabel("分隔符:"))
        self.cmb_delimiter = QComboBox()
        self.cmb_delimiter.addItems(["自动检测", "逗号 (,)", "制表符 (\\t)", "空格"])
        fmt_row.addWidget(self.cmb_delimiter)
        self.chk_header = QCheckBox("包含标题行")
        fmt_row.addWidget(self.chk_header)
        fmt_row.addStretch()
        survey_layout.addLayout(fmt_row)

        self.lbl_survey_info = QLabel("未加载测线数据")
        self.lbl_survey_info.setStyleSheet("color: gray; font-size: 10px;")
        survey_layout.addWidget(self.lbl_survey_info)

        layout.addWidget(grp_survey)

        # ======== ③ Depth 深度底图 ========
        grp_depth = QGroupBox("③ Depth 深度底图（GeoTIFF，辅助训练）")
        depth_layout = QVBoxLayout(grp_depth)
        h3 = QHBoxLayout()
        self.depth_path_edit = QLineEdit()
        self.depth_path_edit.setPlaceholderText("选择深度底图 .tif...")
        self.depth_path_edit.setReadOnly(True)
        self.btn_browse_depth = QPushButton("浏览")
        self.btn_browse_depth.clicked.connect(self._browse_depth)
        self.btn_clear_depth = QPushButton("清除")
        self.btn_clear_depth.clicked.connect(self._clear_depth)
        h3.addWidget(self.depth_path_edit); h3.addWidget(self.btn_browse_depth); h3.addWidget(self.btn_clear_depth)
        depth_layout.addLayout(h3)
        self.lbl_depth_info = QLabel("未加载深度底图")
        self.lbl_depth_info.setStyleSheet("color: gray; font-size: 10px;")
        depth_layout.addWidget(self.lbl_depth_info)
        layout.addWidget(grp_depth)

        # ======== ④ 底质分类底图 ========
        grp_class = QGroupBox("④ 底质分类底图（GeoTIFF，分类标签）")
        class_layout = QVBoxLayout(grp_class)
        h4 = QHBoxLayout()
        self.classmap_path_edit = QLineEdit()
        self.classmap_path_edit.setPlaceholderText("选择分类底图 .tif...")
        self.classmap_path_edit.setReadOnly(True)
        self.btn_browse_classmap = QPushButton("浏览")
        self.btn_browse_classmap.clicked.connect(self._browse_classmap)
        self.btn_clear_classmap = QPushButton("清除")
        self.btn_clear_classmap.clicked.connect(self._clear_classmap)
        h4.addWidget(self.classmap_path_edit); h4.addWidget(self.btn_browse_classmap); h4.addWidget(self.btn_clear_classmap)
        class_layout.addLayout(h4)
        self.lbl_classmap_info = QLabel("未加载分类底图")
        self.lbl_classmap_info.setStyleSheet("color: gray; font-size: 10px;")
        class_layout.addWidget(self.lbl_classmap_info)
        layout.addWidget(grp_class)

        # ======== 操作 ========
        btn_row = QHBoxLayout()
        self.btn_load_all = QPushButton("加载所有")
        self.btn_load_all.clicked.connect(self._load_all)
        self.btn_clear_all = QPushButton("清空全部")
        self.btn_clear_all.clicked.connect(self._clear_all)
        btn_row.addWidget(self.btn_load_all); btn_row.addWidget(self.btn_clear_all)
        layout.addLayout(btn_row)

        # 摘要
        grp_sum = QGroupBox("数据摘要")
        sum_layout = QVBoxLayout(grp_sum)
        self.txt_summary = QTextEdit()
        self.txt_summary.setReadOnly(True); self.txt_summary.setMaximumHeight(100)
        sum_layout.addWidget(self.txt_summary)
        layout.addWidget(grp_sum)

    # ========== ① SEG-Y ==========

    def _browse_seg(self):
        folder = QFileDialog.getExistingDirectory(self, "选择 SEG-Y 文件夹")
        if not folder: return
        self.seg_path_edit.setText(folder)
        from core.seg_reader import read_seg_folder
        segs = read_seg_folder(folder)
        self.lbl_seg_info.setText(f"[OK] {len(segs)} 个 SEG 文件")
        self.seg_loaded.emit(segs)
        self._update_summary()

    def _clear_seg(self):
        self.seg_path_edit.clear()
        self.lbl_seg_info.setText("未加载 SEG-Y 数据")
        self._update_summary()

    # ========== ② TXT ==========

    def _browse_survey(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择测线数据", "", "文本文件 (*.txt *.csv);;所有文件 (*)")
        if not file_path: return
        self.survey_path_edit.setText(file_path)
        lines = self.data_loader.load_survey_txt(file_path)
        if lines:
            sl = lines[-1]
            self.lbl_survey_info.setText(f"[OK] {len(lines)}条测线, {len(sl.longitude)}点")
            try: self.survey_loaded.emit(lines)
            except Exception: pass
            self._update_summary()
        else:
            self.lbl_survey_info.setText("[FAIL]")

    def _clear_survey(self):
        self.survey_path_edit.clear()
        self.lbl_survey_info.setText("未加载测线数据")
        self.data_loader.survey_lines.clear()
        self._update_summary()

    # ========== ③ Depth ==========

    def _browse_depth(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择深度底图", "", "GeoTIFF (*.tif *.tiff);;所有文件 (*)")
        if file_path:
            self.depth_path_edit.setText(file_path)
            dem = self.data_loader.load_dem(file_path)
            if dem:
                self.lbl_depth_info.setText(f"[OK] {os.path.basename(file_path)} ({dem.raster.shape})")
                self.dem_loaded.emit(dem)
                self._update_summary()
            else:
                self.lbl_depth_info.setText("[FAIL]")

    def _clear_depth(self):
        self.depth_path_edit.clear()
        self.lbl_depth_info.setText("未加载深度底图")
        self.data_loader.dem = None
        self._update_summary()

    # ========== ④ 分类底图 ==========

    def _browse_classmap(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择分类底图", "", "GeoTIFF (*.tif *.tiff);;所有文件 (*)")
        if file_path:
            self.classmap_path_edit.setText(file_path)
            dem = self.data_loader.load_dem(file_path)
            if dem:
                self.lbl_classmap_info.setText(f"[OK] {os.path.basename(file_path)} ({dem.raster.shape})")
                self.dem_loaded.emit(dem)
                self._update_summary()
            else:
                self.lbl_classmap_info.setText("[FAIL]")

    def _clear_classmap(self):
        self.classmap_path_edit.clear()
        self.lbl_classmap_info.setText("未加载分类底图")
        self._update_summary()

    # ========== 操作 ==========

    def _load_all(self):
        if self.survey_path_edit.text():
            self.data_loader.load_survey_txt(self.survey_path_edit.text())
        if self.depth_path_edit.text():
            self.data_loader.load_dem(self.depth_path_edit.text())
        if self.classmap_path_edit.text():
            self.data_loader.load_dem(self.classmap_path_edit.text())
        self._update_summary()

    def _clear_all(self):
        reply = QMessageBox.question(self, "确认", "清空所有数据？", QMessageBox.Yes|QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes: return
        self.data_loader.clear()
        self._clear_survey(); self._clear_depth(); self._clear_classmap()
        self.txt_summary.clear()
        self.data_cleared.emit()

    def _update_summary(self):
        dl = self.data_loader
        lines = []
        if dl.dem: lines.append(f"底图: {os.path.basename(dl.dem.file_path)} ({dl.dem.raster.shape})")
        if dl.survey_lines:
            total = sum(len(sl.longitude) for sl in dl.survey_lines)
            lines.append(f"测线: {len(dl.survey_lines)}条, {total}点")
        self.txt_summary.setText("\n".join(lines) if lines else "暂无数据")
