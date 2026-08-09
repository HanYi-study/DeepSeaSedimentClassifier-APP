"""
成果导出面板
==========
将分类结果导出为 GPS 标记的分类图 TXT 文件。
"""

import os
import numpy as np
from datetime import datetime

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QPushButton, QLineEdit, QFileDialog,
    QTextEdit, QCheckBox, QMessageBox, QProgressBar,
)
from PyQt5.QtCore import Qt, pyqtSignal

from core.exporter import Exporter
from config.settings import SEDIMENT_CLASSES
from utils.logger import logger


class ExportPanel(QWidget):
    """导出面板"""

    export_completed = pyqtSignal(str)  # 导出文件路径

    def __init__(self, exporter: Exporter, parent=None):
        super().__init__(parent)
        self.exporter = exporter

        # 待导出数据
        self._coordinates: np.ndarray = None
        self._predictions: np.ndarray = None
        self._probabilities: np.ndarray = None

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # ======== 输出路径 ========
        grp_output = QGroupBox("输出设置")
        output_layout = QVBoxLayout(grp_output)

        path_layout = QHBoxLayout()
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setPlaceholderText("输出文件路径 (自动生成)...")
        self.output_path_edit.setReadOnly(True)
        self.btn_browse_output = QPushButton("选择目录...")
        self.btn_browse_output.clicked.connect(self._browse_output)
        path_layout.addWidget(self.output_path_edit)
        path_layout.addWidget(self.btn_browse_output)
        output_layout.addLayout(path_layout)

        # 输出格式选项
        self.chk_header = QCheckBox("包含列标题")
        self.chk_header.setChecked(True)
        output_layout.addWidget(self.chk_header)

        self.chk_metadata = QCheckBox("包含元数据注释 (# 开头的行)")
        self.chk_metadata.setChecked(True)
        output_layout.addWidget(self.chk_metadata)

        self.chk_probability = QCheckBox("输出各类别概率")
        self.chk_probability.setChecked(True)
        output_layout.addWidget(self.chk_probability)

        layout.addWidget(grp_output)

        # ======== 分类统计预览 ========
        grp_stats = QGroupBox("分类统计")
        stats_layout = QVBoxLayout(grp_stats)
        self.txt_stats = QTextEdit()
        self.txt_stats.setReadOnly(True)
        self.txt_stats.setMaximumHeight(180)
        self.txt_stats.setPlaceholderText("执行分类后，统计信息将在此显示...")
        stats_layout.addWidget(self.txt_stats)
        layout.addWidget(grp_stats)

        # ======== 导出按钮 ========
        btn_layout = QHBoxLayout()

        self.btn_export_txt = QPushButton("[TXT] 导出 GPS 分类 TXT")
        self.btn_export_txt.setStyleSheet(
            "QPushButton { background-color: #FF9800; color: white; padding: 8px; font-weight: bold; }"
            "QPushButton:disabled { background-color: #ccc; }"
        )
        self.btn_export_txt.clicked.connect(self._export_txt)
        self.btn_export_txt.setEnabled(False)
        btn_layout.addWidget(self.btn_export_txt)

        self.btn_export_report = QPushButton("[RPT] 导出统计报告")
        self.btn_export_report.clicked.connect(self._export_report)
        self.btn_export_report.setEnabled(False)
        btn_layout.addWidget(self.btn_export_report)

        layout.addLayout(btn_layout)

        self.btn_export_png = QPushButton("导出分类地图 (PNG)")
        self.btn_export_png.clicked.connect(self._export_map_png)
        self.btn_export_png.setEnabled(False)
        layout.addWidget(self.btn_export_png)

        # ======== 状态 ========
        self.lbl_export_status = QLabel("等待分类结果...")
        self.lbl_export_status.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(self.lbl_export_status)

        layout.addStretch()

    # ==================== 槽函数 ====================

    def set_results(self, coordinates, predictions, probabilities=None):
        """接收分类结果"""
        self._coordinates = coordinates
        self._predictions = predictions
        self._probabilities = probabilities

        self.btn_export_txt.setEnabled(True)
        self.btn_export_report.setEnabled(True)
        self.btn_export_png.setEnabled(True)

        self._update_stats()

    def _update_stats(self):
        """更新统计预览"""
        if self._predictions is None:
            return

        n = len(self._predictions)
        lines = []
        lines.append(f"总样本数: {n}")
        if self._coordinates is not None:
            lines.append(
                f"经度范围: [{self._coordinates[:, 0].min():.6f}, "
                f"{self._coordinates[:, 0].max():.6f}]"
            )
            lines.append(
                f"纬度范围: [{self._coordinates[:, 1].min():.6f}, "
                f"{self._coordinates[:, 1].max():.6f}]"
            )
        lines.append("")
        lines.append(f"{'类别':<32} {'数量':>8} {'比例':>10}")
        lines.append("-" * 50)
        for cls_id, cls_name in SEDIMENT_CLASSES.items():
            count = int(np.sum(self._predictions == cls_id))
            ratio = count / n * 100
            bar = "#" * int(ratio / 2)
            lines.append(f"{cls_name:<32} {count:>8} {ratio:>9.1f}% {bar}")
        lines.append("-" * 50)

        if self._probabilities is not None:
            conf = np.max(self._probabilities, axis=1)
            lines.append(f"\n平均置信度: {conf.mean():.4f}")
            lines.append(f"低置信度样本 (<0.5): {(conf < 0.5).sum()} ({(conf < 0.5).sum()/n*100:.1f}%)")

        self.txt_stats.setText("\n".join(lines))

    def _browse_output(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if dir_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_path_edit.setText(
                os.path.join(dir_path, f"seafloor_classification_{timestamp}.txt")
            )

    def _get_output_path(self, default_ext=".txt"):
        path = self.output_path_edit.text()
        if not path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"seafloor_classification_{timestamp}{default_ext}"
        return path

    def _export_txt(self):
        """导出 GPS 分类 TXT"""
        if self._predictions is None:
            QMessageBox.warning(self, "无数据", "请先执行分类。")
            return

        output_path = self._get_output_path()
        probs = self._probabilities if self.chk_probability.isChecked() else None

        try:
            result_path = self.exporter.export_classification_txt(
                coordinates=self._coordinates,
                predictions=self._predictions,
                probabilities=probs,
                output_path=output_path,
                include_header=self.chk_header.isChecked(),
                include_metadata=self.chk_metadata.isChecked(),
            )
            self.lbl_export_status.setText(f"[OK] 已导出: {result_path}")
            self.export_completed.emit(result_path)
            QMessageBox.information(
                self, "导出成功",
                f"分类结果已成功导出:\n{result_path}\n\n"
                f"共 {len(self._predictions)} 条记录。"
            )
        except Exception as e:
            logger.exception(f"导出失败: {e}")
            QMessageBox.critical(self, "导出失败", str(e))

    def _export_report(self):
        if self._predictions is None:
            return

        output_path = self._get_output_path().replace(".txt", "_report.txt")
        try:
            result_path = self.exporter.export_statistics_report(
                coordinates=self._coordinates,
                predictions=self._predictions,
                probabilities=self._probabilities,
                output_path=output_path,
            )
            self.lbl_export_status.setText(f"[OK] 报告已导出: {result_path}")
            QMessageBox.information(self, "导出成功", f"统计报告已导出:\n{result_path}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))

    def _export_map_png(self):
        """导出分类结果地图为 PNG 图片"""
        if self._predictions is None:
            QMessageBox.warning(self, "无数据", "请先执行分类。")
            return

        output_path = self._get_output_path().replace(".txt", "_map.png")
        try:
            # 用 matplotlib 渲染分类地图并保存
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from config.settings import SEDIMENT_COLORS

            fig, ax = plt.subplots(figsize=(12, 10), dpi=150)
            for cls_id, cls_name in SEDIMENT_CLASSES.items():
                mask = self._predictions == cls_id
                if mask.any():
                    rgb = tuple(c/255.0 for c in SEDIMENT_COLORS.get(cls_id, (128,128,128)))
                    ax.scatter(self._coordinates[mask, 0], self._coordinates[mask, 1],
                               c=[rgb], s=5, alpha=0.7, label=cls_name, edgecolors='none')

            ax.set_xlabel("Longitude", fontsize=11)
            ax.set_ylabel("Latitude", fontsize=11)
            ax.set_title(f"Seafloor Sediment Classification ({len(self._predictions)} points)",
                         fontsize=14, fontweight="bold")
            ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=7, framealpha=0.8)
            ax.grid(True, alpha=0.2, linestyle='--')
            ax.set_aspect('equal')
            fig.tight_layout(pad=1.0, rect=[0, 0, 0.85, 1])
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            self.lbl_export_status.setText(f"[OK] 地图已导出: {output_path}")
            self.export_completed.emit(output_path)
            QMessageBox.information(self, "导出成功",
                f"分类地图已导出:\n{output_path}\n\n{len(self._predictions)} 个分类点")
        except Exception as e:
            logger.exception(f"地图导出失败: {e}")
            QMessageBox.critical(self, "导出失败", str(e))

    def clear(self):
        self._coordinates = None
        self._predictions = None
        self._probabilities = None
        self.btn_export_txt.setEnabled(False)
        self.btn_export_report.setEnabled(False)
        self.btn_export_png.setEnabled(False)
        self.txt_stats.clear()
        self.lbl_export_status.setText("等待分类结果...")
