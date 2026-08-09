"""
主窗口
=====
整合所有子面板和可视化组件的顶层窗口。
"""

import os
import sys
from datetime import datetime

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QMenuBar, QMenu, QAction, QStatusBar, QMessageBox,
    QDockWidget, QTabWidget, QLabel, QApplication, QFileDialog,
)
from PyQt5.QtCore import Qt, QSettings
from PyQt5.QtGui import QIcon

from core.data_loader import DataLoader
from core.data_processor import DataProcessor
from core.classifier import SedimentClassifier
from core.exporter import Exporter

from ui.panels.import_panel import ImportPanel
from ui.panels.classification_panel import ClassificationPanel
from ui.panels.export_panel import ExportPanel
from ui.widgets.map_view import MapView

from utils.logger import logger

APP_NAME = "DeepSeaSedimentClassifier"
APP_VERSION = "1.0.0"
APP_AUTHOR = "DeepSea Lab"


class MainWindow(QMainWindow):
    """主窗口"""

    def __init__(self):
        super().__init__()

        # ---- 核心模块 ----
        self.data_loader = DataLoader()
        self.data_processor = DataProcessor()
        self.classifier = SedimentClassifier()
        self.exporter = Exporter()

        # ---- UI 组件 ----
        self.import_panel: ImportPanel = None
        self.classification_panel: ClassificationPanel = None
        self.export_panel: ExportPanel = None
        self.map_view: MapView = None

        self._init_ui()
        self._connect_signals()
        self._restore_state()

        logger.info(f"{APP_NAME} v{APP_VERSION} 已启动")

    # ==================== UI 初始化 ====================

    def _init_ui(self):
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION} - Seafloor Sediment Classifier")
        self.setMinimumSize(1200, 800)
        self.resize(1600, 950)

        # 窗口图标
        import os
        icon_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "icon.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        # ---- 菜单栏 ----
        self._create_menus()

        # ---- 状态栏 ----
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪 | 请导入测线数据开始")

        # ---- 中心区域: 可视化 (对应四类数据) ----
        self.central_tab = QTabWidget()
        self.central_tab.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #ccc; background: white; }
            QTabBar::tab { padding: 6px 16px; font-size: 11px; font-weight: bold; }
            QTabBar::tab:selected { background: #1565C0; color: white; }
        """)

        self.map_view = MapView()
        self.central_tab.addTab(self.map_view, "测线航迹图")

        from ui.panels.profile_panel import ProfilePanel
        self.profile_panel = ProfilePanel()
        self.central_tab.addTab(self.profile_panel, "海底声学切面")

        from ui.widgets.seg_view import SegView
        self.seg_view = SegView()
        self.central_tab.addTab(self.seg_view, "SEG浅剖剖面")

        self.setCentralWidget(self.central_tab)

        # ---- 左侧面板: 导入 + 剖面 + 分类 (可滚动) ----
        from PyQt5.QtWidgets import QSplitter, QScrollArea
        left_splitter = QSplitter(Qt.Vertical)
        left_splitter.setChildrenCollapsible(False)

        self.import_panel = ImportPanel(self.data_loader)
        left_splitter.addWidget(self.import_panel)

        from ui.panels.seg_panel import SegPanel
        self.seg_panel = SegPanel()
        left_splitter.addWidget(self.seg_panel)

        self.classification_panel = ClassificationPanel(self.data_processor)
        self.classification_panel.data_loader = self.data_loader
        left_splitter.addWidget(self.classification_panel)

        left_splitter.setSizes([320, 180, 400])

        # 包裹在滚动区域中
        scroll = QScrollArea()
        scroll.setWidget(left_splitter)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        left_dock = QDockWidget("数据导入 & 分类控制", self)
        left_dock.setWidget(scroll)
        left_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        left_dock.setMinimumWidth(400)
        self.addDockWidget(Qt.LeftDockWidgetArea, left_dock)

        # ---- 右侧面板: 导出 ----
        self.export_panel = ExportPanel(self.exporter)
        right_dock = QDockWidget("成果导出", self)
        right_dock.setWidget(self.export_panel)
        right_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        right_dock.setMinimumWidth(280)
        right_dock.setMaximumWidth(420)
        self.addDockWidget(Qt.RightDockWidgetArea, right_dock)

    def _create_menus(self):
        """创建菜单栏"""
        menubar = self.menuBar()

        # ---- 文件 ----
        file_menu = menubar.addMenu("文件(&F)")

        act_import_dem = QAction("导入 DEM 底图...", self)
        act_import_dem.setShortcut("Ctrl+D")
        act_import_dem.triggered.connect(self._menu_import_dem)
        file_menu.addAction(act_import_dem)

        act_import_survey = QAction("导入测线 TXT...", self)
        act_import_survey.setShortcut("Ctrl+T")
        act_import_survey.triggered.connect(self._menu_import_survey)
        file_menu.addAction(act_import_survey)

        act_export = QAction("导出分类结果...", self)
        act_export.setShortcut("Ctrl+E")
        act_export.triggered.connect(self._menu_export)
        file_menu.addAction(act_export)

        file_menu.addSeparator()

        act_exit = QAction("退出(&X)", self)
        act_exit.setShortcut("Ctrl+Q")
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

        # ---- 处理 ----
        process_menu = menubar.addMenu("处理(&P)")

        act_extract = QAction("提取特征", self)
        act_extract.setShortcut("F5")
        act_extract.triggered.connect(self._extract_features)
        process_menu.addAction(act_extract)

        act_train = QAction("开始训练", self)
        act_train.setShortcut("F6")
        act_train.triggered.connect(self._start_training)
        process_menu.addAction(act_train)

        act_stop = QAction("停止训练", self)
        act_stop.setShortcut("F7")
        act_stop.triggered.connect(self._stop_training)
        process_menu.addAction(act_stop)

        process_menu.addSeparator()

        act_save_model = QAction("保存模型...", self)
        act_save_model.setShortcut("Ctrl+S")
        act_save_model.triggered.connect(self._save_model)
        process_menu.addAction(act_save_model)

        act_load_model = QAction("加载模型...", self)
        act_load_model.setShortcut("Ctrl+O")
        act_load_model.triggered.connect(self._load_model)
        process_menu.addAction(act_load_model)

        # ---- 视图 ----
        view_menu = menubar.addMenu("视图(&V)")

        act_reset_layout = QAction("重置窗口布局", self)
        act_reset_layout.triggered.connect(self._reset_layout)
        view_menu.addAction(act_reset_layout)

        # ---- 帮助 ----
        help_menu = menubar.addMenu("帮助(&H)")

        act_about = QAction("关于", self)
        act_about.triggered.connect(self._show_about)
        help_menu.addAction(act_about)

        act_usage = QAction("使用说明", self)
        act_usage.triggered.connect(self._show_usage)
        help_menu.addAction(act_usage)

    # ==================== 信号连接 ====================

    def _connect_signals(self):
        # 导入 -> 可视化
        self.import_panel.dem_loaded.connect(self.map_view.set_dem)
        self.import_panel.seg_loaded.connect(self.seg_view.set_data)
        self.import_panel.seg_loaded.connect(lambda _: self.central_tab.setCurrentIndex(2))
        self.import_panel.survey_loaded.connect(self._safe_map_update)
        self.import_panel.survey_loaded.connect(self._on_survey_loaded)
        self.import_panel.survey_loaded.connect(self._safe_profile_update)
        # SEG 面板 → SEG 视图
        self.seg_panel.seg_loaded.connect(self.seg_view.set_data)
        self.seg_panel.seg_loaded.connect(lambda _: self.central_tab.setCurrentIndex(2))
        # 地图点击 → 弹出剖面
        self.map_view.point_clicked.connect(self._on_map_point_clicked)

        self.import_panel.data_cleared.connect(self._on_data_cleared)

        # 分类 -> 导出 + 可视化
        self.classification_panel.predictions_ready.connect(self.export_panel.set_results)
        self.classification_panel.predictions_ready.connect(self.map_view.set_classification_results)
        self.classification_panel.training_finished.connect(
            lambda r: self.map_view.set_training_history(
                self.classification_panel.get_training_history()
            )
        )
        self.classification_panel.training_started.connect(
            lambda: self.status_bar.showMessage("训练中...")
        )
        self.classification_panel.training_finished.connect(
            lambda r: self.status_bar.showMessage(
                f"训练完成 | 最佳 Val Loss: {r['best_val_loss']:.4f}"
            )
        )
        self.classification_panel.training_error.connect(
            lambda msg: self.status_bar.showMessage(f"训练错误: {msg}")
        )

        # 导出
        self.export_panel.export_completed.connect(
            lambda path: self.status_bar.showMessage(f"已导出: {path}")
        )

    # ==================== 菜单动作 ====================

    def _menu_import_dem(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "导入 DEM 底图", "", "GeoTIFF (*.tif *.tiff);;所有文件 (*)"
        )
        if file_path:
            self.data_loader.load_dem(file_path)
            self.import_panel._update_summary()
            if self.data_loader.dem:
                self.map_view.set_dem(self.data_loader.dem)

    def _menu_import_survey(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "导入测线 TXT", "", "文本文件 (*.txt *.csv);;所有文件 (*)"
        )
        if file_path:
            self.data_loader.load_survey_txt(file_path)
            self.import_panel._update_summary()
            self.map_view.set_survey_lines(self.data_loader.survey_lines)

    def _menu_export(self):
        self.export_panel._export_txt()

    def _extract_features(self):
        if not self.data_loader.survey_lines:
            QMessageBox.warning(self, "无数据", "请先导入测线数据。")
            return
        processed = self.data_processor.extract_features(
            self.data_loader.survey_lines, self.data_loader.dem
        )
        if processed and processed.num_samples > 0:
            norm_features = self.data_processor.get_normalized_features()
            self.classification_panel.set_features(norm_features)
            self.status_bar.showMessage(
                f"特征提取完成: {processed.num_features} 个特征, {processed.num_samples} 条记录"
            )
            QMessageBox.information(
                self, "特征提取完成",
                f"成功提取 {processed.num_features} 个特征:\n"
                + "\n".join(f"  - {n}" for n in processed.feature_names)
                + f"\n\n共 {processed.num_samples} 条记录。\n请设置超参数后开始训练。"
            )

    def _start_training(self):
        if self.data_processor.processed is None:
            # 自动提取特征
            self._extract_features()
            if self.data_processor.processed is None:
                return
        self.classification_panel._start_training()

    def _stop_training(self):
        self.classification_panel._stop_training()

    def _save_model(self):
        if not self.classification_panel.classifier or not self.classification_panel.classifier.is_trained:
            QMessageBox.warning(self, "无模型", "请先训练模型。")
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存模型", "msc_transformer_model.pt", "PyTorch 模型 (*.pt *.pth)"
        )
        if file_path:
            self.classification_panel.classifier.save(file_path)

    def _load_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "加载模型", "", "PyTorch 模型 (*.pt *.pth);;所有文件 (*)"
        )
        if file_path:
            self.classification_panel.classifier = SedimentClassifier()
            self.classification_panel.classifier.load(file_path)
            self.status_bar.showMessage(f"模型已加载: {file_path}")
            # 如果有数据，执行推理
            if self.classification_panel.features_normalized is not None:
                self.classification_panel._run_inference()

    def _on_map_point_clicked(self, survey_line, distance):
        """航迹图点击 → 弹出 Backscatter Profile 剖面窗口"""
        try:
            from ui.dialogs.profile_dialog import ProfileDialog
            tif_path = self.data_loader.dem.file_path if self.data_loader.dem else None
            dlg = ProfileDialog(survey_line, distance, tif_path=tif_path, parent=self)
            dlg.setAttribute(Qt.WA_DeleteOnClose)
            dlg.show()
        except Exception as e:
            import traceback
            QMessageBox.critical(self, "剖面错误",
                f"无法创建剖面窗口:\n{traceback.format_exc()[-600:]}")
            logger.exception(f"Profile dialog failed: {e}")

    def _safe_dem_update(self, dem):
        try:
            self.map_view.set_dem(dem)
        except Exception:
            pass

    def _safe_map_update(self, lines):
        try:
            self.map_view.set_survey_lines(lines)
        except Exception:
            pass

    def _safe_profile_update(self, lines):
        try:
            self.profile_panel.set_survey_lines(lines)
        except Exception:
            pass

    def _on_survey_loaded(self, lines):
        """测线加载后自动切换到海底切面标签页"""
        self.central_tab.setCurrentIndex(1)
        self.status_bar.showMessage(
            f"测线已加载: {len(lines)} 条 | 查看「海底声学切面」"
        )

    def _on_data_cleared(self):
        self.map_view.clear_all()
        self.export_panel.clear()

    def _reset_layout(self):
        """重置窗口布局"""
        self.resize(1600, 1000)
        QMessageBox.information(self, "已重置", "窗口布局已重置。")

    def _show_about(self):
        QMessageBox.about(
            self, f"关于 {APP_NAME}",
            f"<h2>{APP_NAME} v{APP_VERSION}</h2>"
            f"<p>深海底质分类系统</p>"
            f"<p>基于 MSC-Transformer 模型 (JMSE 2023):<br>"
            f"<i>Research on Seabed Sediment Classification Based on "
            f"the MSC-Transformer and Sub-Bottom Profiler</i></p>"
            f"<hr>"
            f"<p><b>功能:</b></p>"
            f"<ul>"
            f"<li>DEM 底图 + 测线数据导入</li>"
            f"<li>SEPY 浅地层剖面可视化</li>"
            f"<li>MSC-Transformer 沉积物自动分类</li>"
            f"<li>GPS 标记分类图输出</li>"
            f"</ul>"
            f"<p>作者: {APP_AUTHOR}</p>"
        )

    def _show_usage(self):
        QMessageBox.information(
            self, "使用说明",
            "<h3>使用步骤</h3>"
            "<ol>"
            "<li><b>导入测线数据:</b> 在左侧面板选择带 GPS 和反射强度的 TXT 文件</li>"
            "<li><b>(可选) 导入 DEM 底图:</b> 加载 GeoTIFF 格式海底地形图</li>"
            "<li><b>导入 SEG-Y 剖面:</b> 加载浅地层剖面数据</li>"
            "<li><b>提取特征:</b> 点击菜单 处理->提取特征 或按 F5</li>"
            "<li><b>设置超参数:</b> 调整学习率、Epochs、Batch Size 等</li>"
            "<li><b>开始训练:</b> 点击 '开始训练' 或按 F6</li>"
            "<li><b>查看结果:</b> 在 '分类结果' 标签页查看空间分布</li>"
            "<li><b>导出成果:</b> 在右侧面板导出 GPS 分类 TXT</li>"
            "</ol>"
            "<p><b>快捷操作:</b></p>"
            "<ul>"
            "<li>F5: 提取特征</li>"
            "<li>F6: 开始训练</li>"
            "<li>F7: 停止训练</li>"
            "<li>Ctrl+S: 保存模型</li>"
            "<li>Ctrl+O: 加载模型</li>"
            "<li>Ctrl+E: 导出结果</li>"
            "</ul>"
        )

    # ==================== 状态持久化 ====================

    def _restore_state(self):
        """恢复窗口状态"""
        settings = QSettings(APP_AUTHOR, APP_NAME)
        geometry = settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        state = settings.value("windowState")
        if state:
            self.restoreState(state)

    def closeEvent(self, event):
        """关闭时保存状态"""
        settings = QSettings(APP_AUTHOR, APP_NAME)
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("windowState", self.saveState())
        logger.info(f"{APP_NAME} 已关闭")
        event.accept()
