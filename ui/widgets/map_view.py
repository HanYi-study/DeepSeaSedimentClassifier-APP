"""
地图/图表可视化组件
==================
- 测线航迹完整显示 (含拐弯)
- DEM 底图叠加
- 训练损失/精度曲线
- 分类结果空间分布图
"""

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QLabel,
    QCheckBox,
)
from PyQt5.QtCore import Qt, pyqtSignal

from config.settings import (
    FIGURE_DPI, MAP_FIGSIZE, SEDIMENT_CLASSES_EN, SEDIMENT_COLORS,
)
from ui.widgets.profile_view import _distance_along_track

# 高对比度航线色板
TRACK_COLORS = [
    "#E53935", "#1E88E5", "#43A047", "#FB8C00", "#8E24AA",
    "#00ACC1", "#D81B60", "#5E35B1", "#3949AB", "#00897B",
    "#C0CA33", "#6D4C41", "#546E7A", "#F4511E", "#039BE5",
]


class MapView(QWidget):
    """
    多视图可视化组件:
      Tab 1: 测线航迹地图 (完整测线, 含拐弯)
      Tab 2: 训练曲线 (Loss & Accuracy)
      Tab 3: 分类结果空间分布
    """

    # 信号: 用户点击航迹图上某点
    point_clicked = pyqtSignal(object, float)  # (SurveyLine, distance_m)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._dem_data = None
        self._survey_lines = []
        self._classification_coords = None
        self._classification_preds = None
        self._training_history = None
        self._show_point_markers = False
        self._cached_distances = []  # 每条测线的沿航迹距离

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # ---- 控制栏 ----
        ctrl_row = QHBoxLayout()

        self.chk_markers = QCheckBox("显示首尾标记")
        self.chk_markers.setChecked(True)
        self.chk_markers.setToolTip("在每条测线的起点(圆)和终点(方)加标记")
        self.chk_markers.toggled.connect(self._on_mode_changed)
        ctrl_row.addWidget(self.chk_markers)

        self.chk_points = QCheckBox("显示采样点")
        self.chk_points.setChecked(False)
        self.chk_points.setToolTip("在每个 GPS 位置画点 (数据量大时可能很密)")
        self.chk_points.toggled.connect(self._on_mode_changed)
        ctrl_row.addWidget(self.chk_points)

        self.lbl_track_info = QLabel("")
        self.lbl_track_info.setStyleSheet("color: #555; font-size: 11px;")
        ctrl_row.addWidget(self.lbl_track_info)
        ctrl_row.addStretch()

        layout.addLayout(ctrl_row)

        # ---- 选项卡 ----
        self.tab_widget = QTabWidget()

        # Tab 1: 测线航迹
        self.map_tab = QWidget()
        map_layout = QVBoxLayout(self.map_tab)
        self.map_figure = Figure(figsize=MAP_FIGSIZE, dpi=FIGURE_DPI)
        self.map_canvas = FigureCanvas(self.map_figure)
        self.map_toolbar = NavigationToolbar(self.map_canvas, self)
        map_layout.addWidget(self.map_toolbar)
        map_layout.addWidget(self.map_canvas)
        self.tab_widget.addTab(self.map_tab, "测线航迹")

        # Tab 2: 训练曲线
        self.train_tab = QWidget()
        train_layout = QVBoxLayout(self.train_tab)
        self.train_figure = Figure(figsize=(8, 5), dpi=FIGURE_DPI)
        self.train_canvas = FigureCanvas(self.train_figure)
        self.train_toolbar = NavigationToolbar(self.train_canvas, self)
        train_layout.addWidget(self.train_toolbar)
        train_layout.addWidget(self.train_canvas)
        self.tab_widget.addTab(self.train_tab, "训练曲线")

        # Tab 3: 分类结果
        self.result_tab = QWidget()
        result_layout = QVBoxLayout(self.result_tab)
        self.result_figure = Figure(figsize=MAP_FIGSIZE, dpi=FIGURE_DPI)
        self.result_canvas = FigureCanvas(self.result_figure)
        self.result_toolbar = NavigationToolbar(self.result_canvas, self)
        result_layout.addWidget(self.result_toolbar)
        result_layout.addWidget(self.result_canvas)
        self.tab_widget.addTab(self.result_tab, "分类结果")

        layout.addWidget(self.tab_widget)

        # 绑定双击事件 (用两次点击间隔 <400ms 模拟)
        self._last_click_time = 0
        self.map_canvas.mpl_connect("button_press_event", self._on_map_click)

    def _on_mode_changed(self):
        self._show_point_markers = self.chk_points.isChecked()
        self._render_map()

    # ==================== Tab 1: 测线航迹地图 ====================

    def set_dem(self, dem_data):
        self._dem_data = dem_data
        self._render_map()

    def set_survey_lines(self, survey_lines):
        self._survey_lines = survey_lines
        self._render_map()

    def _render_map(self):
        """渲染完整测线航迹 (保留所有点包括拐弯)"""
        self.map_figure.clear()
        ax = self.map_figure.add_subplot(111)

        all_lon = []
        all_lat = []
        total_points = 0
        has_content = False

        # ---- DEM 底图 (大图自动降采样) ----
        if self._dem_data is not None and self._dem_data.raster.size > 0:
            try:
                raster = self._dem_data.raster
                # 超过 200万像素时降采样，避免 matplotlib 渲染卡死
                if raster.size > 2_000_000:
                    step = max(1, int(np.sqrt(raster.size / 2_000_000)))
                    raster = raster[::step, ::step]
                left, bottom, right, top = self._dem_data.bounds
                ax.imshow(
                    raster,
                    extent=[left, right, bottom, top],
                    cmap="terrain",
                    alpha=0.7,
                    aspect="auto",
                    zorder=0,
                )
            except Exception:
                pass

        # ---- 逐条测线绘制 (完整, 不改数据) ----
        self._track_lines = []  # 存 (line_obj, lon, lat, name, color)
        for i, sl in enumerate(self._survey_lines):
            lon = np.asarray(sl.longitude, dtype=float)
            lat = np.asarray(sl.latitude, dtype=float)
            if len(lon) < 2:
                continue

            total_points += len(lon)
            all_lon.extend(lon)
            all_lat.extend(lat)

            color = TRACK_COLORS[i % len(TRACK_COLORS)]

            # 画完整连线 (无 label, 无 legend)
            line = ax.plot(lon, lat, '-', color=color, linewidth=1.5,
                           alpha=0.85, solid_capstyle='round', zorder=2)[0]
            self._track_lines.append((line, lon, lat, sl.name, color))

            # 采样点
            if self._show_point_markers:
                ax.scatter(lon, lat, s=3, color=color, alpha=0.3,
                           edgecolors='none', zorder=3)

            # 首尾标记
            if self.chk_markers.isChecked():
                ax.scatter(lon[0], lat[0], s=50, color=color,
                           marker='o', edgecolors='white', linewidths=0.8,
                           zorder=5)
                ax.scatter(lon[-1], lat[-1], s=60, color=color,
                           marker='s', edgecolors='white', linewidths=0.8,
                           zorder=5)

            has_content = True

        # ---- hover 标注 ----
        self._hover_annot = ax.annotate(
            "", xy=(0, 0), xytext=(10, 10), textcoords="offset points",
            fontsize=8, color="white", backgroundcolor=(0, 0, 0, 0.75),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.75),
            zorder=100, visible=False,
        )
        self.map_canvas.mpl_connect("motion_notify_event", self._on_hover)

        # ---- 坐标轴 ----
        if has_content:
            all_lon = np.array(all_lon)
            all_lat = np.array(all_lat)
            lon_span = all_lon.max() - all_lon.min() or 0.001
            lat_span = all_lat.max() - all_lat.min() or 0.001
            m = 0.08  # 8% 边距

            ax.set_xlim(all_lon.min() - lon_span * m,
                         all_lon.max() + lon_span * m)
            ax.set_ylim(all_lat.min() - lat_span * m,
                         all_lat.max() + lat_span * m)

            # 南北纬/东西经格式化
            from matplotlib.ticker import FuncFormatter
            ax.xaxis.set_major_formatter(FuncFormatter(
                lambda v, _: f"{abs(v):.3f}E" if v >= 0 else f"{abs(v):.3f}W"))
            ax.yaxis.set_major_formatter(FuncFormatter(
                lambda v, _: f"{abs(v):.3f}N" if v >= 0 else f"{abs(v):.3f}S"))
            ax.set_xlabel("Longitude", fontsize=10)
            ax.set_ylabel("Latitude", fontsize=10)
            ax.set_title(f"Survey Tracks ({len(self._survey_lines)} lines, {total_points} pts)  |  Hover for name",
                         fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.2, linestyle='--')
            ax.set_aspect('equal')
            ax.ticklabel_format(useOffset=False, style='plain')

            self.lbl_track_info.setText(
                f"Lon: [{all_lon.min():.6f}, {all_lon.max():.6f}]  "
                f"Lat: [{all_lat.min():.6f}, {all_lat.max():.6f}]  "
                f"Files: {len(self._survey_lines)}  Points: {total_points}"
            )
        else:
            ax.text(0.5, 0.5, "No survey data loaded.\nUse left panel to import TXT.",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=14, color="gray")
            self.lbl_track_info.setText("No data")

        self.map_figure.tight_layout()
        self.map_canvas.draw()
        # draw之后再设一次formatter (防被覆盖)
        try:
            from matplotlib.ticker import FuncFormatter
            ax.xaxis.set_major_formatter(FuncFormatter(
                lambda v, _: f"{abs(v):.3f}E" if v >= 0 else f"{abs(v):.3f}W"))
            ax.yaxis.set_major_formatter(FuncFormatter(
                lambda v, _: f"{abs(v):.3f}N" if v >= 0 else f"{abs(v):.3f}S"))
            self.map_canvas.draw_idle()
        except Exception:
            pass

    # ==================== 地图点击 ====================

    def _on_map_click(self, event):
        """双击航迹点 → 弹出剖面; 单击仅用于导航"""
        import time
        now = time.time()
        is_double = (now - self._last_click_time) < 0.4
        self._last_click_time = now

        if not is_double:
            return  # 单击忽略，不弹窗

        try:
            self._handle_map_click(event)
        except Exception as e:
            import traceback
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Click Error",
                f"Failed to process click:\n{traceback.format_exc()[-500:]}")

    def _handle_map_click(self, event):
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        if not self._survey_lines:
            return

        click_lon = event.xdata
        click_lat = event.ydata

        # 在所有测线中找最近的 GPS 点
        best_sl = None
        best_idx = -1
        best_dist = float('inf')

        for sl in self._survey_lines:
            if len(sl.longitude) < 2:
                continue
            lon = np.asarray(sl.longitude, float)
            lat = np.asarray(sl.latitude, float)
            d2 = (lon - click_lon)**2 + (lat - click_lat)**2
            i_min = np.argmin(d2)
            if d2[i_min] < best_dist:
                best_dist = d2[i_min]
                best_sl = sl
                best_idx = i_min

        if best_sl is None or best_idx < 0:
            return

        # 计算该点的沿航迹距离
        dist = _distance_along_track(
            np.asarray(best_sl.longitude, float),
            np.asarray(best_sl.latitude, float),
        )
        click_dist = dist[best_idx]

        self.point_clicked.emit(best_sl, click_dist)

    # ==================== Hover 标签 ====================

    def _on_hover(self, event):
        """鼠标悬停时显示测线名称"""
        if event.inaxes is None or not hasattr(self, '_track_lines'):
            self._hover_annot.set_visible(False)
            self.map_canvas.draw_idle()
            return

        # 找最近的测线
        mx, my = event.xdata, event.ydata
        best_name = None
        best_dist = float('inf')

        for line, lon, lat, name, color in self._track_lines:
            d2 = (lon - mx)**2 + (lat - my)**2
            i_min = np.argmin(d2)
            if d2[i_min] < best_dist:
                best_dist = d2[i_min]
                best_name = name

        # 只在足够近时显示 (避免远处也弹标签)
        if best_name and best_dist < 5e-7:  # ~0.0007 度 ≈ 70m
            self._hover_annot.set_text(best_name)
            self._hover_annot.xy = (mx, my)
            self._hover_annot.set_visible(True)
        else:
            self._hover_annot.set_visible(False)

        self.map_canvas.draw_idle()

    # ==================== Tab 2: 训练曲线 ====================

    def set_training_history(self, history: dict):
        self._training_history = history
        self._render_training_curves()

    def _render_training_curves(self):
        self.train_figure.clear()
        if self._training_history is None:
            ax = self.train_figure.add_subplot(111)
            ax.text(0.5, 0.5, "Please train model first",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=14, color="gray")
            self.train_canvas.draw()
            return

        epochs = range(1, len(self._training_history.get("train_losses", [])) + 1)
        if len(epochs) == 0:
            self.train_canvas.draw()
            return

        ax1 = self.train_figure.add_subplot(121)
        ax1.plot(epochs, self._training_history["train_losses"], "#1E88E5",
                 label="Train Loss", linewidth=1.5)
        ax1.plot(epochs, self._training_history["val_losses"], "#E53935",
                 label="Val Loss", linewidth=1.5)
        ax1.set_xlabel("Epoch", fontsize=10)
        ax1.set_ylabel("Loss", fontsize=10)
        ax1.set_title("Loss", fontsize=11, fontweight="bold")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2 = self.train_figure.add_subplot(122)
        ax2.plot(epochs, self._training_history["train_accs"], "#1E88E5",
                 label="Train Acc", linewidth=1.5)
        ax2.plot(epochs, self._training_history["val_accs"], "#E53935",
                 label="Val Acc", linewidth=1.5)
        ax2.set_xlabel("Epoch", fontsize=10)
        ax2.set_ylabel("Accuracy", fontsize=10)
        ax2.set_title("Accuracy", fontsize=11, fontweight="bold")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        self.train_figure.tight_layout()
        self.train_canvas.draw()

    # ==================== Tab 3: 分类结果 ====================

    def set_classification_results(self, coordinates, predictions, probabilities=None):
        self._classification_coords = coordinates
        self._classification_preds = predictions
        self._render_classification_map()

    def _render_classification_map(self):
        self.result_figure.clear()
        ax = self.result_figure.add_subplot(111)

        if self._classification_coords is None or self._classification_preds is None:
            ax.text(0.5, 0.5, "Please run classification first",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=14, color="gray")
            self.result_canvas.draw()
            return

        coords = self._classification_coords
        preds = self._classification_preds
        n = len(preds)

        for cls_id in sorted(SEDIMENT_CLASSES_EN.keys()):
            mask = preds == cls_id
            if mask.any():
                rgb = SEDIMENT_COLORS.get(cls_id, (128, 128, 128))
                color = tuple(c / 255.0 for c in rgb)
                count = mask.sum()
                ax.scatter(
                    coords[mask, 0], coords[mask, 1],
                    c=[color], s=18, alpha=0.75,
                    label=f"{SEDIMENT_CLASSES_EN[cls_id]} (n={count})",
                    edgecolors="none",
                )

        from matplotlib.ticker import FuncFormatter
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{abs(v):.4f}°{'E' if v>=0 else 'W'}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{abs(v):.4f}°{'N' if v>=0 else 'S'}"))
        ax.set_xlabel("Longitude", fontsize=10)
        ax.set_ylabel("Latitude", fontsize=10)
        ax.set_title(f"Seafloor Sediment Classification ({n} points)",
                     fontsize=12, fontweight="bold")
        ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5),
                  fontsize=6, ncol=1, framealpha=0.85)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

        self.result_figure.tight_layout(pad=1.0, rect=[0, 0, 0.82, 1])
        self.result_canvas.draw()

    def _render_classification_barchart(self, ax, coords, preds):
        """SEG 数据: 按文件分类柱状图"""
        from config.settings import SEDIMENT_CLASSES_EN, SEDIMENT_COLORS
        file_ids = coords[:, 0].astype(int)
        unique_files = sorted(set(file_ids))
        n_files = len(unique_files)
        n_classes = len(SEDIMENT_CLASSES_EN)

        x = np.arange(n_files)
        width = 0.7
        bottoms = np.zeros(n_files)

        for cls_id in sorted(SEDIMENT_CLASSES_EN.keys()):
            counts = np.array([
                ((file_ids == fi) & (preds == cls_id)).sum()
                for fi in unique_files
            ])
            rgb = SEDIMENT_COLORS.get(cls_id, (128, 128, 128))
            color = tuple(c / 255.0 for c in rgb)
            ax.bar(x, counts, width, bottom=bottoms, color=color,
                   label=SEDIMENT_CLASSES_EN[cls_id], edgecolor='white', linewidth=0.5)
            bottoms += counts

        # X轴标签: 文件名 (截短)
        seg_names = []
        main_win = self.window()
        if hasattr(main_win, 'seg_panel') and main_win.seg_panel._all_seg_data:
            seg_names = [s.name[:10] for s in main_win.seg_panel._all_seg_data]
        while len(seg_names) < n_files:
            seg_names.append(f"F{len(seg_names)}")

        ax.set_xticks(x)
        ax.set_xticklabels([seg_names[i] if i < len(seg_names) else f"F{i}"
                            for i in unique_files], rotation=45, ha='right', fontsize=7)
        ax.set_ylabel("Trace Count", fontsize=10)
        ax.set_title(f"SEG Classification by File ({len(preds)} traces)",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=7, loc='upper right', ncol=2)

    # ==================== 清空 ====================

    def clear_all(self):
        self._dem_data = None
        self._survey_lines = []
        self._track_lines = []
        self._classification_coords = None
        self._classification_preds = None
        self._training_history = None
        self.lbl_track_info.setText("")

        for fig in [self.map_figure, self.train_figure, self.result_figure]:
            fig.clear()
        self.map_canvas.draw()
        self.train_canvas.draw()
        self.result_canvas.draw()
