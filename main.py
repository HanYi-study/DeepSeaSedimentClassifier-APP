#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DeepSeaSedimentClassifier - 深海底质分类系统
============================================
基于 MSC-Transformer (JMSE 2023) 的深海底质自动分类软件。

入口文件 - 直接运行即可启动 GUI:
    python main.py

依赖安装:
    pip install -r requirements.txt

功能:
    1. 数据导入: DEM 底图 (tif) + 测线 TXT (GPS + 反射强度) + SEPY 剖面
    2. 可视化: 测线地图、SEPY 剖面、训练曲线、分类结果
    3. 分类器: MSC-Transformer 深度学习模型
    4. 人机交互: 学习率、Epoch、Batch Size 等参数调整
    5. 成果输出: GPS 标记的沉积物分类 TXT

参考论文:
    Wang et al., "Research on Seabed Sediment Classification Based on
    the MSC-Transformer and Sub-Bottom Profiler", JMSE, 2023.
    https://doi.org/10.3390/jmse11051074
"""

import sys
import os

# 确保项目根目录在 Python 路径中
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def check_dependencies():
    """检查关键依赖是否安装"""
    missing = []
    optional_missing = []

    # *** 重要: torch 必须在 PyQt5 之前导入！ ***
    # 否则 Windows 下 PyQt5 的 Qt DLL 会与 PyTorch 的 DLL 冲突，
    # 导致 c10.dll 初始化失败 (WinError 1114)。
    try:
        import numpy
    except ImportError:
        missing.append("numpy (pip install numpy)")

    try:
        import torch
    except ImportError:
        missing.append("torch (pip install torch)")

    try:
        import matplotlib
    except ImportError:
        missing.append("matplotlib (pip install matplotlib)")

    try:
        import PyQt5
    except ImportError:
        missing.append("PyQt5 (pip install PyQt5)")

    try:
        import sklearn
    except ImportError:
        missing.append("scikit-learn (pip install scikit-learn)")

    try:
        import rasterio
    except ImportError:
        optional_missing.append("rasterio (pip install rasterio) - DEM tif 支持")

    try:
        import segyio
    except ImportError:
        optional_missing.append("segyio (pip install segyio) - SEG-Y 格式支持")

    if missing:
        print("=" * 60)
        print("缺少必要依赖:")
        for m in missing:
            print(f"  [MISSING] {m}")
        print("=" * 60)
        return False

    if optional_missing:
        print("[提示] 以下可选依赖未安装 (不影响核心功能):")
        for m in optional_missing:
            print(f"  - {m}")
        print()

    return True


def main():
    """主入口"""
    print("=" * 60)
    print("  DeepSeaSedimentClassifier v1.0.0")
    print("  深海底质分类系统")
    print("  基于 MSC-Transformer (JMSE 2023)")
    print("=" * 60)
    print()

    # 检查依赖
    if not check_dependencies():
        print("\n请安装缺失的依赖后重试:")
        print("  pip install -r requirements.txt")
        sys.exit(1)

    # 启动 GUI
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import Qt

    # ==== 修复 Windows Qt 插件路径问题 ====
    import PyQt5
    _qt_plugin_path = os.path.join(os.path.dirname(PyQt5.__file__), "Qt5", "plugins")
    os.environ["QT_PLUGIN_PATH"] = _qt_plugin_path

    from ui.main_window import MainWindow

    # 高 DPI 支持
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.addLibraryPath(_qt_plugin_path)
    app.setApplicationName("DeepSeaSedimentClassifier")
    app.setOrganizationName("DeepSea Lab")

    # 设置软件图标
    from PyQt5.QtGui import QIcon
    icon_path = os.path.join(PROJECT_ROOT, "icon.ico")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    # 设置全局样式
    app.setStyleSheet("""
        QMainWindow {
            background-color: #f5f5f5;
        }
        QGroupBox {
            font-weight: bold;
            border: 1px solid #ccc;
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
        }
        QPushButton {
            padding: 4px 8px;
            border: 1px solid #999;
            border-radius: 3px;
            background-color: #e8e8e8;
        }
        QPushButton:hover {
            background-color: #d0d0d0;
        }
        QPushButton:pressed {
            background-color: #c0c0c0;
        }
    """)

    # 创建并显示主窗口
    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
