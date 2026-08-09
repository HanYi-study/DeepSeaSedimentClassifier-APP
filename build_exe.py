#!/usr/bin/env python
"""
PyInstaller 打包脚本
==================
将 DeepSeaSedimentClassifier 打包为独立 Windows .exe。

对方无需安装 Python / PyTorch / PyQt5 / 任何依赖，
双击 exe 即可运行。

用法:
    python build_exe.py

输出:
    dist/DeepSeaSedimentClassifier.exe  (约 2-3 GB)
"""

import os, sys, subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))

def build():
    cmd = [
        r"C:\Python314\python.exe", "-m", "PyInstaller",
        "--name", "DeepSeaSedimentClassifier",
        "--distpath", os.path.join(ROOT, "dist"),
        "--workpath", os.path.join(ROOT, "build_temp"),
        "--specpath", ROOT,
        "--noconsole",
        "--onefile",
        "--clean",
        "--windowed",
        "--icon", os.path.join(ROOT, "icon.ico"),

        # Qt 插件
        "--add-data",
        f"{os.path.join(os.path.dirname(os.__file__), '..', 'Lib', 'site-packages', 'PyQt5', 'Qt5', 'plugins', 'platforms')}{os.pathsep}platforms",

        # 项目数据
        "--add-data", f"icon.ico{os.pathsep}.",

        "--add-data", f"config{os.pathsep}config",

        # torch 必须在 PyQt5 前导入 (DLL 冲突修复)
        "--runtime-hook", os.path.join(ROOT, "runtime_hook.py"),
        "--hidden-import", "torch",
        # 隐藏导入
        "--hidden-import", "sklearn.utils._weight_vector",
        "--hidden-import", "sklearn.cluster._kmeans",
        "--hidden-import", "matplotlib.backends.backend_qt5agg",
        "--hidden-import", "PyQt5.QtCore",
        "--hidden-import", "PyQt5.QtGui",
        "--hidden-import", "PyQt5.QtWidgets",
        "--hidden-import", "rasterio._shim",
        "--hidden-import", "scipy.special._special_ufuncs",

        # 收集 PyTorch 所有 DLL
        "--collect-all", "torch",

        # 收集项目模块
        "--collect-submodules", "config",
        "--collect-submodules", "core",
        "--collect-submodules", "models",
        "--collect-submodules", "ui",
        "--collect-submodules", "utils",

        # 入口
        os.path.join(ROOT, "main.py"),
    ]

    print("=" * 60)
    print("  打包 DeepSeaSedimentClassifier")
    print("=" * 60)
    print(f"项目: {ROOT}")
    print(f"输出: {os.path.join(ROOT, 'dist')}")
    print()
    print("首次打包约 5-10 分钟，请耐心等待...")
    print()

    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode == 0:
        exe = os.path.join(ROOT, "dist", "DeepSeaSedimentClassifier.exe")
        size_mb = os.path.getsize(exe) / (1024 * 1024)
        print()
        print("=" * 60)
        print(f"  打包成功!")
        print(f"  文件: {exe}")
        print(f"  大小: {size_mb:.0f} MB")
        print("=" * 60)
        print()
        print("将 DeepSeaSedimentClassifier.exe 复制到任意 Windows 电脑，")
        print("双击即可运行，无需安装任何环境。")
    else:
        print("\n打包失败，检查上方错误。")
        sys.exit(1)

if __name__ == "__main__":
    build()
