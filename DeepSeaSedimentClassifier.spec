# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import collect_all

datas = [('C:\\Python314\\Lib\\..\\Lib\\site-packages\\PyQt5\\Qt5\\plugins\\platforms', 'platforms'), ('icon.ico', '.'), ('config', 'config')]
binaries = []
hiddenimports = ['torch', 'sklearn.utils._weight_vector', 'sklearn.cluster._kmeans', 'matplotlib.backends.backend_qt5agg', 'PyQt5.QtCore', 'PyQt5.QtGui', 'PyQt5.QtWidgets', 'rasterio._shim', 'scipy.special._special_ufuncs']
hiddenimports += collect_submodules('config')
hiddenimports += collect_submodules('core')
hiddenimports += collect_submodules('models')
hiddenimports += collect_submodules('ui')
hiddenimports += collect_submodules('utils')
tmp_ret = collect_all('torch')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['C:\\userapp\\Projects\\DeepSeaSedimentClassifier\\main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['C:\\userapp\\Projects\\DeepSeaSedimentClassifier\\runtime_hook.py'],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='DeepSeaSedimentClassifier',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['C:\\userapp\\Projects\\DeepSeaSedimentClassifier\\icon.ico'],
)
