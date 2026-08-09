"""
环境检查与自动配置
=================
训练前强制检查 Python 环境，缺少的依赖自动安装。

检查项:
  - Python 版本
  - 核心包: numpy, torch, sklearn, matplotlib, PyQt5
  - 可选包: rasterio, segyio, paramiko
  - GPU: 本地 CUDA / 远程 SSH
"""

import sys
import subprocess
import importlib
import os
from dataclasses import dataclass, field
from typing import List, Tuple

# Windows 下抑制 CMD 弹窗
_CREATE_FLAGS = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def _run_silent(cmd, **kwargs):
    """静默运行子进程 (无 CMD 弹窗)"""
    if "creationflags" not in kwargs:
        kwargs["creationflags"] = _CREATE_FLAGS
    return subprocess.run(cmd, **kwargs)


@dataclass
class EnvStatus:
    """环境检查结果"""
    ok: bool = True
    checks: List[Tuple[str, bool, str]] = field(default_factory=list)
    # (name, passed, message)


class EnvironmentChecker:
    """训练前环境检查器"""

    REQUIRED = {
        "numpy": "numpy",
        "torch": "torch",
        "sklearn": "scikit-learn",
        "matplotlib": "matplotlib",
    }
    OPTIONAL = {
        "rasterio": "rasterio",
        "segyio": "segyio",
        "paramiko": "paramiko",
    }

    def __init__(self):
        self._env_status: EnvStatus = None

    def check_required_only(self) -> EnvStatus:
        """仅检查核心依赖 (不检查 GPU 和可选包)"""
        status = EnvStatus()

        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        ok = sys.version_info >= (3, 9)
        status.checks.append(("Python 版本", ok,
                              f"Python {py_ver}" if ok else f"Python {py_ver} (需要 >=3.9)"))

        for name, pkg in self.REQUIRED.items():
            passed, msg = self._check_package(name, pkg)
            status.checks.append((pkg, passed, msg))
            if not passed:
                status.ok = False

        self._env_status = status
        return status

    def check_all(self) -> EnvStatus:
        """运行全部检查 (含可选包, 不含 GPU)"""
        status = self.check_required_only()

        # 可选依赖 (仅提示, 不阻断)
        for name, pkg in self.OPTIONAL.items():
            passed, msg = self._check_package(name, pkg)
            status.checks.append((f"{pkg} (可选)", passed, msg))

        self._env_status = status
        return status

    def check_gpu_local(self) -> Tuple[bool, str]:
        """检查本地 CUDA"""
        return self._check_gpu()

    def check_gpu_remote(self, host: str, gpu_id: int = 0) -> Tuple[bool, str]:
        """通过 paramiko 检查远程 GPU"""
        try:
            from core.gpu_manager import gpu_manager
            gpus = gpu_manager.detect_remote()
            if gpus:
                g = gpus[min(gpu_id, len(gpus)-1)]
                return (True, f"远程 GPU: {g.name} ({g.memory_total_mb}MB)")
            return (False, "远程 GPU 不可用")
        except Exception as e:
            return (False, f"远程检查失败: {e}")

    def _check_package(self, import_name, pkg_name):
        """检查单个包是否可导入"""
        try:
            mod = importlib.import_module(import_name)
            ver = getattr(mod, "__version__", "?")
            return (True, f"v{ver}")
        except ImportError:
            return (False, "未安装")

    def _check_gpu(self):
        """检查 CUDA 是否可用"""
        try:
            import torch
            if torch.cuda.is_available():
                n = torch.cuda.device_count()
                name = torch.cuda.get_device_name(0)
                return (True, f"{n} GPU: {name}")
            return (False, "CUDA 不可用 (将使用CPU)")
        except Exception:
            return (False, "torch 未安装")

    def install_missing(self, status: EnvStatus = None) -> str:
        """自动安装缺失的依赖"""
        if status is None:
            status = self._env_status or self.check_all()

        missing = []
        for name, passed, msg in status.checks:
            if not passed and name in self.REQUIRED.values():
                missing.append(name)

        if not missing:
            return "所有依赖已就绪"

        # 用 pip 安装
        cmd = [sys.executable, "-m", "pip", "install"] + missing
        try:
            result = _run_silent(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                return f"安装成功: {', '.join(missing)}"
            return f"安装失败:\n{result.stderr[-300:]}"
        except Exception as e:
            return f"安装出错: {e}"

    def is_ready(self) -> bool:
        """环境是否已就绪"""
        if self._env_status is None:
            self.check_all()
        return self._env_status is not None and self._env_status.ok


# 全局单例
env_checker = EnvironmentChecker()
