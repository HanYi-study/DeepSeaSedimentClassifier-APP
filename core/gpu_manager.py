"""
GPU 管理器
=========
本地 GPU 检测 + 远程 GPU SSH 训练 (paramiko)。

远程训练流程:
  1. SFTP 上传训练脚本 + 数据文件到 /tmp/deepsea_xxx/
  2. SSH 执行: cat data.txt | python3 train.py '{params}'
  3. SFTP 下载 best_model.pt 到本地
  4. rm -rf 清理远程临时目录 (即用即删)
"""

import os
import sys
import json
import uuid
import subprocess
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class GPUInfo:
    """GPU 信息"""
    index: int = 0
    name: str = ""
    memory_total_mb: int = 0
    memory_free_mb: int = 0
    is_local: bool = True


class GPUManager:
    """GPU 管理器 —— 本地检测 + 远程 SSH 训练"""

    def __init__(self):
        self.local_gpus: List[GPUInfo] = []
        self.remote_config = {"host": "", "user": "", "password": "", "port": 22}

    # ==================== 本地检测 ====================

    def detect_local(self) -> List[GPUInfo]:
        self.local_gpus = []
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            if result.returncode != 0:
                return []
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    self.local_gpus.append(GPUInfo(
                        index=int(parts[0]), name=parts[1],
                        memory_total_mb=int(parts[2]),
                        memory_free_mb=int(parts[3]), is_local=True))
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return self.local_gpus

    # ==================== 远程连接 ====================

    def configure_remote(self, host, user="", password="", port=22):
        self.remote_config = {"host": host, "user": user, "password": password, "port": port}

    def _ssh_exec(self, command, timeout=15):
        try:
            import paramiko
        except ImportError:
            return ("", "paramiko not installed")
        cfg = self.remote_config
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy)
        try:
            client.connect(cfg["host"], port=cfg.get("port", 22),
                           username=cfg.get("user") or None,
                           password=cfg.get("password") or None, timeout=timeout)
            stdin, stdout, stderr = client.exec_command(command)
            out = stdout.read().decode(errors="replace")
            err = stderr.read().decode(errors="replace")
            return (out, err)
        finally:
            client.close()

    def detect_remote(self, host=None, user=None, password=None):
        if host: self.remote_config["host"] = host
        if user: self.remote_config["user"] = user
        if password is not None: self.remote_config["password"] = password
        if not self.remote_config["host"]:
            return []

        out, err = self._ssh_exec(
            "nvidia-smi --query-gpu=index,name,memory.total,memory.free "
            "--format=csv,noheader,nounits")
        if not out.strip():
            return []

        gpus = []
        for line in out.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                gpus.append(GPUInfo(
                    index=int(parts[0]), name=parts[1],
                    memory_total_mb=int(parts[2]),
                    memory_free_mb=int(parts[3]), is_local=False))
        return gpus

    # ==================== 远程训练 ====================

    def run_remote_training(
        self, gpu_id=0, data_path="", epochs=30, batch_size=512,
        learning_rate=0.001, hidden_dim=128, num_layers=2, num_heads=4,
        output_callback=None,
    ) -> bool:
        """
        远程GPU训练:
          1. SFTP 上传脚本+数据到 /tmp/deepsea_xxx/
          2. SSH 执行 cat data.txt | python3 train.py
          3. SFTP 下载 best_model.pt
          4. rm -rf 清理
        """
        cfg = self.remote_config
        if not cfg.get("host"):
            if output_callback: output_callback("ERROR: 未配置远程服务器")
            return False
        if not data_path or not os.path.exists(data_path):
            if output_callback: output_callback(f"ERROR: 数据文件不存在")
            return False

        try: import paramiko
        except ImportError:
            if output_callback: output_callback("ERROR: paramiko 未安装")
            return False

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        script_src = os.path.join(project_root, "core", "remote_train_script.py")
        if not os.path.exists(script_src):
            if output_callback: output_callback("ERROR: remote_train_script.py 不存在")
            return False

        try:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy)
            client.connect(cfg["host"], port=cfg.get("port", 22),
                           username=cfg.get("user") or None,
                           password=cfg.get("password") or None, timeout=15)
            if output_callback: output_callback(f"SSH 已连接: {cfg['host']}")

            # 创建临时目录, SFTP上传脚本+数据
            remote_dir = f"/tmp/deepsea_{uuid.uuid4().hex[:8]}"
            client.exec_command(f"mkdir -p {remote_dir}")
            sftp = client.open_sftp()
            sftp.put(script_src, f"{remote_dir}/train.py")
            sftp.put(data_path, f"{remote_dir}/data.txt")
            sftp.close()
            if output_callback:
                output_callback(f"已上传 → {remote_dir}")

            # 构建参数
            params = json.dumps({
                "epochs": epochs, "batch_size": batch_size, "lr": learning_rate,
                "hidden_dim": hidden_dim, "num_layers": num_layers,
                "num_heads": num_heads, "gpu_id": gpu_id,
                "output": f"{remote_dir}/best_model.pt",
            })

            # 执行训练
            python_path = json.loads(params).get("python_path", "python3") if isinstance(params, str) else params.get("python_path", "python3")
            cmd = f"cd {remote_dir} && cat data.txt | {python_path} train.py '{params}'"
            if output_callback: output_callback(f"训练启动 (GPU {gpu_id})...")
            stdin, stdout, stderr = client.exec_command(cmd)

            training_ok = False
            for line in stdout:
                line = line.strip()
                if line and output_callback: output_callback(line)
                if line.startswith("DONE|"):
                    training_ok = True

            err = stderr.read().decode(errors="replace")
            if err and output_callback: output_callback(f"ERR: {err[:200]}")

            if not training_ok:
                if output_callback: output_callback("训练未完成 (未收到DONE信号)")
                sftp = client.open_sftp()
                sftp.close()
                client.exec_command(f"rm -rf {remote_dir}")
                client.close()
                return False

            # 下载模型
            local_model = os.path.join(project_root, "best_model_remote.pt")
            sftp = client.open_sftp()
            try:
                sftp.get(f"{remote_dir}/best_model.pt", local_model)
                if output_callback: output_callback("模型已下载")
            except Exception as e:
                if output_callback: output_callback(f"模型下载失败: {e}")

            # 清理
            sftp.close()
            client.exec_command(f"rm -rf {remote_dir}")
            if output_callback: output_callback(f"已清理: {remote_dir}")
            client.close()
            return True

        except Exception as e:
            if output_callback: output_callback(f"ERROR: {e}")
            return False


gpu_manager = GPUManager()
