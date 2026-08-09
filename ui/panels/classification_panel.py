"""
沉积物分类面板
=============
基于 MSC-Transformer (JMSE 2023) 的分类参数设置与训练控制。

人机交互参数:
  - 学习率 (Learning Rate)
  - 训练轮数 (Epochs)
  - 批次大小 (Batch Size)
  - 隐藏维度 (Hidden Dim)
  - Dropout
  - 注意力头数 (Num Heads)
  - Transformer 层数 (Num Layers)
  - 训练集比例 (Train Split)
"""

import os
import sys
import subprocess
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QPushButton, QDoubleSpinBox, QSpinBox,
    QProgressBar, QTextEdit, QComboBox, QMessageBox,
    QSlider, QCheckBox, QTabWidget, QLineEdit, QFileDialog, QRadioButton,
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread

from config.settings import (
    DEFAULT_LEARNING_RATE, DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE,
    DEFAULT_HIDDEN_DIM, DEFAULT_DROPOUT, DEFAULT_NUM_HEADS,
    DEFAULT_NUM_LAYERS, DEFAULT_TRAIN_SPLIT,
    SEDIMENT_CLASSES, SEDIMENT_COLORS,
)
from core.classifier import SedimentClassifier, generate_pseudo_labels
from core.data_processor import DataProcessor
from utils.logger import logger

_RUN_FLAGS = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0

def _run_pip(pkg):
    """静默 pip install (无 CMD 弹窗)"""
    return subprocess.run(
        [sys.executable, "-m", "pip", "install", pkg],
        capture_output=True, text=True, timeout=60,
        creationflags=_RUN_FLAGS,
    )


class TrainingThread(QThread):
    """
    训练线程 —— 本地用 QThread, 远程用原生线程避免 Qt 冲突。
    """
    progress = pyqtSignal(int, float, float, float, float, float)
    log_line = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    stopped = pyqtSignal()

    def __init__(self, classifier, features, labels, params,
                 remote_config=None):
        super().__init__()
        self.classifier = classifier
        self.features = features
        self.labels = labels
        self.params = params
        self.remote_config = remote_config
        self._stop_event = None
        self._remote_thread = None  # 远程用原生线程

    def run(self):
        import threading
        self._stop_event = threading.Event()

        if self.remote_config:
            # 远程训练用原生线程, 避免 QThread + paramiko 冲突
            self._remote_thread = threading.Thread(
                target=self._run_remote, daemon=True)
            self._remote_thread.start()
        else:
            self._run_local()

    def _run_local(self):
        try:
            result = self.classifier.fit(
                features=self.features,
                labels=self.labels,
                learning_rate=self.params.get("lr", DEFAULT_LEARNING_RATE),
                epochs=self.params.get("epochs", DEFAULT_EPOCHS),
                batch_size=self.params.get("batch_size", DEFAULT_BATCH_SIZE),
                train_split=self.params.get("train_split", DEFAULT_TRAIN_SPLIT),
                progress_callback=self._on_progress,
                stop_event=self._stop_event,
            )
            if self._stop_event.is_set():
                self.stopped.emit()
            else:
                self.finished.emit(result)
        except Exception as e:
            logger.exception(f"训练错误: {e}")
            self.error.emit(str(e))

    def _run_remote(self):
        """远程 GPU 训练 → 下载模型 → 自动推理"""
        try:
            self._run_remote_safe()
        except Exception as e:
            import traceback
            self.log_line.emit(f"[远程] 异常: {e}")
            self.log_line.emit(traceback.format_exc()[-400:])
            self.error.emit(str(e))

    def _run_remote_safe(self):
        from core.gpu_manager import gpu_manager
        rc = self.remote_config
        gpu_manager.configure_remote(
            rc["host"], rc.get("user", ""), rc.get("password", ""),
            rc.get("port", 22))

        self.log_line.emit("[远程] 正在连接服务器并上传数据+代码...")

        ok = gpu_manager.run_remote_training(
            gpu_id=rc.get("gpu_id", 0),
            data_path=rc.get("data_path", ""),
            epochs=self.params.get("epochs", 30),
            batch_size=self.params.get("batch_size", 512),
            learning_rate=self.params.get("lr", 0.001),
            hidden_dim=self.params.get("hidden_dim", 128),
            num_layers=self.params.get("num_layers", 2),
            num_heads=self.params.get("num_heads", 4),
            output_callback=lambda line: self.log_line.emit(line),
        )

        if not ok:
            self.error.emit("远程训练失败")
            return

        # 下载模型并自动推理
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        local_model = os.path.join(project_root, "best_model_remote.pt")
        if os.path.exists(local_model) and os.path.getsize(local_model) > 1000:
            try:
                self.classifier.load(local_model)
                self.classifier.is_trained = True
                self.log_line.emit("[远程] 模型已载入")
            except Exception as e:
                self.log_line.emit(f"[远程] 模型载入失败: {e}")
                self.error.emit(f"模型载入失败: {e}")
        else:
            self.log_line.emit("[远程] 模型文件无效, 训练可能未完成")
            self.error.emit("远程训练未生成有效模型")

        self.finished.emit({"best_val_loss": 0.0, "remote": True})

        # 清理本地临时数据
        data_path = rc.get("data_path", "")
        if data_path and os.path.exists(data_path):
            try: os.remove(data_path)
            except Exception: pass

    def _on_progress(self, epoch, train_loss, val_loss, train_acc, val_acc, eta):
        self.progress.emit(epoch, train_loss, val_loss, train_acc, val_acc, eta)

    def stop(self):
        if self._stop_event:
            self._stop_event.set()
        # 远程线程标记
        if self.remote_config:
            self.log_line.emit("[远程] 已发送停止信号")


class ClassificationPanel(QWidget):
    """分类控制面板"""

    # 信号
    training_started = pyqtSignal()
    training_finished = pyqtSignal(dict)
    training_error = pyqtSignal(str)
    predictions_ready = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)  # coords, preds, probs

    def __init__(self, data_processor: DataProcessor, parent=None):
        super().__init__(parent)
        self.data_processor = data_processor
        self.data_loader = None  # 由 MainWindow 注入
        self.classifier: SedimentClassifier = None
        self.train_thread: TrainingThread = None
        self.features_normalized = None
        self.labels = None
        self._best_model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "best_model.pt")

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # ========== 模式选择 ==========
        grp_mode = QGroupBox("模型权重")
        mode_layout = QVBoxLayout(grp_mode)

        mode_row = QHBoxLayout()
        self.radio_retrain = QRadioButton("重新训练")
        self.radio_retrain.setChecked(True)
        self.radio_retrain.toggled.connect(self._on_mode_changed)
        mode_row.addWidget(self.radio_retrain)

        self.radio_load = QRadioButton("使用已有权重")
        self.radio_load.toggled.connect(self._on_mode_changed)
        mode_row.addWidget(self.radio_load)
        mode_row.addStretch()
        mode_layout.addLayout(mode_row)

        # 导入权重行 (默认隐藏)
        self.grp_load = QWidget()
        load_row = QHBoxLayout(self.grp_load)
        load_row.setContentsMargins(0, 0, 0, 0)
        self.edit_weight_path = QLineEdit()
        self.edit_weight_path.setPlaceholderText("选择 .pt 权重文件...")
        self.edit_weight_path.setReadOnly(True)
        load_row.addWidget(self.edit_weight_path)
        btn_browse = QPushButton("浏览")
        btn_browse.clicked.connect(self._load_weights)
        load_row.addWidget(btn_browse)
        self.btn_predict = QPushButton("开始分类预测")
        self.btn_predict.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; padding: 6px; font-weight: bold; }")
        self.btn_predict.clicked.connect(self._run_inference)
        load_row.addWidget(self.btn_predict)
        self.grp_load.setVisible(False)
        mode_layout.addWidget(self.grp_load)

        layout.addWidget(grp_mode)

        # ========== GPU 配置 ==========
        grp_gpu = QGroupBox("GPU 配置")
        gpu_layout = QVBoxLayout(grp_gpu)

        self.lbl_gpu_info = QLabel("选择计算设备")
        self.lbl_gpu_info.setStyleSheet("color: gray; font-size: 10px;")
        gpu_layout.addWidget(self.lbl_gpu_info)

        gpu_row = QHBoxLayout()
        gpu_row.addWidget(QLabel("设备:"))
        self.cmb_gpu = QComboBox()
        self.cmb_gpu.setMinimumWidth(150)
        gpu_row.addWidget(self.cmb_gpu)

        self.btn_detect_gpu = QPushButton("检测GPU")
        self.btn_detect_gpu.clicked.connect(self._detect_gpu)
        gpu_row.addWidget(self.btn_detect_gpu)
        gpu_layout.addLayout(gpu_row)

        # 远程配置 (默认隐藏)
        self.grp_remote = QWidget()
        remote_layout = QVBoxLayout(self.grp_remote)
        remote_layout.setContentsMargins(0, 0, 0, 0)

        host_row = QHBoxLayout()
        host_row.addWidget(QLabel("Host:"))
        self.edit_remote_host = QLineEdit()
        self.edit_remote_host.setPlaceholderText("user@192.168.1.100")
        host_row.addWidget(self.edit_remote_host)
        remote_layout.addLayout(host_row)

        pwd_row = QHBoxLayout()
        pwd_row.addWidget(QLabel("密码:"))
        self.edit_remote_pwd = QLineEdit()
        self.edit_remote_pwd.setPlaceholderText("输入SSH密码")
        self.edit_remote_pwd.setEchoMode(QLineEdit.Password)
        pwd_row.addWidget(self.edit_remote_pwd)
        self.btn_connect_remote = QPushButton("连接")
        self.btn_connect_remote.clicked.connect(self._connect_remote)
        pwd_row.addWidget(self.btn_connect_remote)
        remote_layout.addLayout(pwd_row)

        py_row = QHBoxLayout()
        py_row.addWidget(QLabel("Python:"))
        self.edit_remote_py = QLineEdit()
        self.edit_remote_py.setPlaceholderText("默认python3 (或conda环境路径)")
        py_row.addWidget(self.edit_remote_py)
        remote_layout.addLayout(py_row)

        self.grp_remote.setVisible(False)
        gpu_layout.addWidget(self.grp_remote)

        self.chk_use_remote = QCheckBox("使用远程 GPU")
        self.chk_use_remote.toggled.connect(lambda v: self.grp_remote.setVisible(v))
        gpu_layout.addWidget(self.chk_use_remote)

        layout.addWidget(grp_gpu)

        # ========== 环境检查 (GPU 配置之后) ==========
        grp_env = QGroupBox("环境检查")
        env_layout = QVBoxLayout(grp_env)

        env_row = QHBoxLayout()
        self.btn_check_env = QPushButton("检查环境")
        self.btn_check_env.setStyleSheet(
            "QPushButton { background-color: #FF9800; color: white; padding: 6px; font-weight: bold; }")
        self.btn_check_env.clicked.connect(self._check_environment)
        env_row.addWidget(self.btn_check_env)

        self.lbl_env_status = QLabel("请选择GPU设备后检查环境")
        self.lbl_env_status.setStyleSheet("color: #E53935; font-weight: bold; font-size: 11px;")
        env_row.addWidget(self.lbl_env_status)

        self.btn_install_env = QPushButton("安装缺失依赖")
        self.btn_install_env.clicked.connect(self._install_dependencies)
        self.btn_install_env.setVisible(False)
        env_row.addWidget(self.btn_install_env)
        env_row.addStretch()
        env_layout.addLayout(env_row)

        self.txt_env_log = QTextEdit()
        self.txt_env_log.setReadOnly(True)
        self.txt_env_log.setMaximumHeight(80)
        self.txt_env_log.setPlaceholderText("环境检查结果将在此显示...")
        env_layout.addWidget(self.txt_env_log)

        layout.addWidget(grp_env)

        # ========== 数据源选择 ==========
        grp_source = QGroupBox("训练数据源")
        source_layout = QVBoxLayout(grp_source)
        self.radio_txt = QRadioButton("TXT 测线数据")
        self.radio_txt.setChecked(True)
        self.radio_seg = QRadioButton("SEG 浅剖数据")
        source_layout.addWidget(self.radio_txt)
        source_layout.addWidget(self.radio_seg)
        layout.addWidget(grp_source)

        # ========== 标签模式 ==========
        grp_labels = QGroupBox("训练标签")
        label_layout = QVBoxLayout(grp_labels)

        self.chk_pseudo_labels = QCheckBox("使用 K-Means 聚类生成伪标签")
        self.chk_pseudo_labels.setChecked(True)
        label_layout.addWidget(self.chk_pseudo_labels)

        # 沉积物类别勾选
        label_layout.addWidget(QLabel("预测类别 (勾选=包含):"))
        self._sediment_checks = {}
        default_types = [
            "Calcareous bio-silt", "Calcareous bio-clay silt",
            "Silty sand", "Medium sand", "Gravel sand",
            "Sandy silt", "Clay", "Coarse sand", "Mud", "Gravel",
        ]
        for i, name in enumerate(default_types):
            cb = QCheckBox(name)
            cb.setChecked(i < 5)  # 默认前5个
            cb.toggled.connect(self._on_class_selection)
            label_layout.addWidget(cb)
            self._sediment_checks[name] = cb

        self.lbl_label_info = QLabel("已选 5 类")
        self.lbl_label_info.setStyleSheet("color: gray; font-size: 11px;")
        label_layout.addWidget(self.lbl_label_info)

        layout.addWidget(grp_labels)

        # ========== 模型超参数 ==========
        grp_params = QGroupBox("MSC-Transformer 超参数 (参考 JMSE 2023)")
        params_layout = QVBoxLayout(grp_params)

        # 学习率
        row_lr = QHBoxLayout()
        row_lr.addWidget(QLabel("学习率 (Learning Rate):"))
        self.spin_lr = QDoubleSpinBox()
        self.spin_lr.setRange(0.00001, 0.1)
        self.spin_lr.setValue(DEFAULT_LEARNING_RATE)
        self.spin_lr.setDecimals(5)
        self.spin_lr.setSingleStep(0.0001)
        self.spin_lr.setToolTip("AdamW 优化器的初始学习率。论文推荐: 0.001")
        row_lr.addWidget(self.spin_lr)
        params_layout.addLayout(row_lr)

        # 训练轮数
        row_epochs = QHBoxLayout()
        row_epochs.addWidget(QLabel("训练轮数 (Epochs):"))
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(10, 5000)
        self.spin_epochs.setValue(DEFAULT_EPOCHS)
        self.spin_epochs.setSingleStep(10)
        self.spin_epochs.setToolTip("完整遍历训练集的次数。论文推荐: 200")
        row_epochs.addWidget(self.spin_epochs)
        params_layout.addLayout(row_epochs)

        # 批次大小
        row_batch = QHBoxLayout()
        row_batch.addWidget(QLabel("批次大小 (Batch Size):"))
        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(4, 1024)
        self.spin_batch.setValue(DEFAULT_BATCH_SIZE)
        self.spin_batch.setSingleStep(8)
        self.spin_batch.setToolTip("每次迭代处理的样本数。论文推荐: 64")
        row_batch.addWidget(self.spin_batch)
        params_layout.addLayout(row_batch)

        # 隐藏维度
        row_hidden = QHBoxLayout()
        row_hidden.addWidget(QLabel("隐藏维度 (Hidden Dim):"))
        self.spin_hidden = QSpinBox()
        self.spin_hidden.setRange(32, 2048)
        self.spin_hidden.setValue(DEFAULT_HIDDEN_DIM)
        self.spin_hidden.setSingleStep(64)
        self.spin_hidden.setToolTip("Transformer 内部表示维度。论文: 512")
        row_hidden.addWidget(self.spin_hidden)
        params_layout.addLayout(row_hidden)

        # Dropout
        row_dropout = QHBoxLayout()
        row_dropout.addWidget(QLabel("Dropout:"))
        self.spin_dropout = QDoubleSpinBox()
        self.spin_dropout.setRange(0.0, 0.9)
        self.spin_dropout.setValue(DEFAULT_DROPOUT)
        self.spin_dropout.setDecimals(2)
        self.spin_dropout.setSingleStep(0.05)
        self.spin_dropout.setToolTip("防止过拟合。论文: 0.1")
        row_dropout.addWidget(self.spin_dropout)
        params_layout.addLayout(row_dropout)

        # 注意力头数 & 层数
        row_arch = QHBoxLayout()
        row_arch.addWidget(QLabel("注意力头数:"))
        self.spin_heads = QSpinBox()
        self.spin_heads.setRange(1, 16)
        self.spin_heads.setValue(DEFAULT_NUM_HEADS)
        self.spin_heads.setToolTip("多头自注意力头数。论文: 8")
        row_arch.addWidget(self.spin_heads)

        row_arch.addWidget(QLabel("层数:"))
        self.spin_layers = QSpinBox()
        self.spin_layers.setRange(1, 12)
        self.spin_layers.setValue(DEFAULT_NUM_LAYERS)
        self.spin_layers.setToolTip("Transformer 编码器层数。论文: 4")
        row_arch.addWidget(self.spin_layers)
        params_layout.addLayout(row_arch)

        # 训练集比例
        row_split = QHBoxLayout()
        row_split.addWidget(QLabel("训练集比例:"))
        self.spin_split = QDoubleSpinBox()
        self.spin_split.setRange(0.5, 0.95)
        self.spin_split.setValue(DEFAULT_TRAIN_SPLIT)
        self.spin_split.setDecimals(2)
        self.spin_split.setSingleStep(0.05)
        row_split.addWidget(self.spin_split)
        row_split.addStretch()
        params_layout.addLayout(row_split)

        # 一键应用参数
        self.btn_apply_params = QPushButton("[>] 应用超参数 (重建模型)")
        self.btn_apply_params.clicked.connect(self._apply_params)
        self.btn_apply_params.setEnabled(False)  # 环境检查通过后才启用
        params_layout.addWidget(self.btn_apply_params)

        layout.addWidget(grp_params)

        # ========== 训练控制 ==========
        grp_train = QGroupBox("训练控制")
        train_layout = QVBoxLayout(grp_train)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setFormat("就绪")
        train_layout.addWidget(self.progress_bar)

        # 当前状态
        self.lbl_status = QLabel("等待训练...")
        self.lbl_status.setStyleSheet("color: gray;")
        train_layout.addWidget(self.lbl_status)

        # 按钮: 开始训练 / 停止 / 重新训练 (=停止并立即重新开始)
        btn_layout = QHBoxLayout()
        self.btn_train = QPushButton("[>>] 开始训练")
        self.btn_train.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; padding: 6px; font-weight: bold; }"
            "QPushButton:disabled { background-color: #ccc; }"
        )
        self.btn_train.clicked.connect(self._start_training)
        self.btn_train.setEnabled(False)
        self.btn_train.setToolTip("使用当前参数开始训练新模型")

        self.btn_stop = QPushButton("[STOP] 停止")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_training)
        self.btn_stop.setToolTip("中断当前训练")

        self.btn_retrain = QPushButton("重新训练")
        self.btn_retrain.clicked.connect(self._reset_training)
        self.btn_retrain.setToolTip("停止当前训练, 清除日志, 用新参数从头开始")
        btn_layout.addWidget(self.btn_train)
        btn_layout.addWidget(self.btn_stop)
        btn_layout.addWidget(self.btn_retrain)
        train_layout.addLayout(btn_layout)

        # 训练完成后: 开始预测按钮 (默认隐藏)
        self.btn_predict_after = QPushButton("开始预测 (使用最佳模型)")
        self.btn_predict_after.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; padding: 8px; font-weight: bold; }")
        self.btn_predict_after.clicked.connect(self._run_inference)
        self.btn_predict_after.setVisible(False)
        train_layout.addWidget(self.btn_predict_after)

        # 最佳模型状态
        self.lbl_best_model = QLabel("")
        self.lbl_best_model.setStyleSheet("color: gray; font-size: 10px;")
        train_layout.addWidget(self.lbl_best_model)

        layout.addWidget(grp_train)

        # ========== 训练日志 ==========
        grp_log = QGroupBox("训练日志")
        log_layout = QVBoxLayout(grp_log)
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setMinimumHeight(120)
        self.txt_log.setMaximumHeight(300)
        self.txt_log.setPlaceholderText("训练信息将在此显示...")
        log_layout.addWidget(self.txt_log)
        layout.addWidget(grp_log)

    # ==================== 类别选择 ====================

    def _on_class_selection(self):
        checked = [n for n, cb in self._sediment_checks.items() if cb.isChecked()]
        self.lbl_label_info.setText(f"已选 {len(checked)} 类: {', '.join(checked[:4])}...")

    def _get_selected_classes(self):
        """返回勾选的类别字典 {1: name, 2: name, ...}"""
        checked = [n for n, cb in self._sediment_checks.items() if cb.isChecked()]
        return {i+1: name for i, name in enumerate(checked)}

    # ==================== 模式切换 ====================

    def _on_mode_changed(self):
        """重新训练 vs 使用已有权重"""
        if not hasattr(self, 'btn_train'):
            return  # 初始化未完成, 跳过
        if self.radio_retrain.isChecked():
            self.grp_load.setVisible(False)
            self.btn_train.setVisible(True)
            self.btn_stop.setVisible(True)
            self.btn_reset.setVisible(True)
            if hasattr(self, 'btn_predict_after'):
                self.btn_predict_after.setVisible(False)
        else:
            self.grp_load.setVisible(True)
            self.btn_train.setVisible(False)
            self.btn_stop.setVisible(False)
            self.btn_reset.setVisible(False)
            if hasattr(self, 'btn_predict_after'):
                self.btn_predict_after.setVisible(False)
            self.btn_predict_after.setVisible(False)

    # ==================== 环境检查 ====================

    def _check_environment(self):
        """GPU 配置后检查环境 (上下文感知)"""
        from core.env_checker import env_checker
        self.btn_check_env.setEnabled(False)
        self.btn_check_env.setText("检查中...")
        self.lbl_env_status.setText("正在检查...")
        self.lbl_env_status.setStyleSheet("color: #FB8C00; font-size: 11px;")
        self.txt_env_log.clear()

        from PyQt5.QtCore import QTimer
        def do_check():
            status = env_checker.check_required_only()
            lines = []
            all_ok = status.ok

            for name, passed, msg in status.checks:
                icon = "[OK]" if passed else "[FAIL]"
                lines.append(f"  {icon} {name}: {msg}")

            # 根据 GPU 选择做额外检查
            sel = self.cmb_gpu.currentText()
            if self.chk_use_remote.isChecked():
                # 远程 GPU: 需要 paramiko
                try:
                    import paramiko
                    lines.append(f"  [OK] paramiko: v{paramiko.__version__}")
                except ImportError:
                    lines.append("  [FAIL] paramiko: 未安装 (远程GPU必需)")
                    all_ok = False

                if "远程" in sel:
                    gpu_ok, gpu_msg = env_checker.check_gpu_remote(
                        self.edit_remote_host.text().strip())
                    icon = "[OK]" if gpu_ok else "[FAIL]"
                    lines.append(f"  {icon} 远程GPU: {gpu_msg}")
                    device_type = "远程GPU"
                else:
                    lines.append("  [WARN] 请点击[连接]检测远程GPU")
                    device_type = "远程GPU(未检测)"
            elif "GPU" in sel and "无GPU" not in sel:
                gpu_ok, gpu_msg = env_checker.check_gpu_local()
                icon = "[OK]" if gpu_ok else "[WARN]"
                lines.append(f"  {icon} 本地CUDA: {gpu_msg}")
                device_type = "本地GPU" if gpu_ok else "本地GPU(CUDA不可用)"
            else:
                lines.append("  [OK] 计算设备: CPU")
                device_type = "CPU"

            self.txt_env_log.setText("\n".join(lines))

            if all_ok:
                self.lbl_env_status.setText(
                    f"环境已就绪 ({device_type})，请设置参数并开始训练")
                self.lbl_env_status.setStyleSheet(
                    "color: #43A047; font-weight: bold; font-size: 11px;")
                self.btn_train.setEnabled(True)
                self.btn_apply_params.setEnabled(True)
                self.btn_install_env.setVisible(False)
            else:
                self.lbl_env_status.setText("核心依赖缺失，请安装后重试")
                self.lbl_env_status.setStyleSheet(
                    "color: #E53935; font-weight: bold; font-size: 11px;")
                self.btn_train.setEnabled(False)
                self.btn_install_env.setVisible(True)

            self.btn_check_env.setText("重新检查")
            self.btn_check_env.setEnabled(True)

        QTimer.singleShot(50, do_check)

    def _install_dependencies(self):
        """自动安装缺失依赖 (含远程GPU所需的paramiko)"""
        from core.env_checker import env_checker
        self.btn_install_env.setEnabled(False)
        self.btn_install_env.setText("安装中...")
        self.txt_env_log.append("\n[自动安装] 正在安装缺失依赖...")

        from PyQt5.QtCore import QTimer
        def do_install():
            # 安装核心依赖
            env_checker.install_missing()
            # 如果选了远程GPU, 也装 paramiko
            if self.chk_use_remote.isChecked():
                try:
                    import paramiko
                except ImportError:
                    self.txt_env_log.append("[自动安装] 安装 paramiko...")
                    _run_pip("paramiko")

            self.btn_install_env.setText("安装完成")
            self.btn_install_env.setVisible(False)
            self._check_environment()

        QTimer.singleShot(100, do_install)

    # ==================== GPU 管理 ====================

    def _detect_gpu(self):
        """检测本地 + 远程 GPU"""
        from core.gpu_manager import gpu_manager
        self.cmb_gpu.clear()

        # 检测本地
        local = gpu_manager.detect_local()
        if local:
            for g in local:
                self.cmb_gpu.addItem(f"本地 GPU[{g.index}]: {g.name} ({g.memory_free_mb}MB free)")
            self.lbl_gpu_info.setText(f"本地 {len(local)} 块 GPU 可用")
        else:
            self.cmb_gpu.addItem("本地: 无GPU (CPU)")
            self.lbl_gpu_info.setText("本地无 GPU, 建议使用远程 GPU")

        # 检测远程
        if self.chk_use_remote.isChecked() and self.edit_remote_host.text():
            self._connect_remote()

    def _connect_remote(self):
        """连接远程 GPU 服务器"""
        host_text = self.edit_remote_host.text().strip()
        if not host_text:
            QMessageBox.warning(self, "配置错误", "请输入远程服务器地址")
            return

        # 确保 paramiko 已安装
        try:
            import paramiko
        except ImportError:
            reply = QMessageBox.question(
                self, "缺少依赖",
                "远程 GPU 连接需要 paramiko 库，是否自动安装？",
                QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.lbl_gpu_info.setText("正在安装 paramiko...")
                self.btn_connect_remote.setEnabled(False)
                result = _run_pip("paramiko")
                if result.returncode == 0:
                    self.lbl_gpu_info.setText("paramiko 安装成功，请重新点击连接")
                    self.txt_log.append("[GPU] paramiko 安装成功")
                else:
                    self.lbl_gpu_info.setText(f"paramiko 安装失败")
                    self.txt_log.append(f"[GPU] 安装失败: {result.stderr[-200:]}")
                self.btn_connect_remote.setEnabled(True)
            return

        if "@" in host_text:
            user, host = host_text.split("@", 1)
        else:
            user, host = "", host_text

        password = self.edit_remote_pwd.text()

        from core.gpu_manager import gpu_manager
        gpu_manager.configure_remote(host, user, password)
        self.lbl_gpu_info.setText(f"正在连接 {host}...")
        self.lbl_gpu_info.setStyleSheet("color: #FB8C00; font-size: 10px;")
        self.txt_log.append(f"[GPU] 正在连接 {host}:22 ...")
        self.btn_connect_remote.setEnabled(False)

        from PyQt5.QtCore import QTimer
        def do_connect():
            try:
                remote_gpus = gpu_manager.detect_remote()
                if remote_gpus:
                    for i in range(self.cmb_gpu.count() - 1, -1, -1):
                        if "远程" in self.cmb_gpu.itemText(i):
                            self.cmb_gpu.removeItem(i)
                    for g in remote_gpus:
                        self.cmb_gpu.addItem(
                            f"远程 GPU[{g.index}]: {g.name} ({g.memory_free_mb}MB free)")
                    self.cmb_gpu.setCurrentIndex(self.cmb_gpu.count() - 1)
                    self.lbl_gpu_info.setText(
                        f"远程 {host}: {len(remote_gpus)} GPU | {remote_gpus[0].name} | "
                        f"显存: {remote_gpus[0].memory_total_mb}MB")
                    self.lbl_gpu_info.setStyleSheet(
                        "color: #43A047; font-weight: bold; font-size: 10px;")
                    self.txt_log.append(
                        f"[GPU] 连接成功! {len(remote_gpus)} GPU, "
                        f"型号: {remote_gpus[0].name}")
                else:
                    # 详细诊断
                    out, err = gpu_manager._ssh_exec("hostname && nvidia-smi -L 2>&1")
                    if err and ("auth" in err.lower() or "password" in err.lower() or "permission" in err.lower()):
                        detail = f"认证失败: {err[:80]}"
                    elif err and ("refused" in err.lower()):
                        detail = f"连接被拒绝: {err[:80]}"
                    elif err and ("timeout" in err.lower() or "timed out" in err.lower()):
                        detail = f"连接超时: {err[:80]}"
                    elif err and ("resolve" in err.lower() or "name" in err.lower()):
                        detail = f"无法解析主机名: {host}"
                    elif err:
                        detail = f"SSH错误: {err[:100]}"
                    elif out.strip():
                        detail = f"SSH已通但无nvidia-smi: {out[:60]}"
                    else:
                        detail = "未知原因, 请手动SSH测试连接"
                    self.lbl_gpu_info.setText(f"连接失败: {detail}")
                    self.lbl_gpu_info.setStyleSheet("color: #E53935; font-size: 10px;")
                    self.txt_log.append(f"[GPU] 失败详情: {detail}")
                    if err:
                        self.txt_log.append(f"[GPU] 原始错误: {err[:200]}")
            except Exception as e:
                import traceback
                self.lbl_gpu_info.setText(f"异常: {str(e)[:80]}")
                self.lbl_gpu_info.setStyleSheet("color: #E53935; font-size: 10px;")
                self.txt_log.append(f"[GPU] 异常详情: {traceback.format_exc()[-400:]}")
            self.btn_connect_remote.setEnabled(True)

        QTimer.singleShot(100, do_connect)

    # ==================== 槽函数 ====================

    def _ensure_features(self) -> bool:
        """确保特征已提取。TXT模式自动提取, SEG模式提前提取。"""
        if self.radio_seg.isChecked():
            # SEG 模式: 检查是否已提取特征
            if self.features_normalized is not None:
                return True
            QMessageBox.warning(self, "SEG模式",
                "请先选择 SEG 数据源, 然后点击「开始训练」自动提取特征。")
            return False

        # TXT 模式
        if self.data_processor.processed is not None:
            return True
        if self.data_loader is None or not self.data_loader.survey_lines:
            QMessageBox.warning(self, "无数据", "请先在左侧面板导入测线 TXT 数据。")
            return False
        self.data_processor.extract_features(
            self.data_loader.survey_lines, self.data_loader.dem)
        self.features_normalized = self.data_processor.get_normalized_features()
        self.txt_log.append("[自动] 特征已提取，可直接训练")
        return True

    def _apply_params(self):
        """应用超参数，重建模型"""
        # SEG 模式: 不需要 _ensure_features, 直接构建模型
        if self.radio_seg.isChecked():
            if self.features_normalized is not None:
                input_dim = self.features_normalized.shape[1]
            else:
                input_dim = 13  # SEG 特征数
        else:
            if not self._ensure_features():
                return
            input_dim = self.data_processor.processed.num_features
        num_classes = len(SEDIMENT_CLASSES)

        # 根据 GPU 选择确定设备
        sel = self.cmb_gpu.currentText()
        if "远程" in sel:
            device = "cpu"  # 远程训练不在此处处理, 先设CPU
            self.txt_log.append("[设备] 远程GPU将通过SSH执行训练")
        elif "GPU" in sel and "无GPU" not in sel:
            # 本地 GPU
            gpu_idx = 0
            if "GPU[" in sel:
                try:
                    gpu_idx = int(sel.split("[")[1].split("]")[0])
                except ValueError:
                    gpu_idx = 0
            device = f"cuda:{gpu_idx}"
            self.txt_log.append(f"[设备] 使用本地 {device}")
        else:
            device = "cpu"
            self.txt_log.append("[设备] 使用 CPU 训练")

        self.classifier = SedimentClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=self.spin_hidden.value(),
            num_heads=self.spin_heads.value(),
            num_layers=self.spin_layers.value(),
            dropout=self.spin_dropout.value(),
            device=device,
        )

        self.txt_log.append(
            f"[参数已应用] hidden_dim={self.spin_hidden.value()}, "
            f"num_heads={self.spin_heads.value()}, "
            f"num_layers={self.spin_layers.value()}, "
            f"dropout={self.spin_dropout.value():.2f}"
        )
        self.btn_train.setEnabled(True)
        self.btn_train.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; padding: 8px; font-weight: bold; }"
            "QPushButton:disabled { background-color: #ccc; }"
        )
        self.lbl_status.setText("参数已就绪，可以开始训练")
        self.lbl_status.setStyleSheet("color: #2196F3; font-weight: bold;")

    def _start_training(self):
        """启动训练 (支持 TXT 或 SEG 数据源)"""
        # SEG 数据源
        if self.radio_seg.isChecked():
            self._start_seg_training()
            return

        # TXT 数据源
        if not self._ensure_features():
            return
        self.features_normalized = self.data_processor.get_normalized_features()
        self._start_training_common()

    def _start_seg_training(self):
        """使用 SEG 浅剖数据训练"""
        try:
            main_win = self.window()
            if not hasattr(main_win, 'seg_panel'):
                QMessageBox.warning(self, "无数据", "请先在 SEG 面板中加载浅剖数据")
                return
            seg_data = main_win.seg_panel._all_seg_data
            if not seg_data:
                QMessageBox.warning(self, "无数据", "SEG 面板中没有数据，请先浏览文件夹加载")
                return
        except Exception as e:
            QMessageBox.warning(self, "错误", f"无法获取SEG数据: {e}")
            return

        from core.seg_processor import extract_seg_features

        self.txt_log.append(f"[SEG] 正在从 {len(seg_data)} 个文件中提取特征...")
        self.lbl_status.setText("SEG 特征提取中...")
        # 强制立即处理事件, 避免 Qt 认为无响应
        from PyQt5.QtWidgets import QApplication
        QApplication.processEvents()

        try:
            feats, coords = extract_seg_features(seg_data, max_traces_per_file=3000)
            self.features_normalized = feats
            from core.data_processor import ProcessedData
            self.data_processor.processed = ProcessedData(
                features=feats, coordinates=coords,
                feature_names=[f"SEG_F{i}" for i in range(feats.shape[1])],
                num_samples=feats.shape[0], num_features=feats.shape[1],
            )
            self.txt_log.append(
                f"[SEG] 特征提取完成: {feats.shape[0]} 条道, {feats.shape[1]} 个特征")
            self.labels = generate_pseudo_labels(feats, n_clusters=len(self._get_selected_classes()))
            self._start_training_common()
        except Exception as e:
            import traceback
            self.txt_log.append(f"[SEG] 错误: {e}\n{traceback.format_exc()[-400:]}")
            self.lbl_status.setText(f"SEG失败: {e}")
            self.btn_train.setEnabled(True)

    def _start_training_common(self):
        """通用训练启动 (TXT和SEG共用)"""
        try:
            self._start_training_common_unsafe()
        except Exception as e:
            import traceback
            self.txt_log.append(f"[崩溃] {e}\n{traceback.format_exc()[-600:]}")
            self.lbl_status.setText(f"训练启动失败: {e}")
            self.btn_train.setEnabled(True)
            self.btn_stop.setEnabled(False)

    def _start_training_common_unsafe(self):
        # 清理旧训练线程 (防止 Qt 崩溃)
        if self.train_thread and self.train_thread.isRunning():
            self.train_thread.stop()
            self.train_thread.quit()
            self.train_thread.wait(3000)
        self.train_thread = None

        # 生成/获取标签
        if self.chk_pseudo_labels.isChecked():
            self.labels = generate_pseudo_labels(
                self.features_normalized,
                n_clusters=len(SEDIMENT_CLASSES),
            )
        elif self.labels is None:
            QMessageBox.warning(self, "无标签", "请导入标注数据或启用伪标签模式。")
            return

        # 确保分类器已构建
        if self.classifier is None:
            self._apply_params()

        # 配置训练参数
        params = {
            "lr": self.spin_lr.value(),
            "epochs": self.spin_epochs.value(),
            "batch_size": self.spin_batch.value(),
            "train_split": self.spin_split.value(),
            "hidden_dim": self.spin_hidden.value(),
            "num_layers": self.spin_layers.value(),
            "num_heads": self.spin_heads.value(),
        }

        # 判断是否远程训练
        sel = self.cmb_gpu.currentText()
        remote_config = None
        if self.chk_use_remote.isChecked() and "远程" in sel:
            gpu_id = 0
            if "GPU[" in sel:
                try:
                    gpu_id = int(sel.split("[")[1].split("]")[0])
                except ValueError:
                    gpu_id = 0
            # 保存内存中的数据到临时文件以便上传
            import tempfile
            data_path = os.path.join(tempfile.gettempdir(), "deepsea_remote_data.txt")
            if self.radio_seg.isChecked():
                # SEG 模式: 保存特征矩阵 + 标签
                if self.features_normalized is not None:
                    combined = np.column_stack([
                        self.features_normalized,
                        self.labels if self.labels is not None else np.zeros(len(self.features_normalized))
                    ])
                    np.savetxt(data_path, combined, fmt="%.6f", delimiter=",")
                    self.txt_log.append(f"[远程] SEG特征已保存: {data_path} ({combined.shape[0]}行 x {combined.shape[1]}列)")
                else:
                    data_path = ""
                    self.txt_log.append("[远程] 错误: 无SEG特征数据")
            elif self.data_loader and self.data_loader.survey_lines:
                all_lon = np.concatenate([sl.longitude for sl in self.data_loader.survey_lines])
                all_lat = np.concatenate([sl.latitude for sl in self.data_loader.survey_lines])
                all_int = np.concatenate([sl.reflection_intensity for sl in self.data_loader.survey_lines])
                all_seq = np.concatenate([sl.sequence_number for sl in self.data_loader.survey_lines])
                with open(data_path, "w") as f:
                    for i in range(len(all_lon)):
                        f.write(f"{all_lon[i]:.8f},{all_lat[i]:.8f},{all_int[i]:.6f},{all_seq[i]:.0f}\n")
                self.txt_log.append(f"[远程] 数据已保存: {data_path} ({len(all_lon)} 行)")
            else:
                data_path = ""
                self.txt_log.append("[远程] 错误: 无数据可上传")

            remote_config = {
                "host": self.edit_remote_host.text().strip().split("@")[-1]
                        if "@" in self.edit_remote_host.text() else self.edit_remote_host.text(),
                "user": self.edit_remote_host.text().split("@")[0]
                        if "@" in self.edit_remote_host.text() else "",
                "password": self.edit_remote_pwd.text(),
                "gpu_id": gpu_id,
                "data_path": data_path,
                "python_path": self.edit_remote_py.text().strip() or "python3",
            }

        # 创建并启动训练线程
        self.train_thread = TrainingThread(
            self.classifier, self.features_normalized, self.labels, params,
            remote_config=remote_config,
        )
        self.train_thread.progress.connect(self._on_progress)
        self.train_thread.log_line.connect(lambda line: self.txt_log.append(f"  {line}"))
        self.train_thread.finished.connect(self._on_finished)
        self.train_thread.stopped.connect(self._on_stopped)
        self.train_thread.error.connect(self._on_error)

        self.btn_train.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_stop.setStyleSheet(
            "QPushButton { background-color: #E53935; color: white; padding: 6px; }"
        )
        self.progress_bar.setMaximum(params["epochs"])
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Training... %v / %m epochs")
        self.txt_log.clear()
        n_samples = len(self.features_normalized)
        batches_per_epoch = n_samples // params['batch_size']
        self.txt_log.append(
            f"[训练开始] lr={params['lr']:.5f}, epochs={params['epochs']}, "
            f"batch_size={params['batch_size']}, "
            f"样本数={n_samples}, 每epoch约{batches_per_epoch}批次"
        )
        self.txt_log.append(
            f"[提示] 首epoch数据量大, 请耐心等待... "
            f"完成后将显示预估剩余时间 (ETA)"
        )
        self.lbl_status.setText("Training... (first epoch, ETA will show after)")
        self.lbl_status.setStyleSheet("color: #E53935; font-weight: bold;")

        self.training_started.emit()
        self.train_thread.start()

    def _stop_training(self):
        if self.train_thread and self.train_thread.isRunning():
            self.btn_stop.setText("停止中...")
            self.btn_stop.setEnabled(False)
            self.txt_log.append("[训练] 发送停止信号, 等待当前批次完成...")
            self.train_thread.stop()

    def _on_stopped(self):
        self.btn_train.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setText("停止训练")
        self.progress_bar.setFormat("已停止")
        self.lbl_status.setText("训练已停止")
        self.lbl_status.setStyleSheet("color: #FB8C00; font-weight: bold;")
        self.txt_log.append("[训练] 已成功停止")

    def _on_progress(self, epoch, train_loss, val_loss, train_acc, val_acc, eta):
        self.progress_bar.setValue(epoch)
        # 格式化预估剩余时间
        if eta <= 0:
            eta_str = "calculating..."
        elif eta > 3600:
            eta_str = f"{eta/3600:.1f}h"
        elif eta > 60:
            eta_str = f"{eta/60:.1f}min"
        else:
            eta_str = f"{eta:.0f}s"
        self.progress_bar.setFormat(
            f"Epoch {epoch}/{self.spin_epochs.value()} "
            f"| Loss: {train_loss:.4f}/{val_loss:.4f} "
            f"| Acc: {train_acc:.2%}/{val_acc:.2%} "
            f"| ETA: {eta_str}"
        )
        self.lbl_status.setText(
            f"Training... Epoch {epoch}, Loss: {train_loss:.4f}, Acc: {val_acc:.2%}, ETA: {eta_str}"
        )

    def _on_finished(self, result):
        self.btn_train.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet("")
        self.progress_bar.setFormat("Training Done!")
        self.progress_bar.setValue(self.progress_bar.maximum())
        self.lbl_status.setText(
            f"训练完成! 最佳 Val Loss: {result['best_val_loss']:.4f}"
        )
        self.lbl_status.setStyleSheet("color: #43A047; font-weight: bold;")

        # 自动保存最佳模型
        if self.classifier and self.classifier.is_trained:
            self.classifier.save(self._best_model_path)
            self.lbl_best_model.setText(
                f"best_model.pt (loss={result['best_val_loss']:.4f})")
            self.lbl_best_model.setStyleSheet("color: #43A047; font-weight: bold; font-size: 10px;")
            self.txt_log.append(f"\n[模型] 最佳模型已保存: best_model.pt")

        self.txt_log.append(
            f"\n[训练完成] 最佳 Val Loss: {result['best_val_loss']:.4f}"
        )

        # 显示预测按钮
        self.btn_predict_after.setVisible(True)

        # 自动执行推理
        if self.features_normalized is not None and self.classifier.is_trained:
            self._run_inference()

        self.training_finished.emit(result)

    def _reset_training(self):
        """重新训练: 清除状态但保留最佳权重"""
        reply = QMessageBox.question(
            self, "重新训练",
            "将清除当前训练日志和进度，但保留之前的最佳权重文件。\n确认重新开始？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return

        if self.train_thread and self.train_thread.isRunning():
            self.train_thread.stop()
            self.train_thread.wait(2000)

        self.txt_log.clear()
        self.txt_log.append("[重新训练] 已清除日志，保留最佳权重")
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("就绪")
        self.lbl_status.setText("等待训练...")
        self.lbl_status.setStyleSheet("color: gray;")
        self.btn_train.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setText("停止训练")
        self._apply_params()

    def _load_weights(self):
        """导入已训练的权重文件"""
        path, _ = QFileDialog.getOpenFileName(
            self, "导入模型权重", "", "PyTorch 模型 (*.pt *.pth);;所有文件 (*)")

        if path:
            self.edit_weight_path.setText(path)
        else:
            return

        if not self.classifier:
            # 自动构建默认模型
            if not self._ensure_features():
                return
            self._apply_params()

        try:
            self.classifier.load(path)
            self.classifier.is_trained = True
            self._best_model_path = path
            self.lbl_best_model.setText(f"权重: {os.path.basename(path)} (已导入)")
            self.lbl_best_model.setStyleSheet("color: #43A047; font-weight: bold; font-size: 10px;")
            self.txt_log.append(f"[模型] 权重已导入: {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "导入失败", f"无法加载权重:\n{e}")
            self.txt_log.append(f"[错误] 权重导入失败: {e}")

    def _on_error(self, msg):
        self.btn_train.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.txt_log.append(f"[错误] {msg}")
        self.training_error.emit(msg)

    def _run_inference(self):
        """使用最佳模型运行推理"""
        if os.path.exists(self._best_model_path) and not self.classifier.is_trained:
            try:
                self.classifier.load(self._best_model_path)
                self.txt_log.append(f"[推理] 已加载最佳模型: {self._best_model_path}")
            except Exception:
                pass

        if not self.classifier.is_trained:
            self.txt_log.append("[推理] 模型未训练，无法推理")
            return

        probs = self.classifier.predict_proba(self.features_normalized)
        preds = self.classifier.predict(self.features_normalized)

        # 根据数据源选择坐标
        if self.radio_seg.isChecked():
            # SEG 数据: 使用文件索引和道号作为坐标标记
            coords = self.data_processor.processed.coordinates
            # coords 是 (file_idx, trace_idx), 展开为可读格式
            coords = np.column_stack([
                coords[:, 0],  # file index
                coords[:, 1],  # trace index
            ])
        else:
            coords = self.data_processor.processed.coordinates

        selected_classes = self._get_selected_classes()
        self.txt_log.append(f"\n[推理] {len(preds)} 个样本已分类:")
        for cls_id in sorted(selected_classes.keys()):
            cls_name = selected_classes[cls_id]
            count = np.sum(preds == cls_id)
            self.txt_log.append(f"  {cls_name}: {count} ({count/len(preds)*100:.1f}%)")

        # SEG 模式: 额外按文件统计
        if self.radio_seg.isChecked() and hasattr(self.window(), 'seg_panel'):
            seg_data = self.window().seg_panel._all_seg_data
            self.txt_log.append(f"\n[SEG] 按文件分类统计:")
            file_ids = coords[:, 0].astype(int)
            for fi in range(len(seg_data)):
                mask = file_ids == fi
                if mask.any():
                    preds_f = preds[mask]
                    unique, counts = np.unique(preds_f, return_counts=True)
                    parts = [f"{selected_classes.get(u, '?')}:{c}" for u, c in zip(unique, counts)]
                    self.txt_log.append(f"  {seg_data[fi].name}: {', '.join(parts)}")

        self.predictions_ready.emit(coords, preds, probs)

    # ==================== 外部接口 ====================

    def set_features(self, features_normalized: np.ndarray, labels: np.ndarray = None):
        """从外部设置特征和标签"""
        self.features_normalized = features_normalized
        self.labels = labels
        if labels is not None:
            self.chk_pseudo_labels.setChecked(False)
            self.lbl_label_info.setText(f"已加载 {len(labels)} 个真实标签")

    def get_training_history(self):
        """获取训练历史用于可视化"""
        if self.classifier is None:
            return None
        return {
            "train_losses": self.classifier.train_losses,
            "val_losses": self.classifier.val_losses,
            "train_accs": self.classifier.train_accs,
            "val_accs": self.classifier.val_accs,
        }
