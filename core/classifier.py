"""
沉积物分类器模块
==============
封装 MSC-Transformer 模型的训练、评估、推理流程。
提供 sklearn 风格的 fit/predict 接口，方便与 GUI 集成。
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Callable, Dict, List, Tuple

from models.msc_transformer import MSCTransformer
from config.settings import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_EPOCHS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_DROPOUT,
    DEFAULT_NUM_HEADS,
    DEFAULT_NUM_LAYERS,
    DEFAULT_TRAIN_SPLIT,
    DEFAULT_RANDOM_SEED,
    SEDIMENT_CLASSES,
)
from utils.logger import logger

# ==================== 设备检测 ====================
_LOCAL_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"本地计算设备: {_LOCAL_DEVICE}")


def get_device(device_str: str = "auto") -> torch.device:
    """获取计算设备. 'auto'=自动, 'cpu'=CPU, 'cuda:0'=指定GPU"""
    if device_str == "auto":
        return _LOCAL_DEVICE
    if device_str == "cpu" or device_str.startswith("cpu"):
        return torch.device("cpu")
    if device_str.startswith("cuda") or device_str.startswith("gpu"):
        idx = 0
        if ":" in device_str:
            idx = int(device_str.split(":")[-1])
        if torch.cuda.is_available():
            return torch.device(f"cuda:{idx}")
        logger.warning(f"CUDA 不可用, 回退到 CPU")
        return torch.device("cpu")
    return torch.device("cpu")


class SedimentClassifier:
    """
    沉积物分类器

    使用方法:
      classifier = SedimentClassifier(input_dim=7, device='cpu')
      classifier.fit(features, labels)
      predictions = classifier.predict(features)
    """

    def __init__(
        self,
        input_dim: int = 7,
        num_classes: int = 5,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        num_heads: int = DEFAULT_NUM_HEADS,
        num_layers: int = DEFAULT_NUM_LAYERS,
        dropout: float = DEFAULT_DROPOUT,
        device: str = "auto",
    ):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = get_device(device) if isinstance(device, str) else device
        logger.info(f"模型设备: {self.device}")

        self.model: Optional[MSCTransformer] = None
        self._build_model()

        # 训练状态
        self.is_trained = False
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.train_accs: List[float] = []
        self.val_accs: List[float] = []

    def _build_model(self):
        """构建/重建模型"""
        self.model = MSCTransformer(
            input_dim=self.input_dim,
            num_classes=self.num_classes,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)
        logger.info(
            f"模型已构建: MSC-Transformer "
            f"(input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, "
            f"num_layers={self.num_layers}, num_classes={self.num_classes})"
        )

    def update_params(
        self,
        hidden_dim: int = None,
        num_heads: int = None,
        num_layers: int = None,
        dropout: float = None,
    ):
        """
        更新模型超参数并重建模型。

        用户可在 GUI 中调整这些参数后重新训练。
        """
        if hidden_dim is not None:
            self.hidden_dim = hidden_dim
        if num_heads is not None:
            self.num_heads = num_heads
        if num_layers is not None:
            self.num_layers = num_layers
        if dropout is not None:
            self.dropout = dropout
        self._build_model()

    # ==================== 训练 ====================

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        train_split: float = DEFAULT_TRAIN_SPLIT,
        random_seed: int = DEFAULT_RANDOM_SEED,
        progress_callback: Optional[Callable] = None,
        stop_event=None,  # threading.Event, 每批次检查
    ) -> Dict:
        """
        训练模型。

        Parameters
        ----------
        features : np.ndarray (N, F)
            特征矩阵
        labels : np.ndarray (N,)
            标签 (0-based 整数)
        learning_rate : float
            学习率
        epochs : int
            训练轮数
        batch_size : int
            批次大小
        train_split : float
            训练集比例
        random_seed : int
            随机种子
        progress_callback : callable(epoch, train_loss, val_loss, train_acc, val_acc, eta_seconds)
            每 epoch 回调，eta_seconds 为预估剩余秒数
        stop_event : threading.Event
            每 batch 检查, set() 即中断训练

        Returns
        -------
        dict: 训练结果汇总
        """
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        # 划分训练/验证集
        n = len(features)
        indices = np.random.permutation(n)
        split = int(n * train_split)
        train_idx = indices[:split]
        val_idx = indices[split:]

        X_train = torch.FloatTensor(features[train_idx])
        y_train = torch.LongTensor(labels[train_idx])
        X_val = torch.FloatTensor(features[val_idx]) if len(val_idx) > 0 else X_train[:0]
        y_val = torch.LongTensor(labels[val_idx]) if len(val_idx) > 0 else y_train[:0]

        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            TensorDataset(X_val, y_val),
            batch_size=batch_size,
            shuffle=False,
        ) if len(val_idx) > 0 else None

        # 优化器 & 损失函数
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=15
        )
        criterion = nn.CrossEntropyLoss()

        # 重置记录
        self.train_losses.clear()
        self.val_losses.clear()
        self.train_accs.clear()
        self.val_accs.clear()

        best_val_loss = float("inf")
        best_state = None

        logger.info(f"开始训练: epochs={epochs}, lr={learning_rate}, batch_size={batch_size}")
        logger.info(f"训练集: {len(train_idx)} 样本, 验证集: {len(val_idx)} 样本")

        self.model.train()

        t_start = time.time()

        for epoch in range(epochs):
            # 检查停止请求
            # 每epoch开头也检查一次
            if stop_event and stop_event.is_set():
                logger.info(f"训练在 epoch {epoch + 1} 前被中断")
                return {"train_losses": self.train_losses,
                        "val_losses": self.val_losses, "stopped": True}

            # ---- 训练阶段 ----
            epoch_train_loss = 0.0
            epoch_train_correct = 0
            epoch_train_total = 0

            for batch_x, batch_y in train_loader:
                # 每批次检查停止信号
                if stop_event and stop_event.is_set():
                    logger.info(f"训练在 epoch {epoch + 1} 被用户中断")
                    self.is_trained = False
                    return {"train_losses": self.train_losses, "val_losses": self.val_losses,
                            "stopped": True}

                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_train_loss += loss.item() * batch_x.size(0)
                preds = torch.argmax(logits, dim=1)
                epoch_train_correct += (preds == batch_y).sum().item()
                epoch_train_total += batch_x.size(0)

            avg_train_loss = epoch_train_loss / epoch_train_total
            train_acc = epoch_train_correct / epoch_train_total

            # ---- 验证阶段 ----
            if val_loader:
                self.model.eval()
                epoch_val_loss = 0.0
                epoch_val_correct = 0
                epoch_val_total = 0

                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device)
                        logits = self.model(batch_x)
                        loss = criterion(logits, batch_y)

                        epoch_val_loss += loss.item() * batch_x.size(0)
                        preds = torch.argmax(logits, dim=1)
                        epoch_val_correct += (preds == batch_y).sum().item()
                        epoch_val_total += batch_x.size(0)

                avg_val_loss = epoch_val_loss / epoch_val_total
                val_acc = epoch_val_correct / epoch_val_total
                self.model.train()
            else:
                avg_val_loss = avg_train_loss
                val_acc = train_acc

            # 学习率调度
            scheduler.step(avg_val_loss)

            # 记录
            self.train_losses.append(avg_train_loss)
            self.val_losses.append(avg_val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)

            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            # 进度回调
            if progress_callback:
                elapsed = time.time() - t_start
                eta = (elapsed / (epoch + 1)) * (epochs - epoch - 1) if epoch > 0 else 0
                progress_callback(epoch + 1, avg_train_loss, avg_val_loss, train_acc, val_acc, eta)

            # 日志 (每 20 epoch 或最后)
            if (epoch + 1) % 20 == 0 or epoch == 0 or epoch == epochs - 1:
                logger.info(
                    f"Epoch {epoch + 1:4d}/{epochs} | "
                    f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.2%} | "
                    f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.2%}"
                )

        # 恢复最佳模型
        if best_state:
            self.model.load_state_dict(best_state)
            logger.info(f"已恢复最佳模型 (val_loss={best_val_loss:.4f})")

        self.is_trained = True
        logger.info("训练完成!")

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accs": self.train_accs,
            "val_accs": self.val_accs,
            "best_val_loss": best_val_loss,
        }

    # ==================== 推理 ====================

    def predict(self, features: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """
        预测沉积物类别。

        Returns
        -------
        np.ndarray (N,) : 类别标签 (1-based, 与论文一致)
        """
        if not self.is_trained:
            logger.warning("模型尚未训练，返回默认预测")
            return np.ones(len(features), dtype=np.int32)

        self.model.eval()
        X = torch.FloatTensor(features)
        dataset = TensorDataset(X)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_preds = []
        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                logits = self.model(batch_x)
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu().numpy())

        preds = np.concatenate(all_preds)
        return preds + 1  # 转为 1-based 与论文一致

    def predict_proba(self, features: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """预测各类别概率"""
        if not self.is_trained:
            return np.zeros((len(features), self.num_classes))

        self.model.eval()
        X = torch.FloatTensor(features)
        dataset = TensorDataset(X)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_probs = []
        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                logits = self.model(batch_x)
                probs = F.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())

        return np.concatenate(all_probs)

    # ==================== 模型持久化 ====================

    def save(self, file_path: str):
        """保存模型到文件"""
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "input_dim": self.input_dim,
                "num_classes": self.num_classes,
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
                "is_trained": self.is_trained,
            },
            file_path,
        )
        logger.info(f"模型已保存: {file_path}")

    def load(self, file_path: str):
        """从文件加载模型"""
        checkpoint = torch.load(file_path, map_location=self.device, weights_only=False)

        self.input_dim = checkpoint["input_dim"]
        self.num_classes = checkpoint["num_classes"]
        self.hidden_dim = checkpoint["hidden_dim"]
        self.num_heads = checkpoint["num_heads"]
        self.num_layers = checkpoint["num_layers"]
        self.dropout = checkpoint["dropout"]

        self._build_model()
        self.model.load_state_dict(checkpoint["model_state"])
        self.is_trained = checkpoint["is_trained"]
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])

        logger.info(f"模型已加载: {file_path}")


# ==================== 辅助: 自动/伪标签生成 ====================

def generate_pseudo_labels(
    features: np.ndarray,
    n_clusters: int = 5,
    random_seed: int = 42,
) -> np.ndarray:
    """
    当没有真实标签时，使用 K-Means 聚类生成伪标签用于初始训练。

    Parameters
    ----------
    features : np.ndarray
    n_clusters : int
        聚类数 (应等于沉积物类别数)
    random_seed : int

    Returns
    -------
    np.ndarray : 聚类标签 (0-based)
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    logger.info(f"K-Means 伪标签生成完成: {n_clusters} 类")
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        logger.info(f"  类 {u + 1}: {c} 样本")

    return labels


# 需要在此处导入（避免循环导入，放文件末尾）
import torch.nn.functional as F
