"""
MSC-Transformer 模型
==================
基于 JMSE 2023 论文:
  "Research on Seabed Sediment Classification Based on
   the MSC-Transformer and Sub-Bottom Profiler"

架构: TabTransformer 变体
  - 输入嵌入层: input_dim -> hidden_dim
  - Dropout
  - N 层 Transformer Encoder
  - 全连接分类头
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== 位置编码 ====================

class PositionalEncoding(nn.Module):
    """可学习的位置编码"""
    def __init__(self, hidden_dim: int, max_len: int = 256):
        super().__init__()
        self.encoding = nn.Parameter(torch.randn(1, max_len, hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.encoding[:, :seq_len, :]


# ==================== MSC-Transformer ====================

class MSCTransformer(nn.Module):
    """
    MSC-Transformer: Transformer of Marine Sediment Classification

    Parameters
    ----------
    input_dim : int
        输入特征维度
    num_classes : int
        分类类别数 (默认 5)
    hidden_dim : int
        Transformer 隐藏维度 (论文: 512)
    num_heads : int
        多头注意力头数 (论文: 8)
    num_layers : int
        Transformer 编码器层数 (论文: 4)
    dropout : float
        Dropout 比例 (论文: 0.1)
    dim_feedforward : int
        前馈网络维度 (默认 hidden_dim * 4)
    """

    def __init__(
        self,
        input_dim: int = 7,
        num_classes: int = 5,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        dim_feedforward: int = None,
    ):
        if dim_feedforward is None:
            dim_feedforward = hidden_dim * 4
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        # ---- 输入嵌入 ----
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ---- 位置编码 ----
        self.pos_encoding = PositionalEncoding(hidden_dim)

        # ---- Transformer 编码器 ----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # ---- 全局池化 ----
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1),
        )

        # ---- 分类头 ----
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 4, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Parameters
        ----------
        x : torch.Tensor (B, F)
            输入特征, B=batch_size, F=input_dim

        Returns
        -------
        torch.Tensor (B, num_classes)
            分类 logits
        """
        # 如果输入是 2D (B, F)，扩展为序列 (B, 1, F)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, F)

        # 嵌入
        x = self.input_embedding(x)  # (B, 1, H)

        # 位置编码
        x = self.pos_encoding(x)

        # Transformer 编码
        x = self.transformer_encoder(x)  # (B, 1, H)

        # 注意力池化
        attn_weights = self.attention_pool(x)  # (B, 1, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        x = (x * attn_weights).sum(dim=1)  # (B, H)

        # 分类
        logits = self.classifier(x)  # (B, num_classes)
        return logits

    def predict(self, x: torch.Tensor) -> np.ndarray:
        """预测类别标签 (0-based)"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            preds = torch.argmax(logits, dim=1)
        return preds.cpu().numpy()

    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        """预测类别概率"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
        return probs.cpu().numpy()
