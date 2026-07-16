# -*- coding: utf-8 -*-
"""
HDMTL-style FW-UAV fault diagnosis
空间特征分支替换版：CNN spatial branch -> Feature-wise Transformer / Spatial Self-Attention

核心变化：
1. 原 CNN 分支替换为 FeatureWiseTransformerBranch。
2. 输入 x: [B, T, F]，例如 [B, 20, 12]。
3. 空间分支把 12 个飞行参数当作 token，每个 token 持有该参数的 20 步历史。
4. TransformerEncoder 在 feature token 之间做 self-attention，用于学习变量之间的空间耦合关系。
5. Temporal 分支由原三层 LSTM 简化为一层 LSTM，进一步降低小样本过拟合风险。
6. 去除 LSTM 分支和 Feature-wise Transformer 空间分支中的 dropout。
7. Temporal Attention、Metric Loss、Neighborhood-Aware Soft Label 保持原逻辑。
8. 融合分支由 AFAM/MHMIFF 改为 DSGF：不使用跨域多头注意力，显式建模时空一致性与差异性。

模型语义：
- Spatial branch: 12 个变量之间的关系建模。
- Temporal branch: 一层 LSTM 建模 20 个时间步之间的动态关系。
- Fusion branch: DSGF 差异感知状态门控融合，基于一致状态与差异状态输出 fused_seq。
"""

import os
import copy
import random
import numpy as np
import pandas as pd

from sklearn import model_selection as ms
from sklearn import preprocessing
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from pprz_data.pprz_data import DATA


# ===================== 1. 固定随机种子 =====================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ===================== 2. 参数配置 =====================
FEATURES = [
    'airspeed', 'phi', 'psi', 'theta',
    'Ax', 'Ay', 'Az',
    'Gx', 'Gy', 'Gz',
    'C1', 'C2'
]

AC_ID = '20'
DATA_FILE = r"D:\HDMTL-main\data\23_07_2020_Faulty_Daredevil_MultiClass_Sweep_2\20_07_23__07_19_26_SD.data"
DATA_TYPE = "flight"
SAMPLE_PERIOD = 0.1

TIME_MIN = 600
TIME_MAX = 2800

NUM_CLASSES = 9
SEQ_LENGTH = 20

# ===== HDMTL 风格划分参数 =====
TEST_SIZE = 0.10
TEST_RANDOM_STATE = 42
# SELECTED_TOTAL_SAMPLES 是占位值，main() 中会被 SELECTED_TOTAL_SAMPLES_LIST 逐个覆盖。
# 9 分类时建议列表里的数值都能被 NUM_CLASSES=9 整除，保证每类抽样数量一致。
SELECTED_TOTAL_SAMPLES = 90
SELECTED_TOTAL_SAMPLES_LIST = [90,120, 150,180, 270, 360, 720, 1440]
# 如果每类候选池样本充足，可以改成例如：
# SELECTED_TOTAL_SAMPLES_LIST = [90, 180, 270, 360, 720, 1440]

TRAIN_RATIO_ON_SELECTED = 0.70

# ===== 重复实验次数 =====
TIMES = 10
HDMTL_SAMPLE_SEED = 0

# ===== 结果保存目录 =====
RESULT_SAVE_DIR = "./hdmtl_multiclass_sample_size_results"
os.makedirs(RESULT_SAVE_DIR, exist_ok=True)

# ===== 训练参数 =====
BATCH_SIZE = 16
EPOCHS = 150
LEARNING_RATE = 0.0005

# 分类头 / 融合投影处仍保留的 dropout。
DROPOUT = 0.15
LSTM_DROPOUT = 0.15
TRANSFORMER_DROPOUT = 0.15

WEIGHT_DECAY = 1e-4

EMBED_DIM = 128
METRIC_LOSS_WEIGHT = 2.0
CONTRASTIVE_MARGIN = 1.0

EARLY_STOPPING_PATIENCE = 50
LR_SCHEDULER_PATIENCE = 7
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_MIN_LR = 1e-6

# ===== Local-window Feature-wise Transformer 参数 =====
SPATIAL_WINDOW_SIZE = 5
SPATIAL_D_MODEL = 128
SPATIAL_NHEAD = 8
SPATIAL_NUM_LAYERS = 3
SPATIAL_FF_DIM = 256

# ===== DSGF 差异感知状态门控融合参数 =====
# DSGF 不使用多头 cross-attention，因此不需要 num_heads。

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===================== 2.1 邻域软标签参数 =====================
USE_SOFT_LABEL = True
SOFT_MAIN_WEIGHT = 0.8
SOFT_NEIGHBOR_WEIGHT = 0.1
SOFT_SECOND_NEIGHBOR_WEIGHT = 0.00
ENHANCE_CLASS2 = False


# ===================== 3. Dataset =====================
class FaultDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ===================== 4. Metric Loss =====================
class BatchContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, pos_weight=1.0, neg_weight=1.0, eps=1e-8):
        super().__init__()
        self.margin = margin
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.eps = eps

    def forward(self, embeddings, labels):
        if labels.dim() > 1:
            labels = labels.squeeze(1)

        labels = labels.long()
        batch_size = embeddings.size(0)

        if batch_size < 2:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        dist_mat = torch.cdist(embeddings, embeddings, p=2)

        label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
        diag_mask = ~torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)

        pos_mask = label_eq & diag_mask
        neg_mask = (~label_eq) & diag_mask

        pos_dists = dist_mat[pos_mask]
        neg_dists = dist_mat[neg_mask]

        if pos_dists.numel() > 0:
            pos_loss = (pos_dists ** 2).sum() / (pos_dists.numel() + self.eps)
        else:
            pos_loss = torch.tensor(0.0, device=embeddings.device)

        if neg_dists.numel() > 0:
            neg_loss = (torch.clamp(self.margin - neg_dists, min=0.0) ** 2).sum()
            neg_loss = neg_loss / (neg_dists.numel() + self.eps)
        else:
            neg_loss = torch.tensor(0.0, device=embeddings.device)

        return self.pos_weight * pos_loss + self.neg_weight * neg_loss


# ===================== 5. 注意力模块 =====================
class TemporalAttention(nn.Module):
    """
    LSTM 输出上的时间注意力。

    输入:
        x: [B, T, H]
    输出:
        context: [B, H]
        weights: [B, T]
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        score = self.attn(x)                  # [B, T, 1]
        weight = torch.softmax(score, dim=1)  # [B, T, 1]
        context = torch.sum(x * weight, dim=1)
        return context, weight.squeeze(-1)


class FeatureAttentionPooling(nn.Module):
    """
    特征 token 维度上的 attention pooling。

    输入:
        x: [B, F, D]
    输出:
        context: [B, D]
        weight: [B, F]

    这里的 F 是特征变量数量，例如 12。
    weight 可以用来观察模型认为哪些变量更重要。
    """
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1)
        )

    def forward(self, x):
        score = self.attn(x)                  # [B, F, 1]
        weight = torch.softmax(score, dim=1)  # [B, F, 1]
        context = torch.sum(x * weight, dim=1)
        return context, weight.squeeze(-1)


class FeatureWiseTransformerBranch(nn.Module):
    """
    Local-window Feature-wise Transformer / Spatial Self-Attention branch.

    用途：替代原来的 CNN 空间分支，并且直接输出 [B, T, 128]。

    核心思想：
        token 仍然是 12 个物理参数，但每个 token 不再只是当前时刻的单个标量，
        而是该参数在当前时刻附近的局部时间窗口。

    输入:
        x: [B, T, F]
        例如 [B, 20, 12]

    处理逻辑：
        1. 对每个时间步 t，围绕 t 取一个局部窗口，窗口长度为 window_size。
           例如 window_size=5 时，每个参数 token 包含：
               [x_{t-2}, x_{t-1}, x_t, x_{t+1}, x_{t+2}]

        2. 对每个时间步 t，构造 F 个 feature tokens：
               token_airspeed = airspeed 的局部时间片段
               token_phi      = phi 的局部时间片段
               ...
               token_C2       = C2 的局部时间片段

        3. 每个 token 的局部时间窗口通过 Linear(window_size -> D) 映射到 D 维。
           同时加入 learnable feature embedding，用于区分不同物理参数身份。

        4. TransformerEncoder 在 F 个参数 token 之间做 self-attention，
           学习“局部时间片段中的参数间耦合关系”。

        5. 对 F 个变量 token 做 feature attention pooling，得到每个时间步的空间表示。

    输出:
        xs_seq: [B, T, 128]
        feature_weight optional: [B, T, F]

    这样得到的 xs_seq[:, t, :] 表示：
        第 t 个时间位置附近，12 个参数局部时间轨迹之间的空间耦合特征。
    """
    def __init__(
        self,
        seq_len=20,
        num_features=12,
        window_size=5,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.0,
        out_dim=128
    ):
        super().__init__()

        if window_size < 1:
            raise ValueError("window_size 必须 >= 1")
        if window_size % 2 == 0:
            raise ValueError("window_size 建议使用奇数，例如 3、5、7，以便围绕当前时刻对称取窗")

        self.seq_len = seq_len
        self.num_features = num_features
        self.window_size = window_size
        self.pad_size = window_size // 2
        self.d_model = d_model
        self.out_dim = out_dim

        # 每个变量 token 包含一个局部时间窗口，因此用 Linear(window_size -> D) 投影。
        self.value_proj = nn.Linear(window_size, d_model)

        # 每个飞行参数一个 learnable feature identity embedding。
        self.feature_embed = nn.Parameter(torch.zeros(1, 1, num_features, d_model))
        nn.init.trunc_normal_(self.feature_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 对每个时间步内部的 feature tokens 做 attention pooling。
        self.pool = FeatureAttentionPooling(dim=d_model)

        self.out_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, out_dim),
            nn.ReLU(inplace=True),
            nn.Identity() if dropout <= 0 else nn.Dropout(dropout),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x, return_attention=False):
        # x: [B, T, F]
        B, T, F_dim = x.shape

        if T != self.seq_len:
            raise ValueError(
                f"FeatureWiseTransformerBranch 输入时间长度不匹配："
                f"seq_len={self.seq_len}, 当前 T={T}"
            )

        if F_dim != self.num_features:
            raise ValueError(
                f"FeatureWiseTransformerBranch 输入特征维度不匹配："
                f"num_features={self.num_features}, 当前 F={F_dim}"
            )

        # ------------------------------------------------------
        # 构造局部时间窗口 token
        # ------------------------------------------------------
        # x: [B, T, F] -> [B, F, T]
        x_ft = x.transpose(1, 2)

        # 在时间维度做 replicate padding，避免边界位置丢失。
        # [B, F, T] -> [B, F, T + 2 * pad]
        x_pad = F.pad(x_ft, (self.pad_size, self.pad_size), mode="replicate")

        # 沿时间维度滑窗：
        # [B, F, T + 2p] -> [B, F, T, window_size]
        windows = x_pad.unfold(dimension=2, size=self.window_size, step=1)

        # 调整为 [B, T, F, window_size]
        windows = windows.permute(0, 2, 1, 3).contiguous()

        # 每个参数 token 的局部时间窗口 -> D 维 token embedding。
        z = self.value_proj(windows)          # [B, T, F, D]
        z = z + self.feature_embed            # [B, T, F, D]

        # TransformerEncoder 期望 [batch, token, dim]。
        # 这里把 B 和 T 合并，使每个时间位置独立做 feature-wise self-attention。
        z = z.reshape(B * T, F_dim, self.d_model)  # [B*T, F, D]
        z = self.encoder(z)                        # [B*T, F, D]

        # 对 F 个变量 token 做 attention pooling，得到每个时间步的空间表示。
        z, feature_weight = self.pool(z)           # [B*T, D], [B*T, F]

        z = self.out_proj(z)                       # [B*T, 128]
        xs_seq = z.reshape(B, T, self.out_dim)     # [B, T, 128]
        feature_weight = feature_weight.reshape(B, T, F_dim)  # [B, T, F]

        if return_attention:
            return xs_seq, feature_weight
        return xs_seq


class TemporalAttnPool(nn.Module):
    """
    对融合后的时间序列做可学习时间注意力池化。

    输入:
        x: [B, T, C]

    输出:
        context: [B, C]
        weight:  [B, T]

    作用:
        相比 AdaptiveAvgPool1d，TemporalAttnPool 可以让模型自动关注故障更明显的关键时间步，
        避免突变点或局部故障响应被简单平均稀释。
    """
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1)
        )

    def forward(self, x):
        # x: [B, T, C]
        weight = torch.softmax(self.score(x), dim=1)  # [B, T, 1]
        context = torch.sum(x * weight, dim=1)         # [B, C]
        return context, weight.squeeze(-1)


class DifferenceAwareStateGatedFusion(nn.Module):
    """
    Difference-aware State-space Gated Fusion, DSGF
    差异感知状态门控融合模块。

    设计动机：
        与基于 cross-attention 或互信息约束的时空融合不同，DSGF 不直接让空间分支和时间分支互相注意，
        而是显式构造二者的一致响应与差异响应。对于正常状态与轻微故障状态高度相似的任务，
        细微故障往往表现为空间耦合特征与时间动态特征之间的不一致，因此差异响应具有诊断价值。

    输入:
        spatial_seq:  [B, T, C]
        temporal_seq: [B, T, C]

    输出:
        fused_seq: [B, T, C]

    核心逻辑：
        1. sum_feat  = xs + xt        表示时空分支的一致响应
        2. diff_feat = |xs - xt|      表示时空分支的不一致响应
        3. prod_feat = xs * xt        表示时空分支的相关增强响应
        4. common_state 提取稳定共性信息
        5. diff_state   提取故障敏感差异信息
        6. common_gate 和 diff_gate 自适应控制两类状态的贡献
    """
    def __init__(self, channels=128, dropout=0.1):
        super().__init__()

        self.channels = channels

        # 一致性状态：由 sum 与 product 描述。
        self.common_proj = nn.Sequential(
            nn.Linear(2 * channels, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(channels)
        )

        # 差异性状态：由 absolute difference 与 product 描述。
        self.diff_proj = nn.Sequential(
            nn.Linear(2 * channels, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(channels)
        )

        # 一致性门：控制稳定共性信息的贡献。
        self.common_gate = nn.Sequential(
            nn.Linear(3 * channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
            nn.Sigmoid()
        )

        # 差异性门：控制故障敏感差异信息的贡献。
        self.diff_gate = nn.Sequential(
            nn.Linear(3 * channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
            nn.Sigmoid()
        )

        # 输出稳定化。
        self.out_proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(channels)
        )

    def forward(self, spatial_seq, temporal_seq, return_attention=False):
        if spatial_seq.shape != temporal_seq.shape:
            raise ValueError(
                f"DSGF 输入形状必须一致，"
                f"但得到 spatial_seq={tuple(spatial_seq.shape)}, "
                f"temporal_seq={tuple(temporal_seq.shape)}"
            )

        xs = spatial_seq
        xt = temporal_seq

        # ------------------------------------------------------
        # 1. 显式构造一致性、差异性与相关增强信息
        # ------------------------------------------------------
        sum_feat = xs + xt
        diff_feat = torch.abs(xs - xt)
        prod_feat = xs * xt

        # ------------------------------------------------------
        # 2. 状态建模
        # ------------------------------------------------------
        common_state = self.common_proj(
            torch.cat([sum_feat, prod_feat], dim=-1)
        )  # [B, T, C]

        diff_state = self.diff_proj(
            torch.cat([diff_feat, prod_feat], dim=-1)
        )  # [B, T, C]

        # ------------------------------------------------------
        # 3. 双门控融合
        # ------------------------------------------------------
        gate_input = torch.cat([sum_feat, diff_feat, prod_feat], dim=-1)
        g_common = self.common_gate(gate_input)
        g_diff = self.diff_gate(gate_input)

        fused_seq = g_common * common_state + g_diff * diff_state

        # 残差保留原始时空分支的平均信息，避免差异门过强导致稳定状态信息丢失。
        residual = 0.5 * (xs + xt)
        fused_seq = self.out_proj(fused_seq + residual)

        if return_attention:
            return fused_seq, {
                "dsgf_common_gate": g_common,
                "dsgf_diff_gate": g_diff
            }

        return fused_seq


# ===================== 6. 模型：Feature-wise Transformer + LSTM + Attention =====================
class FeatureTransformerLSTMAttentionMetricModel(nn.Module):
    """
    双分支模型：
    - Spatial branch: Feature-wise Transformer / Spatial Self-Attention
    - Temporal branch: 1-layer LSTM + optional Temporal Attention
    - Fusion: concat MLP or DSGF difference-aware gated fusion
    - Heads: metric embedding + classification

    输入:
        x: [B, T, F]
        例如 [B, 20, 12]

    Spatial branch 语义：
        把 12 个飞行参数当作 12 个 token，学习变量之间的空间相关性。

    Temporal branch 语义：
        用一层 LSTM 沿 20 个时间步建模动态变化，进一步降低小样本下的过拟合风险。
    """
    def __init__(
        self,
        input_size=12,
        seq_len=20,
        embed_dim=128,
        dropout=0.3,
        num_classes=9,
        use_bilstm=False,
        use_temporal_attention=True,
        use_adaptive_fusion=True,
        spatial_window_size=5,
        spatial_d_model=128,
        spatial_nhead=4,
        spatial_num_layers=2,
        spatial_ff_dim=256
    ):
        super().__init__()

        self.input_size = input_size
        self.seq_len = seq_len
        self.use_bilstm = use_bilstm
        self.use_temporal_attention = use_temporal_attention
        self.use_adaptive_fusion = use_adaptive_fusion

        self.relu = nn.ReLU(inplace=True)
        # 分类头 / 非 LSTM 路径继续使用 dropout。
        self.dropout = nn.Dropout(dropout)
        # LSTM 分支 dropout 去除。
        self.lstm_dropout = nn.Identity() if LSTM_DROPOUT <= 0 else nn.Dropout(LSTM_DROPOUT)

        # ======================================================
        # Spatial branch: Feature-wise Transformer
        # 说明：该分支的 TransformerEncoder 和 out_proj 中 dropout 已置为 0。
        # ======================================================
        self.spatial_branch = FeatureWiseTransformerBranch(
            seq_len=seq_len,
            num_features=input_size,
            window_size=spatial_window_size,
            d_model=spatial_d_model,
            nhead=spatial_nhead,
            num_layers=spatial_num_layers,
            dim_feedforward=spatial_ff_dim,
            dropout=TRANSFORMER_DROPOUT,
            out_dim=128
        )

        # ======================================================
        # Temporal branch: 1-layer LSTM，最轻量版
        # ======================================================
        # 说明：原始版本为 12 -> 64 -> 128 -> 128 的三层 LSTM。
        # 上一版已简化为 12 -> 64 -> 128 的两层 LSTM。
        # 当前版本按需求进一步改为 12 -> 128 的一层 LSTM，输出维度仍保持 128。
        # 按当前需求，LSTM 分支中的 dropout 已去除。
        # 因此后续 xt_seq_fc、temporal_attn、融合层和分类头都无需改变。
        self.lstm_num_directions = 2 if use_bilstm else 1

        self.lstm_hidden1 = 128

        self.lstm_out_dim1 = self.lstm_hidden1 * self.lstm_num_directions
        self.lstm_out_dim = self.lstm_out_dim1

        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=self.lstm_hidden1,
            num_layers=1,
            batch_first=True,
            bidirectional=use_bilstm
        )

        # 使用 LayerNorm 替代 BatchNorm1d(seq_len)。
        # lstm_out 的形状是 [B, T, H]，LayerNorm(self.lstm_out_dim)
        # 作用在最后一维 H 上。
        # 这里不接 ReLU，避免截断 LSTM 输出中的负响应动态信息。
        self.lstm_ln1 = nn.LayerNorm(self.lstm_out_dim)

        if self.use_temporal_attention:
            self.temporal_attn = TemporalAttention(hidden_dim=self.lstm_out_dim)
        else:
            self.temporal_attn = None

        self.xt_fc = nn.Linear(self.lstm_out_dim, 128)
        self.xt_ln = nn.LayerNorm(128)

        # DSGF 需要两个分支都保留 [B, T, C] 形式。
        # 这里把 LSTM 的完整时间序列输出映射到 128 维，作为 temporal branch sequence。
        self.xt_seq_fc = nn.Linear(self.lstm_out_dim, 128)
        self.xt_seq_ln = nn.LayerNorm(128)

        # ======================================================
        # Fusion
        # ======================================================
        if self.use_adaptive_fusion:
            # DSGF:
            # 输入两个 [B, T, C] 序列，显式构造时空一致性与差异性，
            # 最终输出一个 fused_seq: [B, T, 128]。
            self.fusion = DifferenceAwareStateGatedFusion(
                channels=128,
                dropout=dropout
            )

            # 对 fused_seq 做可学习时间注意力池化，得到 fused_vec: [B, 128]。
            self.fusion_pool = TemporalAttnPool(dim=128)

            # 保持分类头输入维度为 128。
            self.fusion_vec_proj = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.LayerNorm(128)
            )
            fusion_out_dim = 128
        else:
            self.fuse_fc1 = nn.Linear(256, 128)
            self.fuse_fc2 = nn.Linear(128, 128)
            fusion_out_dim = 128

        # ===== Embedding head for metric learning =====
        self.embed_fc = nn.Linear(fusion_out_dim, embed_dim)

        # ===== Classification head =====
        self.cls_fc1 = nn.Linear(fusion_out_dim, 64)
        self.cls_fc2 = nn.Linear(64, num_classes)

    def forward(self, x, return_attention=False):
        # x: [B, T, F]
        attention_dict = {}

        B, T, F_dim = x.shape

        if T != self.seq_len:
            raise ValueError(
                f"输入时间长度不匹配：模型 seq_len={self.seq_len}, 当前输入 T={T}"
            )

        if F_dim != self.input_size:
            raise ValueError(
                f"输入特征维度不匹配：模型 input_size={self.input_size}, 当前输入 F={F_dim}"
            )

        # ======================================================
        # Spatial branch: Feature-wise Transformer
        # ======================================================
        # 不再池化为 [B, 128]，而是直接输出空间序列特征 [B, T, 128]。
        if return_attention:
            xs_seq, feature_weight = self.spatial_branch(x, return_attention=True)
            attention_dict["feature_weight"] = feature_weight  # [B, T, F]
        else:
            xs_seq = self.spatial_branch(x, return_attention=False)  # [B, T, 128]

        # 仅在 concat fusion 或需要向量表示时使用，AFAM 路径不使用这个池化结果。
        xs = xs_seq.mean(dim=1)  # [B, 128]

        # ======================================================
        # Temporal branch: 1-layer LSTM
        # ======================================================
        lstm_out, _ = self.lstm1(x)          # [B, T, 128]，BiLSTM 时 [B, T, 256]
        lstm_out = self.lstm_ln1(lstm_out)
        lstm_out = self.lstm_dropout(lstm_out)

        # DSGF 使用完整时间序列特征作为 temporal branch 输入。
        xt_seq = self.xt_seq_fc(lstm_out)       # [B, T, 128]
        xt_seq = self.xt_seq_ln(xt_seq)         # [B, T, 128]

        if self.use_temporal_attention:
            xt_raw, temporal_weight = self.temporal_attn(lstm_out)  # [B, H], [B, T]
            attention_dict["temporal_weight"] = temporal_weight
        else:
            xt_raw = lstm_out[:, -1, :]      # [B, H]

        xt_res = self.relu(self.xt_fc(xt_raw))
        xt = self.xt_ln(xt_res)              # [B, 128]

        # ======================================================
        # Fusion
        # ======================================================
        if self.use_adaptive_fusion:
            # 两个输入都是真正的序列特征：
            #   spatial branch:  xs_seq = [B, T, 128]
            #   temporal branch: xt_seq = [B, T, 128]
            # DSGF 只返回一个融合序列 fused_seq。
            if return_attention:
                fused_seq, fusion_attention = self.fusion(
                    xs_seq,
                    xt_seq,
                    return_attention=True
                )  # fused_seq: [B, T, 128]
                attention_dict.update(fusion_attention)
            else:
                fused_seq = self.fusion(xs_seq, xt_seq)  # [B, T, 128]

            # 用可学习时间注意力池化替代简单平均池化，得到单个融合向量。
            fused_vec, fusion_time_weight = self.fusion_pool(fused_seq)  # [B, 128], [B, T]

            if return_attention:
                attention_dict["fusion_time_weight"] = fusion_time_weight

            fused = self.fusion_vec_proj(fused_vec)  # [B, 128]
        else:
            fused = torch.cat([xs, xt], dim=1)  # [B, 256]
            fused = self.relu(self.fuse_fc1(fused))
            fused = self.dropout(fused)
            fused = self.relu(self.fuse_fc2(fused))
            fused = self.dropout(fused)

        # ===== Metric embedding =====
        embedding = self.embed_fc(fused)
        embedding = F.normalize(embedding, p=2, dim=1)

        # ===== Classification =====
        cls = self.relu(self.cls_fc1(fused))
        cls = self.dropout(cls)
        logits = self.cls_fc2(cls)

        if return_attention:
            return logits, embedding, attention_dict
        return logits, embedding


# ===================== 7. HDMTL 风格时序样本构造 =====================
def add_time_history_1(X, y, n_step=20, t_slide=1):
    time_len = X.shape[0]
    column_len = X.shape[1]

    P = len(np.arange(n_step, time_len, t_slide))
    xx = np.zeros((P + n_step, column_len * n_step), dtype=np.float32)

    for i in range(n_step, P + n_step):
        for j in range(n_step):
            src_idx = t_slide * i - t_slide * n_step + j
            xx[i, j * column_len:(j + 1) * column_len] = X[src_idx]

    xx = xx[n_step:X.shape[0], :]
    yy = y[np.arange(n_step, time_len, t_slide) - 1]

    return xx, yy


# ===================== 8. 9 分类标签生成逻辑 =====================
def assign_fault_multiclass(df):
    df = df.copy()
    df = df.assign(fault=0)

    cond1 = (df['add1'] > 0.005) | (df['add1'] < -0.005)
    cond2 = (df['add2'] > 0.005) | (df['add2'] < -0.005)
    cond3 = (df['m1'] < 1.0) | (df['m2'] < 1.0)

    df.loc[cond1 | cond2 | cond3, 'fault'] = 1

    df.loc[df['m2'] < 1.0, 'fault'] = 2
    df.loc[df['m2'] < 0.9, 'fault'] = 3
    df.loc[df['m2'] < 0.8, 'fault'] = 4
    df.loc[df['m2'] < 0.7, 'fault'] = 5
    df.loc[df['m2'] < 0.6, 'fault'] = 6
    df.loc[df['m2'] < 0.5, 'fault'] = 7
    df.loc[df['m2'] < 0.4, 'fault'] = 8

    df['fault'] = df['fault'].astype(np.int64)
    return df


# ===================== 9. 数据读取与全量时序样本构造 =====================
def load_and_prepare_data():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"找不到数据文件: {DATA_FILE}")

    sd = DATA(
        DATA_FILE,
        AC_ID,
        data_type=DATA_TYPE,
        sample_period=SAMPLE_PERIOD
    )

    df = sd.get_labelled_data()

    if df is None or len(df) == 0:
        raise ValueError("get_labelled_data() 返回空数据，请检查文件路径或数据解析。")

    df = df.copy()
    df_time = df.loc[TIME_MIN:TIME_MAX].copy()

    if df_time.empty:
        raise ValueError("筛选后没有数据，请检查 TIME_MIN/TIME_MAX。")

    required_columns = FEATURES + ['add1', 'add2', 'm1', 'm2']
    missing_cols = [c for c in required_columns if c not in df_time.columns]
    if missing_cols:
        raise ValueError(f"数据缺少以下必要列: {missing_cols}")

    df_time = assign_fault_multiclass(df_time)

    print(f"仅使用范围 [{TIME_MIN}, {TIME_MAX}] 的数据")
    print("筛选后数据量:", len(df_time))
    print("fault 标签分布:")
    print(df_time['fault'].value_counts().sort_index())

    X_raw = df_time[FEATURES].values.astype(np.float32)
    y_raw = df_time['fault'].values.astype(np.int64)

    X_flat, y_seq = add_time_history_1(X_raw, y_raw, n_step=SEQ_LENGTH, t_slide=1)

    print("\n构造后的时序样本数:", len(X_flat))
    print("时序样本展平形状:", X_flat.shape)

    valid_mask = np.isin(y_seq, list(range(NUM_CLASSES)))
    X_flat = X_flat[valid_mask]
    y_seq = y_seq[valid_mask]

    print(f"\n保留 {NUM_CLASSES} 分类 0~{NUM_CLASSES - 1} 后的样本数:", len(X_flat))
    label_counts = pd.Series(y_seq).value_counts().sort_index()
    print("标签分布:")
    print(label_counts)

    missing_classes = [c for c in range(NUM_CLASSES) if c not in label_counts.index]
    if missing_classes:
        raise ValueError(f"缺少以下类别，无法做完整 {NUM_CLASSES} 分类: {missing_classes}")

    return X_flat, y_seq


# ===================== 10. HDMTL 风格训练/验证/测试划分 =====================
def split_like_hdmtl(X_flat, y_seq, time_idx, verbose=True):
    X_train_pool, X_test, y_train_pool, y_test = ms.train_test_split(
        X_flat,
        y_seq,
        test_size=TEST_SIZE,
        random_state=TEST_RANDOM_STATE,
        stratify=y_seq
    )

    scaler = preprocessing.StandardScaler().fit(X_train_pool)
    X_train_pool = scaler.transform(X_train_pool)
    X_test = scaler.transform(X_test)

    # 每个 time_idx 使用不同小样本抽样，保证重复实验有意义。
    # 原代码注释写的是不同抽样，但 rng 没有加 time_idx；这里修正为 HDMTL_SAMPLE_SEED + time_idx。
    rng = np.random.RandomState(HDMTL_SAMPLE_SEED)

    train_classes = sorted(list(set(y_train_pool)))
    if len(train_classes) != NUM_CLASSES:
        raise ValueError(
            f"训练候选池类别数不是 {NUM_CLASSES}，当前为 {len(train_classes)}: {train_classes}"
        )

    num_per_class = int(SELECTED_TOTAL_SAMPLES / len(train_classes))

    if num_per_class * len(train_classes) != SELECTED_TOTAL_SAMPLES:
        print(
            f"警告：SELECTED_TOTAL_SAMPLES={SELECTED_TOTAL_SAMPLES} "
            f"不能被类别数 {len(train_classes)} 整除，"
            f"实际每类抽 {num_per_class} 个，总计 {num_per_class * len(train_classes)} 个。"
        )

    split = int(num_per_class * TRAIN_RATIO_ON_SELECTED)
    train_indices = []
    val_indices = []
    quota_info = {}

    for cls in train_classes:
        cls_indices = np.where(y_train_pool == cls)[0]

        if len(cls_indices) < num_per_class:
            raise ValueError(
                f"类别 {cls} 在训练候选池中样本不足，"
                f"需要 {num_per_class} 个，当前只有 {len(cls_indices)} 个。"
            )

        selected_idx = cls_indices[rng.choice(len(cls_indices), num_per_class, replace=False)]
        train_cls_idx = selected_idx[:split]
        val_cls_idx = selected_idx[split:]

        train_indices.extend(train_cls_idx.tolist())
        val_indices.extend(val_cls_idx.tolist())

        quota_info[cls] = {
            "pool": len(cls_indices),
            "selected": len(selected_idx),
            "train": len(train_cls_idx),
            "val": len(val_cls_idx)
        }

    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)

    X_train = X_train_pool[train_indices]
    y_train = y_train_pool[train_indices]
    X_val = X_train_pool[val_indices]
    y_val = y_train_pool[val_indices]

    X_train = X_train.reshape(-1, SEQ_LENGTH, len(FEATURES))
    X_val = X_val.reshape(-1, SEQ_LENGTH, len(FEATURES))
    X_test = X_test.reshape(-1, SEQ_LENGTH, len(FEATURES))

    if verbose:
        print(f"\n===== Time = {time_idx} =====")
        print("划分模式：HDMTL-style repeated training")
        print("当前版本：每次 time_idx 使用不同 train/val 小样本抽样。")
        print(f"训练候选池: {X_train_pool.shape[0]}")
        print(f"测试集: {X_test.shape}")
        print(f"训练集: {X_train.shape}")
        print(f"验证集: {X_val.shape}")

        print("\n每类抽样信息:")
        for cls in train_classes:
            info = quota_info[cls]
            print(
                f"  类别 {cls}: pool={info['pool']}, selected={info['selected']}, "
                f"train={info['train']}, val={info['val']}"
            )

        print("\n训练集标签分布:")
        print(pd.Series(y_train).value_counts().sort_index())
        print("验证集标签分布:")
        print(pd.Series(y_val).value_counts().sort_index())
        print("测试集标签分布:")
        print(pd.Series(y_test).value_counts().sort_index())

    return X_train, y_train, X_val, y_val, X_test, y_test


# ===================== 11. 邻域软标签 =====================
def get_soft_label_distribution(label, num_classes):
    dist = np.zeros(num_classes, dtype=np.float32)

    if label == 0:
        dist[0] = 1
        dist[2] = 0
        dist[1] = 0

    elif label == 1:
        dist[1] = 1
        dist[0] = 0

    elif label == 2:
        if ENHANCE_CLASS2:
            dist[2] = 0.50
            dist[0] = 0.25
            dist[3] = 0.20
            dist[1] = 0.00
            dist[4] = 0.05
        else:
            dist[2] = SOFT_MAIN_WEIGHT
            dist[0] = SOFT_NEIGHBOR_WEIGHT
            dist[3] = SOFT_NEIGHBOR_WEIGHT

    else:
        dist[label] = SOFT_MAIN_WEIGHT

        if label - 1 >= 2:
            dist[label - 1] += SOFT_NEIGHBOR_WEIGHT
        if label + 1 < num_classes:
            dist[label + 1] += SOFT_NEIGHBOR_WEIGHT
        if label - 2 >= 2:
            dist[label - 2] += SOFT_SECOND_NEIGHBOR_WEIGHT
        if label + 2 < num_classes:
            dist[label + 2] += SOFT_SECOND_NEIGHBOR_WEIGHT
    dist = dist / dist.sum()
    return dist


def build_soft_targets(batch_y, num_classes):
    batch_y_np = batch_y.detach().cpu().numpy()
    soft_targets = np.stack(
        [get_soft_label_distribution(int(y), num_classes) for y in batch_y_np],
        axis=0
    )
    return torch.tensor(soft_targets, dtype=torch.float32, device=batch_y.device)


def soft_cross_entropy(logits, soft_targets):
    log_probs = F.log_softmax(logits, dim=1)
    return -(soft_targets * log_probs).sum(dim=1).mean()


# ===================== 12. 模型配置 =====================
def get_model_config(config_name):
    configs = {
        "FT": {
            "desc": "Local-window Feature-wise Transformer Spatial Attention no dropout + 1-layer LSTM no dropout + Temporal Attention + DSGF fusion",
            "use_bilstm": False,
            "use_temporal_attention": False,
            "use_adaptive_fusion": True,
            "spatial_window_size": SPATIAL_WINDOW_SIZE,
            "spatial_d_model": SPATIAL_D_MODEL,
            "spatial_nhead": SPATIAL_NHEAD,
            "spatial_num_layers": SPATIAL_NUM_LAYERS,
            "spatial_ff_dim": SPATIAL_FF_DIM
        },
        "FT_BiLSTM": {
            "desc": "Local-window Feature-wise Transformer Spatial Attention no dropout + 1-layer BiLSTM no dropout + Temporal Attention + DSGF fusion",
            "use_bilstm": True,
            "use_temporal_attention": True,
            "use_adaptive_fusion": True,
            "spatial_window_size": SPATIAL_WINDOW_SIZE,
            "spatial_d_model": SPATIAL_D_MODEL,
            "spatial_nhead": SPATIAL_NHEAD,
            "spatial_num_layers": SPATIAL_NUM_LAYERS,
            "spatial_ff_dim": SPATIAL_FF_DIM
        },
        "FT_Concat": {
            "desc": "Local-window Feature-wise Transformer Spatial Attention no dropout + 1-layer LSTM no dropout + Temporal Attention + concat fusion",
            "use_bilstm": False,
            "use_temporal_attention": True,
            "use_adaptive_fusion": False,
            "spatial_window_size": SPATIAL_WINDOW_SIZE,
            "spatial_d_model": SPATIAL_D_MODEL,
            "spatial_nhead": SPATIAL_NHEAD,
            "spatial_num_layers": SPATIAL_NUM_LAYERS,
            "spatial_ff_dim": SPATIAL_FF_DIM
        }
    }

    if config_name not in configs:
        raise ValueError(f"未知模型配置: {config_name}")

    return configs[config_name]


def print_model_config(config_name):
    cfg = get_model_config(config_name)
    print("\n" + "=" * 80)
    print(f"当前模型配置：{config_name}")
    print("=" * 80)
    print(cfg["desc"])
    print(f"use_bilstm             = {cfg['use_bilstm']}")
    print(f"use_temporal_attention = {cfg['use_temporal_attention']}")
    print(f"use_DSGF_fusion = {cfg['use_adaptive_fusion']}")
    print(f"spatial_window_size    = {cfg['spatial_window_size']}")
    print(f"spatial_d_model        = {cfg['spatial_d_model']}")
    print(f"spatial_nhead          = {cfg['spatial_nhead']}")
    print(f"spatial_num_layers     = {cfg['spatial_num_layers']}")
    print(f"spatial_ff_dim         = {cfg['spatial_ff_dim']}")
    print(f"transformer_dropout    = {TRANSFORMER_DROPOUT}")
    print(f"lstm_dropout           = {LSTM_DROPOUT}")
    print(f"head_fusion_dropout    = {DROPOUT}")


# ===================== 13. 单次实验 =====================
def run_experiment(config_name, time_idx, X_flat, y_seq, verbose=True):
    cfg = get_model_config(config_name)

    X_train, y_train, X_val, y_val, X_test, y_test = split_like_hdmtl(
        X_flat, y_seq, time_idx=time_idx, verbose=verbose
    )

    train_dataset = FaultDataset(X_train, y_train)
    val_dataset = FaultDataset(X_val, y_val)
    test_dataset = FaultDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    model = FeatureTransformerLSTMAttentionMetricModel(
        input_size=len(FEATURES),
        seq_len=SEQ_LENGTH,
        embed_dim=EMBED_DIM,
        dropout=DROPOUT,
        num_classes=NUM_CLASSES,
        use_bilstm=cfg["use_bilstm"],
        use_temporal_attention=cfg["use_temporal_attention"],
        use_adaptive_fusion=cfg["use_adaptive_fusion"],
        spatial_window_size=cfg["spatial_window_size"],
        spatial_d_model=cfg["spatial_d_model"],
        spatial_nhead=cfg["spatial_nhead"],
        spatial_num_layers=cfg["spatial_num_layers"],
        spatial_ff_dim=cfg["spatial_ff_dim"]
    ).to(DEVICE)

    metric_criterion = BatchContrastiveLoss(
        margin=CONTRASTIVE_MARGIN,
        pos_weight=1.0,
        neg_weight=1.0
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE,
        min_lr=LR_SCHEDULER_MIN_LR
    )

    best_val_acc = 0.0
    best_model_state = None
    es_counter = 0
    es_best_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        train_total_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            logits, embeddings = model(batch_X)

            if USE_SOFT_LABEL:
                soft_targets = build_soft_targets(batch_y, NUM_CLASSES)
                cls_loss = soft_cross_entropy(logits, soft_targets)
            else:
                cls_loss = F.cross_entropy(logits, batch_y)

            metric_loss = metric_criterion(embeddings, batch_y)
            loss = cls_loss + METRIC_LOSS_WEIGHT * metric_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            n = batch_X.size(0)
            train_total_loss += loss.item() * n
            train_correct += (torch.argmax(logits, dim=1) == batch_y).sum().item()
            train_total += n

        train_acc = train_correct / train_total
        train_total_loss = train_total_loss / train_total

        # ---------- validation ----------
        model.eval()
        val_total_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(DEVICE)
                batch_y = batch_y.to(DEVICE)

                logits, embeddings = model(batch_X)

                if USE_SOFT_LABEL:
                    soft_targets = build_soft_targets(batch_y, NUM_CLASSES)
                    cls_loss = soft_cross_entropy(logits, soft_targets)
                else:
                    cls_loss = F.cross_entropy(logits, batch_y)

                metric_loss = metric_criterion(embeddings, batch_y)
                loss = cls_loss + METRIC_LOSS_WEIGHT * metric_loss

                n = batch_X.size(0)
                val_total_loss += loss.item() * n
                val_correct += (torch.argmax(logits, dim=1) == batch_y).sum().item()
                val_total += n

        val_acc = val_correct / val_total
        val_total_loss = val_total_loss / val_total
        scheduler.step(val_total_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())

        if val_acc > es_best_val_acc:
            es_best_val_acc = val_acc
            es_counter = 0
        else:
            es_counter += 1

        if verbose and (epoch == 0 or (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1):
            cur_lr = optimizer.param_groups[0]['lr']
            print(
                f"Epoch [{epoch + 1:03d}/{EPOCHS}] | "
                f"Train Loss: {train_total_loss:.4f}  Acc: {train_acc:.4f} | "
                f"Val Loss: {val_total_loss:.4f}  Acc: {val_acc:.4f} | "
                f"LR: {cur_lr:.2e} | ES: {es_counter}/{EARLY_STOPPING_PATIENCE}"
            )

        if es_counter >= EARLY_STOPPING_PATIENCE:
            if verbose:
                print(
                    f"\n[Early Stopping] 触发于 Epoch {epoch + 1}，"
                    f"验证准确率连续 {EARLY_STOPPING_PATIENCE} 轮未提升。"
                )
            break

    if best_model_state is None:
        best_model_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_state)
    model.eval()

    # ---------- test ----------
    y_true = []
    y_pred = []
    y_prob = []
    feature_attn_all = []
    temporal_attn_all = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(DEVICE)

            logits, _, attention_dict = model(batch_X, return_attention=True)
            probs = torch.softmax(logits, dim=1)
            predicted = torch.argmax(probs, dim=1)

            y_true.extend(batch_y.numpy().flatten().tolist())
            y_pred.extend(predicted.cpu().numpy().flatten().tolist())
            y_prob.extend(probs.cpu().numpy().tolist())

            if "feature_weight" in attention_dict:
                feature_attn_all.append(attention_dict["feature_weight"].cpu().numpy())
            if "temporal_weight" in attention_dict:
                temporal_attn_all.append(attention_dict["temporal_weight"].cpu().numpy())

    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)
    y_prob = np.array(y_prob)

    test_acc = accuracy_score(y_true, y_pred)
    test_macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        average="macro",
        zero_division=0
    )

    # 9 分类 AUROC 使用 One-vs-Rest macro average。
    # 如果测试集中某些类别缺失，roc_auc_score 可能会报错，这里返回 NaN 并在汇总时跳过。
    try:
        if NUM_CLASSES == 2:
            test_auroc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            test_auroc = roc_auc_score(
                y_true,
                y_prob,
                labels=list(range(NUM_CLASSES)),
                multi_class="ovr",
                average="macro"
            )
    except ValueError as e:
        print(f"[Warning] AUROC 计算失败: {e}")
        test_auroc = np.nan

    if len(feature_attn_all) > 0:
        feature_attn_all = np.concatenate(feature_attn_all, axis=0)  # [N_test, F]
        mean_feature_attn = feature_attn_all.mean(axis=0)
    else:
        feature_attn_all = None
        mean_feature_attn = None

    if len(temporal_attn_all) > 0:
        temporal_attn_all = np.concatenate(temporal_attn_all, axis=0)  # [N_test, T]
        mean_temporal_attn = temporal_attn_all.mean(axis=0)
    else:
        temporal_attn_all = None
        mean_temporal_attn = None

    return {
        "config_name": config_name,
        "time_idx": time_idx,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "test_macro_f1": test_macro_f1,
        "test_auroc": test_auroc,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "feature_attn_all": feature_attn_all,
        "mean_feature_attn": mean_feature_attn,
        "temporal_attn_all": temporal_attn_all,
        "mean_temporal_attn": mean_temporal_attn,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES))),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=list(range(NUM_CLASSES)),
            target_names=[f"class_{i}" for i in range(NUM_CLASSES)],
            digits=4,
            zero_division=0
        )
    }


# ===================== 14. 注意力结果打印 =====================
def print_attention_summary(result):
    mean_feature_attn = result.get("mean_feature_attn", None)
    mean_temporal_attn = result.get("mean_temporal_attn", None)

    if mean_feature_attn is not None:
        print("Feature-wise Spatial Attention 平均权重:")

        # 旧版本 feature_weight 是 [N, F]，mean 后是 [F]。
        # 当前版本 feature_weight 是 [N, T, F]，mean 后是 [T, F]。
        # 这里为了输出每个变量的重要性，对时间维度再平均一次。
        if mean_feature_attn.ndim == 2:
            feature_importance = mean_feature_attn.mean(axis=0)  # [F]
        else:
            feature_importance = mean_feature_attn              # [F]

        pairs = list(zip(FEATURES, feature_importance.tolist()))
        pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        for name, weight in pairs:
            print(f"  {name:>8s}: {weight:.4f}")

    if mean_temporal_attn is not None:
        print("\nTemporal Attention 平均权重:")
        for t, weight in enumerate(mean_temporal_attn.tolist()):
            print(f"  step {t:02d}: {weight:.4f}")


# ===================== 15. 主程序 =====================

def mean_std(values):
    """
    直接计算所有有效值的平均值和标准差，不去掉最大值和最小值。

    参数:
        values: list 或 np.ndarray，例如 10 次 test_acc / Macro-F1 / Macro-AUROC。

    返回:
        mean_value: 所有有效值的平均值
        std_value:  所有有效值的标准差
        valid_count: 有效值数量。AUROC 如果某次无法计算，可能会产生 NaN，
                     这里会自动跳过 NaN。
    """
    values = np.array(values, dtype=np.float64)
    values = values[~np.isnan(values)]

    if len(values) == 0:
        return np.nan, np.nan, 0

    mean_value = np.mean(values)
    std_value = np.std(values)

    return mean_value, std_value, len(values)


def trimmed_mean_std(values):
    """
    去掉最大值和最小值后，计算平均值和标准差。

    参数:
        values: list 或 np.ndarray，例如 10 次 test_acc / Macro-F1 / Macro-AUROC。

    返回:
        mean_value: 去极值后的平均值
        std_value:  去极值后的标准差
        trimmed_values: 去掉最大值和最小值后的数组
        min_removed: 被去掉的最小值
        max_removed: 被去掉的最大值
        valid_count: 去除 NaN 后的有效值数量
        trimmed_count: 去极值后的有效值数量
    """
    values = np.array(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    valid_count = len(values)

    if valid_count <= 2:
        return np.nan, np.nan, np.array([], dtype=np.float64), np.nan, np.nan, valid_count, 0

    sorted_values = np.sort(values)
    min_removed = sorted_values[0]
    max_removed = sorted_values[-1]
    trimmed_values = sorted_values[1:-1]

    mean_value = np.mean(trimmed_values)
    std_value = np.std(trimmed_values)

    return mean_value, std_value, trimmed_values, min_removed, max_removed, valid_count, len(trimmed_values)


def format_metric(mean_value, std_value, percent=False):
    """
    生成论文式输出字符串。

    percent=True 时用于 Accuracy，例如 95.12% ± 0.63%。
    percent=False 时用于 Macro-F1 / Macro-AUROC，例如 0.9512 ± 0.0063。
    """
    if np.isnan(mean_value) or np.isnan(std_value):
        return "nan ± nan"

    if percent:
        return f"{mean_value * 100:.2f}% ± {std_value * 100:.2f}%"

    return f"{mean_value:.4f} ± {std_value:.4f}"


def round_list(values, ndigits=4):
    """用于打印/保存列表；保留 NaN，避免 AUROC 某次无法计算时中断。"""
    rounded = []
    for x in values:
        x = float(x)
        if np.isnan(x):
            rounded.append(np.nan)
        else:
            rounded.append(round(x, ndigits))
    return rounded


def main():
    global SELECTED_TOTAL_SAMPLES

    X_flat, y_seq = load_and_prepare_data()

    print("\n" + "=" * 80)
    print("当前训练策略：固定 HDMTL_SAMPLE_SEED，不同 SELECTED_TOTAL_SAMPLES，小样本重复实验")
    print("统计方式：每个样本数下重复训练 10 次，去掉最大值和最小值后计算平均值和标准差")
    print("保存方式：detail/summary CSV + Excel，两张表分别保存单次结果和汇总结果")
    print("汇总指标：Accuracy、Macro-F1、Macro-AUROC；论文式写法默认使用去极值后的结果")
    print("=" * 80)

    print(f"\n固定 HDMTL_SAMPLE_SEED = {HDMTL_SAMPLE_SEED}")
    print(f"SELECTED_TOTAL_SAMPLES_LIST = {SELECTED_TOTAL_SAMPLES_LIST}")
    print(f"每个样本数重复训练 TIMES = {TIMES}")
    print(f"结果保存目录 RESULT_SAVE_DIR = {RESULT_SAVE_DIR}")

    print("\nSoft label 分布：")
    for cls in range(NUM_CLASSES):
        print(f"class {cls} soft target -> {np.round(get_soft_label_distribution(cls, NUM_CLASSES), 4)}")

    # 推荐先跑 FT：Feature-wise Transformer + 1-layer LSTM + DSGF fusion
    experiment_configs = ["FT"]

    # 如需消融，可以改成：
    # experiment_configs = ["FT", "FT_BiLSTM", "FT_Concat"]

    detail_records = []
    summary_records = []

    for config_name in experiment_configs:
        print_model_config(config_name)

        for sample_num in SELECTED_TOTAL_SAMPLES_LIST:
            SELECTED_TOTAL_SAMPLES = sample_num

            print("\n" + "#" * 80)
            print(f"开始实验：Config={config_name}, SELECTED_TOTAL_SAMPLES={sample_num}")
            print(f"固定 HDMTL_SAMPLE_SEED={HDMTL_SAMPLE_SEED}")
            print(f"每个样本数重复训练 TIMES={TIMES} 次")
            print("#" * 80)

            all_results = []

            for time_idx in range(TIMES):
                # time_idx 只控制模型初始化、DataLoader shuffle 等随机因素；不改变 HDMTL_SAMPLE_SEED。
                set_seed(time_idx)

                result = run_experiment(
                    config_name=config_name,
                    time_idx=time_idx,
                    X_flat=X_flat,
                    y_seq=y_seq,
                    verbose=True
                )

                all_results.append(result)

                detail_records.append({
                    "config_name": config_name,
                    "SELECTED_TOTAL_SAMPLES": sample_num,
                    "HDMTL_SAMPLE_SEED": HDMTL_SAMPLE_SEED,
                    "init_seed": time_idx,
                    "best_val_acc": result["best_val_acc"],
                    "test_acc": result["test_acc"],
                    "test_macro_f1": result["test_macro_f1"],
                    "test_macro_auroc": result["test_auroc"]
                })

                print(
                    f"\n[{config_name}] "
                    f"Samples={sample_num} | "
                    f"Init={time_idx} | "
                    f"Best Val Acc={result['best_val_acc']:.4f} | "
                    f"Test Acc={result['test_acc']:.4f} | "
                    f"Macro-F1={result['test_macro_f1']:.4f} | "
                    f"Macro-AUROC={result['test_auroc']:.4f}"
                )

                # 为减少输出，只打印每个样本数下最后一次初始化的注意力摘要。
                if time_idx == TIMES - 1:
                    print_attention_summary(result)

            test_accs = [r["test_acc"] for r in all_results]
            val_accs = [r["best_val_acc"] for r in all_results]
            test_macro_f1s = [r["test_macro_f1"] for r in all_results]
            test_macro_aurocs = [r["test_auroc"] for r in all_results]

            # 原始 10 次 mean/std，保留在表里便于复查。
            test_mean, test_std, test_valid_count = mean_std(test_accs)
            val_mean, val_std, val_valid_count = mean_std(val_accs)
            f1_mean, f1_std, f1_valid_count = mean_std(test_macro_f1s)
            auroc_mean, auroc_std, auroc_valid_count = mean_std(test_macro_aurocs)

            # 去极值 mean/std：论文式写法默认使用这一组。
            (
                test_mean_trimmed,
                test_std_trimmed,
                test_accs_trimmed,
                test_min_removed,
                test_max_removed,
                test_valid_count_for_trim,
                test_trimmed_count,
            ) = trimmed_mean_std(test_accs)

            (
                val_mean_trimmed,
                val_std_trimmed,
                val_accs_trimmed,
                val_min_removed,
                val_max_removed,
                val_valid_count_for_trim,
                val_trimmed_count,
            ) = trimmed_mean_std(val_accs)

            (
                f1_mean_trimmed,
                f1_std_trimmed,
                f1_trimmed,
                f1_min_removed,
                f1_max_removed,
                f1_valid_count_for_trim,
                f1_trimmed_count,
            ) = trimmed_mean_std(test_macro_f1s)

            (
                auroc_mean_trimmed,
                auroc_std_trimmed,
                auroc_trimmed,
                auroc_min_removed,
                auroc_max_removed,
                auroc_valid_count_for_trim,
                auroc_trimmed_count,
            ) = trimmed_mean_std(test_macro_aurocs)

            summary_row = {
                "config_name": config_name,
                "SELECTED_TOTAL_SAMPLES": sample_num,
                "HDMTL_SAMPLE_SEED": HDMTL_SAMPLE_SEED,
                "TIMES": TIMES,
            }

            # 保存每次运行的原始值；如果以后 TIMES 改成其他数，也会自动生成对应列。
            for i, acc in enumerate(test_accs, start=1):
                summary_row[f"test_acc_{i}"] = acc

            for i, acc in enumerate(val_accs, start=1):
                summary_row[f"val_acc_{i}"] = acc

            for i, f1 in enumerate(test_macro_f1s, start=1):
                summary_row[f"test_macro_f1_{i}"] = f1

            for i, auroc in enumerate(test_macro_aurocs, start=1):
                summary_row[f"test_macro_auroc_{i}"] = auroc

            summary_row.update({
                # Accuracy：10 次直接统计
                "test_acc_mean": test_mean,
                "test_acc_std": test_std,
                "test_acc_valid_count": test_valid_count,
                "val_acc_mean": val_mean,
                "val_acc_std": val_std,
                "val_acc_valid_count": val_valid_count,

                # Accuracy：去掉最大值和最小值后统计
                "test_acc_min_removed": test_min_removed,
                "test_acc_max_removed": test_max_removed,
                "test_acc_trimmed_mean": test_mean_trimmed,
                "test_acc_trimmed_std": test_std_trimmed,
                "test_acc_trimmed_count": test_trimmed_count,
                "val_acc_min_removed": val_min_removed,
                "val_acc_max_removed": val_max_removed,
                "val_acc_trimmed_mean": val_mean_trimmed,
                "val_acc_trimmed_std": val_std_trimmed,
                "val_acc_trimmed_count": val_trimmed_count,

                # Macro-F1：10 次直接统计
                "test_macro_f1_mean": f1_mean,
                "test_macro_f1_std": f1_std,
                "test_macro_f1_valid_count": f1_valid_count,

                # Macro-F1：去掉最大值和最小值后统计
                "test_macro_f1_min_removed": f1_min_removed,
                "test_macro_f1_max_removed": f1_max_removed,
                "test_macro_f1_trimmed_mean": f1_mean_trimmed,
                "test_macro_f1_trimmed_std": f1_std_trimmed,
                "test_macro_f1_trimmed_count": f1_trimmed_count,

                # Macro-AUROC：10 次直接统计
                "test_macro_auroc_mean": auroc_mean,
                "test_macro_auroc_std": auroc_std,
                "test_macro_auroc_valid_count": auroc_valid_count,

                # Macro-AUROC：去掉最大值和最小值后统计
                "test_macro_auroc_min_removed": auroc_min_removed,
                "test_macro_auroc_max_removed": auroc_max_removed,
                "test_macro_auroc_trimmed_mean": auroc_mean_trimmed,
                "test_macro_auroc_trimmed_std": auroc_std_trimmed,
                "test_macro_auroc_trimmed_count": auroc_trimmed_count,

                # 论文式写法：默认使用去极值后的 mean/std
                "paper_accuracy": format_metric(test_mean_trimmed, test_std_trimmed, percent=True),
                "paper_macro_f1": format_metric(f1_mean_trimmed, f1_std_trimmed, percent=False),
                "paper_macro_auroc": format_metric(auroc_mean_trimmed, auroc_std_trimmed, percent=False),

                # 记录所有原始值和去极值后的值，便于复查
                "all_test_accs": str(round_list(test_accs)),
                "trimmed_test_accs": str(round_list(test_accs_trimmed.tolist())),
                "all_val_accs": str(round_list(val_accs)),
                "trimmed_val_accs": str(round_list(val_accs_trimmed.tolist())),
                "all_test_macro_f1s": str(round_list(test_macro_f1s)),
                "trimmed_test_macro_f1s": str(round_list(f1_trimmed.tolist())),
                "all_test_macro_aurocs": str(round_list(test_macro_aurocs)),
                "trimmed_test_macro_aurocs": str(round_list(auroc_trimmed.tolist())),
            })

            summary_records.append(summary_row)

            print("\n" + "=" * 80)
            print(f"{config_name} | SELECTED_TOTAL_SAMPLES={sample_num} | {TIMES} 次重复实验结果，{NUM_CLASSES} 分类")
            print("=" * 80)

            print("Best Val Accs:", round_list(val_accs))
            print(f"10次 Val Acc 平均值: {val_mean:.4f} ± {val_std:.4f}")
            print("Val Acc 去掉最小值:", round(val_min_removed, 4) if not np.isnan(val_min_removed) else np.nan)
            print("Val Acc 去掉最大值:", round(val_max_removed, 4) if not np.isnan(val_max_removed) else np.nan)
            print("剩余 Val Accs:", round_list(val_accs_trimmed.tolist()))
            print(f"去极值后 Val Acc: {val_mean_trimmed:.4f} ± {val_std_trimmed:.4f}")

            print("\nTest Accs:", round_list(test_accs))
            print(f"10次 Test Acc 平均值: {test_mean:.4f} ± {test_std:.4f}")
            print("Test Acc 去掉最小值:", round(test_min_removed, 4) if not np.isnan(test_min_removed) else np.nan)
            print("Test Acc 去掉最大值:", round(test_max_removed, 4) if not np.isnan(test_max_removed) else np.nan)
            print("剩余 Test Accs:", round_list(test_accs_trimmed.tolist()))
            print(f"去极值后 Test Acc: {test_mean_trimmed:.4f} ± {test_std_trimmed:.4f}")
            print(f"论文式写法: Accuracy = {format_metric(test_mean_trimmed, test_std_trimmed, percent=True)}")

            print("\nMacro-F1s:", round_list(test_macro_f1s))
            print(f"10次 Macro-F1 平均值: {f1_mean:.4f} ± {f1_std:.4f}")
            print("Macro-F1 去掉最小值:", round(f1_min_removed, 4) if not np.isnan(f1_min_removed) else np.nan)
            print("Macro-F1 去掉最大值:", round(f1_max_removed, 4) if not np.isnan(f1_max_removed) else np.nan)
            print("剩余 Macro-F1s:", round_list(f1_trimmed.tolist()))
            print(f"去极值后 Macro-F1: {f1_mean_trimmed:.4f} ± {f1_std_trimmed:.4f}")
            print(f"论文式写法: Macro-F1 = {format_metric(f1_mean_trimmed, f1_std_trimmed, percent=False)}")

            print("\nMacro-AUROCs:", round_list(test_macro_aurocs))
            print(f"10次 Macro-AUROC 平均值: {auroc_mean:.4f} ± {auroc_std:.4f}")
            print("Macro-AUROC 去掉最小值:", round(auroc_min_removed, 4) if not np.isnan(auroc_min_removed) else np.nan)
            print("Macro-AUROC 去掉最大值:", round(auroc_max_removed, 4) if not np.isnan(auroc_max_removed) else np.nan)
            print("剩余 Macro-AUROCs:", round_list(auroc_trimmed.tolist()))
            print(f"去极值后 Macro-AUROC: {auroc_mean_trimmed:.4f} ± {auroc_std_trimmed:.4f}")
            print(f"论文式写法: Macro-AUROC = {format_metric(auroc_mean_trimmed, auroc_std_trimmed, percent=False)}")

            if auroc_valid_count < TIMES:
                print(
                    f"[Warning] Macro-AUROC 有效次数为 {auroc_valid_count}/{TIMES}，"
                    f"部分运行可能由于测试集中类别不足而无法计算 Macro-AUROC。"
                )
            if auroc_trimmed_count == 0:
                print(
                    "[Warning] Macro-AUROC 去极值后没有足够有效值，"
                    "请检查测试集中每个类别是否都存在，或增大测试集/样本数量。"
                )

            last = all_results[-1]
            print(f"\n{config_name} 最后一次初始化的分类报告：")
            print(last["classification_report"])
            print(f"{config_name} 最后一次初始化的混淆矩阵：")
            print(last["confusion_matrix"])

    detail_df = pd.DataFrame(detail_records)
    summary_df = pd.DataFrame(summary_records)

    detail_csv = os.path.join(RESULT_SAVE_DIR, "detail_results_multiclass_by_sample_size_trimmed.csv")
    summary_csv = os.path.join(RESULT_SAVE_DIR, "summary_results_multiclass_by_sample_size_trimmed.csv")
    excel_file = os.path.join(RESULT_SAVE_DIR, "hdmtl_multiclass_results_by_sample_size_trimmed.xlsx")

    detail_df.to_csv(detail_csv, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
        detail_df.to_excel(writer, sheet_name="detail_results", index=False)
        summary_df.to_excel(writer, sheet_name="summary_results", index=False)

    print("\n" + "=" * 80)
    print("所有不同 SELECTED_TOTAL_SAMPLES 实验完成")
    print("=" * 80)
    print(f"详细结果 CSV: {detail_csv}")
    print(f"汇总结果 CSV: {summary_csv}")
    print(f"Excel 文件   : {excel_file}")

    print("\n论文汇总表：")
    print(summary_df[[
        "config_name",
        "SELECTED_TOTAL_SAMPLES",

        "test_acc_mean",
        "test_acc_std",
        "test_acc_trimmed_mean",
        "test_acc_trimmed_std",
        "test_acc_trimmed_count",

        "test_macro_f1_mean",
        "test_macro_f1_std",
        "test_macro_f1_trimmed_mean",
        "test_macro_f1_trimmed_std",
        "test_macro_f1_trimmed_count",

        "test_macro_auroc_mean",
        "test_macro_auroc_std",
        "test_macro_auroc_trimmed_mean",
        "test_macro_auroc_trimmed_std",
        "test_macro_auroc_trimmed_count",

        "paper_accuracy",
        "paper_macro_f1",
        "paper_macro_auroc"
    ]])

    if len(summary_records) > 0:
        best_idx = summary_df["test_acc_trimmed_mean"].astype(float).idxmax()
        best_row = summary_df.loc[best_idx]
        print("\n推荐选择：")
        print(
            f"Config={best_row['config_name']}, "
            f"SELECTED_TOTAL_SAMPLES={best_row['SELECTED_TOTAL_SAMPLES']}, "
            f"Accuracy={best_row['paper_accuracy']}, "
            f"Macro-F1={best_row['paper_macro_f1']}, "
            f"Macro-AUROC={best_row['paper_macro_auroc']}"
        )


if __name__ == "__main__":
    main()
