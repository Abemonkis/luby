"""
model_v37.py —— V37 完整版关系感知分支 + 官方保底分支（双专家）

V37 的"新专家"是对官方 Polynormer 的完整再设计，核心是"关系感知三件套"：
  1. graph_context：可微的逐节点图上下文统计
     stats = [邻居一致性 agreement, 一致性方差 variance, 度可靠性 reliability]
  2. 局部块 RelationAwareLocalBlock：在官方局部层上注入
     · filter  差异滤波：对"自身残差 - 邻居消息"的差异做低秩非线性变换
     · route   软路由：softmax(MLP(stats)) 决定差异注入强度（已被证明冗余）
     · 逐节点 beta：β = β_base + Δ(MLP(stats))，多项式混合比例逐节点自适应
  3. 全局块 ContextualGlobalBlock：线性注意力 + context_scale(QKV 调制) + 输出门控

no_qkv 瘦身 = 本文件 + --ablation no_qkv_context,no_depth_weight：
  · _NO_QKV_CONTEXT：去掉全局块 QKV 的 stats 调制（context_scale）
  · _NO_DEPTH_WEIGHT：去掉局部层深度加权（depth_score），各层简单求和
每个开关由 set_ablate_* 全局变量控制，main_v37.py 按 --ablation 设置。
"""
import math

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import scatter

from model_official_v35 import Polynormer as OfficialPolynormer

# ---- 消融全局开关（由 main 按 --ablation 设置，默认全部关闭 = 完整 V37）----
_NO_GRAPH_CONTEXT = False   # 去掉 graph context stats（置零）
_NO_RELATION_ROUTE = False  # 完全去掉差异注入（回到官方局部更新）
_NO_QKV_CONTEXT = False     # 去掉全局块 QKV 的 context 调制
_NO_DEPTH_WEIGHT = False    # 去掉局部层深度加权（均匀求和）
_NO_FILTER = False          # 去掉 filter 非线性（差异用恒等 residual-message）
_NO_BETA_CTX = False        # 去掉逐节点 beta 的上下文调制
_NO_ROUTE_SELECTION = False # 路由改为均匀 0.5（保留差异信号，去掉选择偏好）



def set_ablate_graph_context(flag):
    global _NO_GRAPH_CONTEXT
    _NO_GRAPH_CONTEXT = flag


def set_ablate_relation_route(flag):
    global _NO_RELATION_ROUTE
    _NO_RELATION_ROUTE = flag


def set_ablate_qkv_context(flag):
    global _NO_QKV_CONTEXT
    _NO_QKV_CONTEXT = flag


def set_ablate_depth_weight(flag):
    global _NO_DEPTH_WEIGHT
    _NO_DEPTH_WEIGHT = flag


def set_ablate_filter(flag):
    global _NO_FILTER
    _NO_FILTER = flag


def set_ablate_beta_ctx(flag):
    global _NO_BETA_CTX
    _NO_BETA_CTX = flag


def set_ablate_route_selection(flag):
    global _NO_ROUTE_SELECTION
    _NO_ROUTE_SELECTION = flag


def graph_context(x, edge_index):
    """Differentiable node context: agreement, uncertainty, and reliability."""
    src, dst = edge_index
    keep = src != dst
    src, dst = src[keep], dst[keep]
    z = F.normalize(x, p=2, dim=-1, eps=1e-12)
    agreement = 0.5 * ((z[src] * z[dst]).sum(-1) + 1.0)
    mean = scatter(agreement, dst, dim=0, dim_size=x.size(0), reduce='mean')
    second = scatter(agreement.square(), dst, dim=0,
                     dim_size=x.size(0), reduce='mean')
    variance = (second - mean.square()).clamp_min(0.0)
    degree = scatter(torch.ones_like(agreement), dst, dim=0,
                     dim_size=x.size(0), reduce='sum')
    reliability = degree / (degree + 2.0)
    stats = torch.stack((mean, variance, reliability), dim=-1)
    if _NO_GRAPH_CONTEXT:
        stats = torch.zeros_like(stats)
    return stats, src, dst


class ContextInputEncoder(torch.nn.Module):
    def __init__(self, in_channels, channels):
        super().__init__()
        self.feature = torch.nn.Linear(in_channels, channels)
        self.norm = torch.nn.LayerNorm(channels)
        self.context = torch.nn.Sequential(
            torch.nn.Linear(3, min(32, channels)), torch.nn.SiLU(),
            torch.nn.Linear(min(32, channels), channels))
        self.context_strength = torch.nn.Parameter(torch.tensor(-2.0))

    def reset_parameters(self):
        self.feature.reset_parameters()
        self.norm.reset_parameters()
        for layer in self.context:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        torch.nn.init.constant_(self.context_strength, -2.0)

    def forward(self, x, edge_index):
        x = self.feature(x)
        stats, _, _ = graph_context(x, edge_index)
        context = self.context(stats)
        x = self.norm(x + torch.sigmoid(self.context_strength) * context)
        return x, stats


class RelationAwareLocalBlock(torch.nn.Module):
    """关系感知局部块（V37 新专家对官方局部层的核心改动）。

    官方局部层：x = (1-β)*LN(h·x) + β*x（只聚合邻居消息）。
    本块在 message 之外显式引入"自身残差 - 邻居消息"的差异信号：
      message   = GAT(z, edge_index)                  # 邻居聚合消息
      residual  = residual_lin(x)                     # 自身线性映射
      discrepancy = filter(residual - message)        # 差异滤波（核心有效模块）
      update    = message + residual + π·discrepancy  # π: 路由(softmax) 或 固定 0.5
      beta      = clamp/sigmoid(beta_base + Δ(stats)) # 逐节点多项式混合系数
      return (1-β)·LN(SiLU(residual)·update) + β·update
    意义：官方只"吸收邻居"，本块显式建模"我 vs 邻居差在哪"，
    滤波后的差异提供独立于消息聚合的判别信号。
    """
    def __init__(self, channels, hidden_channels, heads, beta, dropout):
        super().__init__()
        rank = min(32, channels)
        self.beta_mode = beta
        self.dropout = dropout
        self.pre_norm = torch.nn.LayerNorm(channels)
        self.gat = GATConv(channels, hidden_channels, heads=heads,
                           concat=True, add_self_loops=False, bias=False)
        self.residual = torch.nn.Linear(channels, channels)
        self.filter = torch.nn.Sequential(
            torch.nn.Linear(channels, rank, bias=False), torch.nn.SiLU(),
            torch.nn.Linear(rank, channels, bias=False))
        self.route = torch.nn.Sequential(
            torch.nn.Linear(3, rank), torch.nn.SiLU(), torch.nn.Linear(rank, 2))
        self.beta_context = torch.nn.Sequential(
            torch.nn.Linear(3, rank), torch.nn.SiLU(),
            torch.nn.Linear(rank, channels))
        if beta < 0:
            self.beta_base = torch.nn.Parameter(torch.zeros(channels))
        else:
            self.beta_base = torch.nn.Parameter(torch.full((channels,), beta))
        self.post_norm = torch.nn.LayerNorm(channels)

    def reset_parameters(self):
        self.pre_norm.reset_parameters()
        self.gat.reset_parameters()
        self.residual.reset_parameters()
        self.post_norm.reset_parameters()
        for module in (self.filter, self.route, self.beta_context):
            for layer in module:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        torch.nn.init.zeros_(self.beta_context[-1].weight)
        torch.nn.init.zeros_(self.beta_context[-1].bias)
        torch.nn.init.zeros_(self.route[-1].weight)
        with torch.no_grad():
            self.route[-1].bias.copy_(torch.tensor([1.5, -1.5]))
        if self.beta_mode < 0:
            torch.nn.init.zeros_(self.beta_base)
        else:
            torch.nn.init.constant_(self.beta_base, self.beta_mode)

    def forward(self, x, edge_index, stats):
        z = self.pre_norm(x)
        message = self.gat(z, edge_index)
        residual = self.residual(x)
        route = F.softmax(self.route(stats), dim=-1)
        if _NO_FILTER:
            # 恒等滤波（去掉 filter 非线性变换，保留差异信号）
            discrepancy = residual - message
        else:
            discrepancy = self.filter(residual - message)
        if _NO_RELATION_ROUTE:
            # 旧语义（v37 主实验）：完全去掉差异注入
            update = message + residual
        elif _NO_ROUTE_SELECTION:
            # 均匀路由（π 固定 0.5，无选择偏好），保留差异信号
            pi = torch.full_like(route[:, 1:2], 0.5)
            update = message + residual + pi * discrepancy
        else:
            update = message + residual + route[:, 1:2] * discrepancy
        update = F.silu(update)
        update = F.dropout(update, p=self.dropout, training=self.training)
        delta = 0.2 * torch.tanh(self.beta_context(stats)) if not _NO_BETA_CTX \
            else torch.zeros_like(self.beta_base)
        if self.beta_mode < 0:
            beta = torch.sigmoid(self.beta_base.unsqueeze(0) + delta)
        else:
            beta = (self.beta_base.unsqueeze(0) + delta).clamp(0.0, 1.0)
        interaction = self.post_norm(F.silu(residual) * update)
        return (1.0 - beta) * interaction + beta * update


class ContextualGlobalBlock(torch.nn.Module):
    """上下文感知全局块（线性注意力 + QKV 上下文调制）。

    与官方 GlobalAttn 相比：
      · 官方全局注意力用 sigmoid(q/k) 做线性注意力；
      · 本块额外用 stats 调制 Q/K 幅值（context_scale，1+0.25·tanh），
        使全局注意强度随邻居一致性等上下文自适应——no_qkv 瘦身即关闭此项；
      · 输出再经 mix(stats) 门控的残差与 FFN。
    """
    def __init__(self, channels, dropout):
        super().__init__()
        rank = min(32, channels)
        self.dropout = dropout
        self.norm1 = torch.nn.LayerNorm(channels)
        self.q = torch.nn.Linear(channels, channels)
        self.k = torch.nn.Linear(channels, channels)
        self.v = torch.nn.Linear(channels, channels)
        self.context_scale = torch.nn.Sequential(
            torch.nn.Linear(3, rank), torch.nn.SiLU(), torch.nn.Linear(rank, 2))
        self.mix = torch.nn.Sequential(
            torch.nn.Linear(3, rank), torch.nn.SiLU(), torch.nn.Linear(rank, channels))
        self.norm2 = torch.nn.LayerNorm(channels)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(channels, 2 * channels), torch.nn.SiLU(),
            torch.nn.Dropout(dropout), torch.nn.Linear(2 * channels, channels))

    def reset_parameters(self):
        self.norm1.reset_parameters()
        self.norm2.reset_parameters()
        for module in (self.q, self.k, self.v):
            module.reset_parameters()
        for module in (self.context_scale, self.mix, self.ffn):
            for layer in module:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        torch.nn.init.zeros_(self.mix[-1].weight)
        torch.nn.init.zeros_(self.mix[-1].bias)

    def forward(self, x, stats, heads):
        n, channels = x.shape
        width = channels // heads
        z = self.norm1(x)
        q = (F.elu(self.q(z)) + 1.0).view(n, width, heads)
        k = (F.elu(self.k(z)) + 1.0).view(n, width, heads)
        if not _NO_QKV_CONTEXT:
            scale = 1.0 + 0.25 * torch.tanh(self.context_scale(stats))
            q = q * scale[:, 0:1].unsqueeze(1)
            k = k * scale[:, 1:2].unsqueeze(1)
        v = self.v(z).view(n, width, heads)
        kv = torch.einsum('ndh,nmh->dmh', k, v)
        numerator = torch.einsum('ndh,dmh->nmh', q, kv)
        denominator = torch.einsum('ndh,dh->nh', q, k.sum(0)).unsqueeze(1)
        global_x = (numerator / denominator.clamp_min(1e-6)).reshape(n, channels)
        gate = torch.sigmoid(self.mix(stats))
        x = x + F.dropout(gate * global_x, p=self.dropout, training=self.training)
        x = x + F.dropout(self.ffn(self.norm2(x)), p=self.dropout,
                          training=self.training)
        return x


class RedesignedPolynormer(torch.nn.Module):
    """V37 完整版关系感知分支（main_v37.py 用的"新专家"）。

    结构：
      ContextInputEncoder(上下文注入)
        -> RelationAwareLocalBlock ×L（route + filter + 逐节点 β）
        -> depth_score 深度加权聚合各层状态        # no_depth_weight 消融关闭
        -> ContextualGlobalBlock ×G（QKV context） # no_qkv_context 消融关闭
        -> local_head/global_head + output_gate(σ(MLP(stats))) 输出门控
      relation_head + route_aux_loss：边级辅助损失（预测两节点同标签概率）
    """
    def __init__(self, in_channels, hidden_channels, out_channels,
                 local_layers=3, global_layers=2, in_dropout=0.15,
                 dropout=0.5, global_dropout=0.5, heads=1, beta=-1,
                 pre_ln=False):
        super().__init__()
        del pre_ln
        self._global = False
        self.in_dropout = in_dropout
        self.heads = heads
        channels = hidden_channels * heads
        self.input_encoder = ContextInputEncoder(in_channels, channels)
        self.local_blocks = torch.nn.ModuleList([
            RelationAwareLocalBlock(channels, hidden_channels, heads, beta, dropout)
            for _ in range(local_layers)])
        rank = min(32, channels)
        self.depth_score = torch.nn.Sequential(
            torch.nn.Linear(channels + 3, rank), torch.nn.SiLU(),
            torch.nn.Linear(rank, 1))
        self.global_blocks = torch.nn.ModuleList([
            ContextualGlobalBlock(channels, global_dropout)
            for _ in range(global_layers)])
        self.local_head = torch.nn.Sequential(
            torch.nn.LayerNorm(channels), torch.nn.Linear(channels, channels),
            torch.nn.SiLU(), torch.nn.Linear(channels, out_channels))
        self.global_head = torch.nn.Sequential(
            torch.nn.LayerNorm(channels), torch.nn.Linear(channels, channels),
            torch.nn.SiLU(), torch.nn.Linear(channels, out_channels))
        self.output_gate = torch.nn.Sequential(
            torch.nn.Linear(3, rank), torch.nn.SiLU(), torch.nn.Linear(rank, 1))
        self.relation_head = torch.nn.Sequential(
            torch.nn.Linear(2 * channels + 3, rank), torch.nn.SiLU(),
            torch.nn.Linear(rank, 1))

    def reset_parameters(self):
        self.input_encoder.reset_parameters()
        for block in self.local_blocks:
            block.reset_parameters()
        for block in self.global_blocks:
            block.reset_parameters()
        for module in (self.depth_score, self.local_head, self.global_head,
                       self.output_gate, self.relation_head):
            for layer in module:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        torch.nn.init.zeros_(self.depth_score[-1].weight)
        torch.nn.init.zeros_(self.depth_score[-1].bias)
        torch.nn.init.zeros_(self.output_gate[-1].weight)
        torch.nn.init.zeros_(self.output_gate[-1].bias)

    def encode_local(self, raw_x, edge_index):
        raw_x = F.dropout(raw_x, p=self.in_dropout, training=self.training)
        x, stats = self.input_encoder(raw_x, edge_index)
        states, scores = [], []
        for block in self.local_blocks:
            x = block(x, edge_index, stats)
            states.append(x)
            scores.append(self.depth_score(torch.cat((x, stats), dim=-1)))
        stacked = torch.stack(states, dim=1)
        if _NO_DEPTH_WEIGHT:
            local = stacked.sum(dim=1)
        else:
            weights = F.softmax(torch.stack(scores, dim=1), dim=1)
            local = (weights * stacked).sum(dim=1)
        return local, stats

    def forward(self, x, edge_index):
        local, stats = self.encode_local(x, edge_index)
        local_logits = self.local_head(local)
        if not self._global:
            return local_logits
        global_x = local
        for block in self.global_blocks:
            global_x = block(global_x, stats, self.heads)
        global_logits = self.global_head(global_x)
        gate = torch.sigmoid(self.output_gate(stats))
        return local_logits + gate * (global_logits - local_logits)

    def route_aux_loss(self, raw_x, edge_index, labels, train_idx):
        x, stats = self.input_encoder(raw_x, edge_index)
        _, src, dst = graph_context(x, edge_index)
        train_mask = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
        train_mask[train_idx] = True
        valid = train_mask[src] & train_mask[dst]
        if valid.sum() == 0:
            return x.sum() * 0.0
        src, dst = src[valid], dst[valid]
        target = (labels[src] == labels[dst]).float()
        pair = torch.cat((x[src], x[dst], stats[dst]), dim=-1)
        logits = self.relation_head(pair).squeeze(-1)
        pos = target.sum().clamp_min(1.0)
        neg = (1.0 - target).sum().clamp_min(1.0)
        weights = target * (0.5 / pos) + (1.0 - target) * (0.5 / neg)
        return F.binary_cross_entropy_with_logits(
            logits, target, weight=weights, reduction='sum')


class SafeRedesignedPolynormer(torch.nn.Module):
    """Official safety expert and fully redesigned graph-context expert."""
    def __init__(self, in_channels, hidden_channels, out_channels, **kwargs):
        super().__init__()
        self.official = OfficialPolynormer(
            in_channels, hidden_channels, out_channels, **kwargs)
        self.redesigned = RedesignedPolynormer(
            in_channels, hidden_channels, out_channels, **kwargs)

    @property
    def _global(self):
        return self.official._global

    @_global.setter
    def _global(self, value):
        self.official._global = value
        self.redesigned._global = value

    def reset_parameters(self):
        self.official.reset_parameters()
        self.redesigned.reset_parameters()

    def forward(self, x, edge_index):
        return self.official(x, edge_index), self.redesigned(x, edge_index)

    def route_aux_loss(self, x, edge_index, labels, train_idx):
        return self.redesigned.route_aux_loss(
            x, edge_index, labels, train_idx)
