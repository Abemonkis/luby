import torch
import torch.nn.functional as F

from model_official_v35 import Polynormer as OfficialPolynormer, GlobalAttn
from model_v37 import (graph_context, RelationAwareLocalBlock,
                       ContextInputEncoder, ContextualGlobalBlock)


class SimpleInputEncoder(torch.nn.Module):
    """特征线性投影 + LayerNorm；只计算 graph context stats，不做上下文注入。"""

    def __init__(self, in_channels, channels):
        super().__init__()
        self.feature = torch.nn.Linear(in_channels, channels)
        self.norm = torch.nn.LayerNorm(channels)

    def reset_parameters(self):
        self.feature.reset_parameters()
        self.norm.reset_parameters()

    def forward(self, x, edge_index):
        x = self.feature(x)
        stats, _, _ = graph_context(x, edge_index)
        x = self.norm(x)
        return x, stats


class RedesignedPolynormerSlim(torch.nn.Module):
    """瘦身版 redesigned 分支。

    核心保留：路由 + 差异滤波 + 逐节点 Beta。
    默认删除：深度门控、QKV 调制、复杂全局块、输出门控、辅助损失、上下文注入。

    诊断开关（add back）：
      use_context_inject  加回输入层上下文注入
      use_aux             加回辅助损失（relation_head + route_aux_loss）
      use_global_context  加回复杂全局块(mix+ffn) + 输出门控
    """

    def __init__(self, in_channels, hidden_channels, out_channels,
                 local_layers=3, global_layers=2, in_dropout=0.15,
                 dropout=0.5, global_dropout=0.5, heads=1, beta=-1,
                 pre_ln=False, use_context_inject=False, use_aux=False,
                 use_global_context=False):
        super().__init__()
        del pre_ln
        self._global = False
        self.in_dropout = in_dropout
        self.heads = heads
        self.use_aux = use_aux
        self.use_global_context = use_global_context
        channels = hidden_channels * heads
        rank = min(32, channels)

        if use_context_inject:
            self.input_encoder = ContextInputEncoder(in_channels, channels)
        else:
            self.input_encoder = SimpleInputEncoder(in_channels, channels)

        self.local_blocks = torch.nn.ModuleList([
            RelationAwareLocalBlock(channels, hidden_channels, heads, beta, dropout)
            for _ in range(local_layers)])

        if use_global_context:
            self.global_blocks = torch.nn.ModuleList([
                ContextualGlobalBlock(channels, global_dropout)
                for _ in range(global_layers)])
        else:
            self.global_attn = GlobalAttn(hidden_channels, heads, global_layers,
                                          beta, global_dropout)

        self.ln = torch.nn.LayerNorm(channels)
        self.local_head = torch.nn.Sequential(
            torch.nn.LayerNorm(channels), torch.nn.Linear(channels, channels),
            torch.nn.SiLU(), torch.nn.Linear(channels, out_channels))
        self.global_head = torch.nn.Sequential(
            torch.nn.LayerNorm(channels), torch.nn.Linear(channels, channels),
            torch.nn.SiLU(), torch.nn.Linear(channels, out_channels))

        if use_global_context:
            self.output_gate = torch.nn.Sequential(
                torch.nn.Linear(3, rank), torch.nn.SiLU(), torch.nn.Linear(rank, 1))
        if use_aux:
            self.relation_head = torch.nn.Sequential(
                torch.nn.Linear(2 * channels + 3, rank), torch.nn.SiLU(),
                torch.nn.Linear(rank, 1))

    def reset_parameters(self):
        self.input_encoder.reset_parameters()
        for block in self.local_blocks:
            block.reset_parameters()
        if self.use_global_context:
            for block in self.global_blocks:
                block.reset_parameters()
        else:
            self.global_attn.reset_parameters()
        self.ln.reset_parameters()
        for module in (self.local_head, self.global_head):
            for layer in module:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        if self.use_global_context:
            for layer in self.output_gate:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
            torch.nn.init.zeros_(self.output_gate[-1].weight)
            torch.nn.init.zeros_(self.output_gate[-1].bias)
        if self.use_aux:
            for layer in self.relation_head:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

    def encode_local(self, raw_x, edge_index):
        raw_x = F.dropout(raw_x, p=self.in_dropout, training=self.training)
        x, stats = self.input_encoder(raw_x, edge_index)
        states = []
        for block in self.local_blocks:
            x = block(x, edge_index, stats)
            states.append(x)
        local = torch.stack(states, dim=0).sum(dim=0)
        return local, stats

    def forward(self, x, edge_index):
        local, stats = self.encode_local(x, edge_index)
        local_logits = self.local_head(local)
        if not self._global:
            return local_logits
        if self.use_global_context:
            global_x = local
            for block in self.global_blocks:
                global_x = block(global_x, stats, self.heads)
            global_logits = self.global_head(global_x)
            gate = torch.sigmoid(self.output_gate(stats))
            return local_logits + gate * (global_logits - local_logits)
        global_x = self.global_attn(self.ln(local))
        return self.global_head(global_x)

    def forward_with_stats(self, x, edge_index):
        """与 forward 等价，但额外返回 stats（供端到端融合门控使用）。"""
        local, stats = self.encode_local(x, edge_index)
        local_logits = self.local_head(local)
        if not self._global:
            return local_logits, stats
        if self.use_global_context:
            global_x = local
            for block in self.global_blocks:
                global_x = block(global_x, stats, self.heads)
            global_logits = self.global_head(global_x)
            gate = torch.sigmoid(self.output_gate(stats))
            return local_logits + gate * (global_logits - local_logits), stats
        global_x = self.global_attn(self.ln(local))
        return self.global_head(global_x), stats

    def route_aux_loss(self, raw_x, edge_index, labels, train_idx):
        if not self.use_aux:
            return torch.zeros((), device=raw_x.device, requires_grad=True)
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


class SafeRedesignedPolynormerSlim(torch.nn.Module):
    """官方分支 + 瘦身版 redesigned 分支。

    fuse_mode='alpha'：验证集网格搜 α（原行为）。
    fuse_mode='gate'：端到端可学习逐节点融合门控（global 阶段启用）。
    """

    def __init__(self, in_channels, hidden_channels, out_channels, **kwargs):
        super().__init__()
        fuse_mode = kwargs.pop('fuse_mode', 'alpha')
        slim_backbone = kwargs.pop('slim_backbone', 'slim')
        self.fuse_mode = fuse_mode
        self.slim_backbone = slim_backbone
        official_kwargs = {k: v for k, v in kwargs.items()
                           if k not in ('use_context_inject', 'use_aux',
                                        'use_global_context')}
        self.official = OfficialPolynormer(
            in_channels, hidden_channels, out_channels, **official_kwargs)
        if slim_backbone == 'final':
            from model_slim_final import SlimFinal
            self.redesigned = SlimFinal(
                in_channels, hidden_channels, out_channels, **kwargs)
        else:
            self.redesigned = RedesignedPolynormerSlim(
                in_channels, hidden_channels, out_channels, **kwargs)
        if fuse_mode == 'gate':
            rank = min(32, 3 + out_channels)
            # 输入：[stats(3), 两分支 logits 差 Δ(out_channels)] → 逐节点门控 g
            self.fuse_mlp = torch.nn.Sequential(
                torch.nn.Linear(3 + out_channels, rank), torch.nn.SiLU(),
                torch.nn.Linear(rank, 1))

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
        if self.fuse_mode == 'gate':
            for layer in self.fuse_mlp:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
            # 零初始化末层 → 初始 g=sigmoid(0)=0.5（两分支均分）
            torch.nn.init.zeros_(self.fuse_mlp[-1].weight)
            torch.nn.init.zeros_(self.fuse_mlp[-1].bias)

    def forward(self, x, edge_index):
        return self.official(x, edge_index), self.redesigned(x, edge_index)

    def fused_logits(self, x, edge_index):
        """端到端可学习融合：g = σ(MLP([stats, Δ]))，逐节点软插值。"""
        logits_o = self.official(x, edge_index)
        logits_r, stats = self.redesigned.forward_with_stats(x, edge_index)
        delta = logits_o - logits_r
        g = torch.sigmoid(self.fuse_mlp(torch.cat([stats, delta], dim=-1)))
        return (1.0 - g) * logits_o + g * logits_r, g

    def route_aux_loss(self, x, edge_index, labels, train_idx):
        return self.redesigned.route_aux_loss(x, edge_index, labels, train_idx)
