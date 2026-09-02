"""
model_slim_final.py —— SlimFinal：清理后的最佳 slim 分支（官方+SlimFinal 双专家的新专家）

演进链条（正交消融逐步砍冗余，fused 性能不变、参数递减）：
  V37 完整分支 (model_v37.RedesignedPolynormer)
    → slim 分支   (model_v37_slim.RedesignedPolynormerSlim，删 aux/depth/QKV/输出门控)
    → SlimFinal   (本文件，再删 route：差异注入固定 π=0.5)
保留的有效组件（均有消融证据）：上下文注入、差异滤波、逐节点 β、全局块+输出门控。
本模型被两条路径使用：
  · 单模型：main_slim_final.py / parse_slim_final.py
  · 双专家：main_v37_slim.py / parse_v37_slim.py，--slim_backbone final 时
            SafeRedesignedPolynormerSlim 用本文件的 SlimFinal 作为新分支
"""
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from model_v37 import graph_context, ContextInputEncoder, ContextualGlobalBlock


class FinalLocalBlock(torch.nn.Module):
    """清理版局部块：GAT + 残差 + 差异滤波（固定 π=0.5 注入）+ 逐节点 β + 多项式。

    已删除：路由 MLP（消融证明冗余）。
    保留：差异滤波（minesweeper +0.79）、逐节点 β（minesweeper +0.35）。
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
        for module in (self.filter, self.beta_context):
            for layer in module:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        torch.nn.init.zeros_(self.beta_context[-1].weight)
        torch.nn.init.zeros_(self.beta_context[-1].bias)
        if self.beta_mode < 0:
            torch.nn.init.zeros_(self.beta_base)
        else:
            torch.nn.init.constant_(self.beta_base, self.beta_mode)

    def forward(self, x, edge_index, stats):
        z = self.pre_norm(x)
        message = self.gat(z, edge_index)
        residual = self.residual(x)
        discrepancy = self.filter(residual - message)
        # 固定 π=0.5（路由已删，消融证明均匀路由不损性能）
        update = message + residual + 0.5 * discrepancy
        update = F.silu(update)
        update = F.dropout(update, p=self.dropout, training=self.training)
        delta = 0.2 * torch.tanh(self.beta_context(stats))
        if self.beta_mode < 0:
            beta = torch.sigmoid(self.beta_base.unsqueeze(0) + delta)
        else:
            beta = (self.beta_base.unsqueeze(0) + delta).clamp(0.0, 1.0)
        interaction = self.post_norm(F.silu(residual) * update)
        return (1.0 - beta) * interaction + beta * update


class SlimFinal(torch.nn.Module):
    """清理后的最佳 slim 分支。

    输入：ContextInputEncoder（上下文注入，questions +1.72）
    局部：FinalLocalBlock ×L（无路由，保留差异滤波+逐节点β+多项式）
    全局：ContextualGlobalBlock ×G + 输出门控（questions +2.64）
    """

    def __init__(self, in_channels, hidden_channels, out_channels,
                 local_layers=3, global_layers=2, in_dropout=0.15,
                 dropout=0.5, global_dropout=0.5, heads=1, beta=-1,
                 **kwargs):
        super().__init__()
        del kwargs
        self._global = False
        self.in_dropout = in_dropout
        self.heads = heads
        channels = hidden_channels * heads
        rank = min(32, channels)

        self.input_encoder = ContextInputEncoder(in_channels, channels)
        self.local_blocks = torch.nn.ModuleList([
            FinalLocalBlock(channels, hidden_channels, heads, beta, dropout)
            for _ in range(local_layers)])
        self.global_blocks = torch.nn.ModuleList([
            ContextualGlobalBlock(channels, global_dropout)
            for _ in range(global_layers)])
        self.ln = torch.nn.LayerNorm(channels)
        self.local_head = torch.nn.Sequential(
            torch.nn.LayerNorm(channels), torch.nn.Linear(channels, channels),
            torch.nn.SiLU(), torch.nn.Linear(channels, out_channels))
        self.global_head = torch.nn.Sequential(
            torch.nn.LayerNorm(channels), torch.nn.Linear(channels, channels),
            torch.nn.SiLU(), torch.nn.Linear(channels, out_channels))
        self.output_gate = torch.nn.Sequential(
            torch.nn.Linear(3, rank), torch.nn.SiLU(),
            torch.nn.Linear(rank, 1))

    def reset_parameters(self):
        self.input_encoder.reset_parameters()
        for block in self.local_blocks:
            block.reset_parameters()
        for block in self.global_blocks:
            block.reset_parameters()
        self.ln.reset_parameters()
        for module in (self.local_head, self.global_head):
            for layer in module:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        for layer in self.output_gate:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        torch.nn.init.zeros_(self.output_gate[-1].weight)
        torch.nn.init.zeros_(self.output_gate[-1].bias)

    def _encode(self, x, edge_index):
        x = F.dropout(x, p=self.in_dropout, training=self.training)
        x, stats = self.input_encoder(x, edge_index)
        states = []
        for block in self.local_blocks:
            x = block(x, edge_index, stats)
            states.append(x)
        local = torch.stack(states, dim=0).sum(dim=0)
        return local, stats

    def _logits(self, local, stats):
        local_logits = self.local_head(local)
        if not self._global:
            return local_logits
        global_x = local
        for block in self.global_blocks:
            global_x = block(global_x, stats, self.heads)
        global_logits = self.global_head(global_x)
        gate = torch.sigmoid(self.output_gate(stats))
        return local_logits + gate * (global_logits - local_logits)

    def forward(self, x, edge_index):
        local, stats = self._encode(x, edge_index)
        return self._logits(local, stats)

    def forward_with_stats(self, x, edge_index):
        """与 forward 等价，但额外返回 stats（供端到端融合门控使用）。"""
        local, stats = self._encode(x, edge_index)
        return self._logits(local, stats), stats

    def route_aux_loss(self, x, edge_index, labels, train_idx):
        # SlimFinal 已删除路由辅助损失，返回零梯度占位保持双专家接口一致。
        return torch.zeros((), device=x.device, requires_grad=True)
