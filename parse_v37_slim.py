"""parse_v37_slim.py —— 双专家（官方 + 新分支）main_v37_slim.py 的参数解析与模型构造。

新分支两个候选：
  · slim_backbone=slim  （默认）旧 slim 分支 RedesignedPolynormerSlim
  · slim_backbone=final 清理版 SlimFinal（推荐，官方+SlimFinal 主实验用）
融合方式 fuse_mode：
  · alpha（默认）：验证集网格搜 α，fused=(1-α)·official + α·slim
  · gate：端到端可学习逐节点门控（已证明不如 alpha，保留作对比）
"""
from parse_v37 import parser_add_main_args as _parser_add_main_args
from model_v37_slim import SafeRedesignedPolynormerSlim


def parser_add_main_args(parser):
    _parser_add_main_args(parser)
    parser.add_argument('--use_context_inject', action='store_true',
                        help='diagnostic: add back input context injection')
    parser.add_argument('--use_aux', action='store_true',
                        help='diagnostic: add back route aux loss')
    parser.add_argument('--use_global_context', action='store_true',
                        help='diagnostic: add back complex global block + output gate')
    parser.add_argument('--fuse_mode', type=str, default='alpha',
                        choices=['alpha', 'gate'],
                        help="'alpha': val-set grid search (default); "
                             "'gate': end-to-end learnable per-node fusion gate")
    parser.add_argument('--slim_backbone', type=str, default='slim',
                        choices=['slim', 'final'],
                        help="'slim': original redesigned branch (route MLP); "
                             "'final': cleaned SlimFinal (no route, fixed pi=0.5)")


def parse_method(args, n, c, d, device):
    model = SafeRedesignedPolynormerSlim(
        d, args.hidden_channels, c,
        local_layers=args.local_layers, global_layers=args.global_layers,
        in_dropout=args.in_dropout, dropout=args.dropout,
        global_dropout=args.global_dropout, heads=args.num_heads,
        beta=args.beta, pre_ln=args.pre_ln,
        use_context_inject=args.use_context_inject,
        use_aux=args.use_aux,
        use_global_context=args.use_global_context,
        fuse_mode=args.fuse_mode,
        slim_backbone=args.slim_backbone).to(device)
    return model
