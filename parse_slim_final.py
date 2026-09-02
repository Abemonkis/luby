"""parse_slim_final.py —— SlimFinal 单模型（main_slim_final.py）的参数解析与模型构造。

超参完全继承官方 Polynormer 协议（parse_v37），只把模型换成 SlimFinal。
SlimFinal 固定包含上下文注入 + 全局块/输出门控（无可开关），故无需额外 flag。
"""
from parse_v37 import parser_add_main_args as _parser_add_main_args
from model_slim_final import SlimFinal


def parser_add_main_args(parser):
    _parser_add_main_args(parser)


def parse_method(args, n, c, d, device):
    model = SlimFinal(
        d, args.hidden_channels, c,
        local_layers=args.local_layers, global_layers=args.global_layers,
        in_dropout=args.in_dropout, dropout=args.dropout,
        global_dropout=args.global_dropout, heads=args.num_heads,
        beta=args.beta).to(device)
    return model
