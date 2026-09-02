"""main_v37_slim.py —— 双专家训练主脚本（官方 + slim 分支 / SlimFinal，alpha 融合）。

与 main_v37.py 同协议，差异在新分支可切换：
  --slim_backbone final → 官方 + SlimFinal（本文主实验 duo_final）
  --slim_backbone slim  → 官方 + 旧 slim 分支
  --fuse_mode gate      → 端到端可学习门控（交替更新防 OOM/趋同，已证明不如 alpha）
注意：main_v37_slim.py 的官方/新分支在 local 阶段独立训练并各自保存 ckpt，
global 阶段 reload 后继续；alpha 每 epoch 在验证集网格选择。
"""
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops

from logger import *
from dataset import load_dataset
from data_utils import eval_acc, eval_rocauc, load_fixed_splits
from eval import *
from parse_v37_slim import parse_method, parser_add_main_args


def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

### Parse args ###
parser = argparse.ArgumentParser(description='V37 slim dual-branch Polynormer')
parser_add_main_args(parser)
args = parser.parse_args()
if not args.global_dropout:
    args.global_dropout = args.dropout
print(args)
fusion_grid = [float(v) for v in args.fusion_grid.split(',')]
assert fusion_grid and all(0.0 <= v <= 1.0 for v in fusion_grid)
assert 0.0 in fusion_grid, 'fusion grid must contain official fallback alpha=0'
use_gate = getattr(args, 'fuse_mode', 'alpha') == 'gate'

fix_seed(args.seed)

if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

### Load and preprocess data ###
dataset = load_dataset(args.data_dir, args.dataset)

if len(dataset.label.shape) == 1:
    dataset.label = dataset.label.unsqueeze(1)
dataset.label = dataset.label.to(device)

split_idx_lst = load_fixed_splits(args.data_dir, dataset, name=args.dataset)

### Basic information of datasets ###
n = dataset.graph['num_nodes']
e = dataset.graph['edge_index'].shape[1]
c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
d = dataset.graph['node_feat'].shape[1]

print(f"dataset {args.dataset} | num nodes {n} | num edge {e} | num node feats {d} | num classes {c}")

dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
dataset.graph['edge_index'], _ = remove_self_loops(dataset.graph['edge_index'])
dataset.graph['edge_index'], _ = add_self_loops(dataset.graph['edge_index'], num_nodes=n)

dataset.graph['edge_index'], dataset.graph['node_feat'] = \
    dataset.graph['edge_index'].to(device), dataset.graph['node_feat'].to(device)

### Load method ###
model = parse_method(args, n, c, d, device)

### Loss function (Single-class, Multi-class) ###
if args.dataset in ('questions'):
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.NLLLoss()

### Performance metric (Acc, AUC) ###
if args.metric == 'rocauc':
    eval_func = eval_rocauc
else:
    eval_func = eval_acc

logger = Logger(args.runs, args)
run_summaries = []

model.train()
print('MODEL:', model)

### Training loop ###
for run in range(args.runs):
    run_seed = args.seed + run
    fix_seed(run_seed)
    print(f'RUN_SEED: {run_seed}')
    if args.dataset in ('coauthor-cs', 'coauthor-physics', 'amazon-computer', 'amazon-photo'):
        split_idx = split_idx_lst[0]
    else:
        if args.split_base is not None:
            split_idx = split_idx_lst[run_seed - args.split_base]
        else:
            split_idx = split_idx_lst[run]
    print(f'SPLIT: train={split_idx["train"].numel()} val={split_idx["valid"].numel()} test={split_idx["test"].numel()}')
    train_idx = split_idx['train'].to(device)
    model.reset_parameters()
    model._global = False
    base_optimizer = torch.optim.Adam(
        model.official.parameters(), weight_decay=args.weight_decay, lr=args.lr)
    new_params = list(model.redesigned.parameters())
    if use_gate:
        new_params += list(model.fuse_mlp.parameters())
    new_optimizer = torch.optim.Adam(
        new_params, weight_decay=args.weight_decay, lr=args.lr)
    os.makedirs(f'models/{args.dataset}', exist_ok=True)
    stem = f'models/{args.dataset}/{args.method}_{run}_{args.beta}'
    base_local_path = stem + '.official_local.pt'
    new_local_path = stem + '.redesigned_local.pt'
    combined_path = stem + '.pt'
    best_local_base = float('-inf')
    best_local_new = float('-inf')
    best_val = float('-inf')
    best_test = float('-inf')
    best_official_test = float('-inf')
    best_pure_test = float('-inf')
    best_alpha = 0.0
    for epoch in range(args.local_epochs+args.global_epochs):
        if epoch == args.local_epochs:
            print("start global attention!!!!!!")
            base_ckpt = torch.load(base_local_path, map_location=device)
            new_ckpt = torch.load(new_local_path, map_location=device)
            model.official.load_state_dict(base_ckpt['model'])
            base_optimizer.load_state_dict(base_ckpt['optimizer'])
            model.redesigned.load_state_dict(new_ckpt['model'])
            new_optimizer.load_state_dict(new_ckpt['optimizer'])
            model._global = True
        model.train()
        if use_gate and epoch >= args.local_epochs:
            # 端到端可学习融合（交替更新，一次仅保留一个分支的计算图，省显存+防趋同）
            node_feat = dataset.graph['node_feat']
            edge_index = dataset.graph['edge_index']
            # 子步骤1：更新官方分支（slim 冻结）
            with torch.no_grad():
                out_r, stats = model.redesigned.forward_with_stats(
                    node_feat, edge_index)
            out_o = model.official(node_feat, edge_index)
            delta = out_o - out_r
            g = torch.sigmoid(model.fuse_mlp(torch.cat([stats, delta], dim=-1)))
            fused = (1.0 - g) * out_o + g * out_r
            if args.dataset in ('questions'):
                if dataset.label.shape[1] == 1:
                    true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
                else:
                    true_label = dataset.label
                target = true_label.squeeze(1)[train_idx].to(torch.float)
                loss_value = criterion(fused[train_idx], target)
            else:
                target = dataset.label.squeeze(1)[train_idx]
                loss_value = criterion(
                    F.log_softmax(fused, dim=1)[train_idx], target)
            base_optimizer.zero_grad()
            loss_value.backward()
            base_optimizer.step()
            del out_o, out_r, stats, delta, g, fused
            # 子步骤2：更新 slim + 融合门控（official 冻结）
            with torch.no_grad():
                out_o = model.official(node_feat, edge_index)
            out_r, stats = model.redesigned.forward_with_stats(
                node_feat, edge_index)
            delta = out_o - out_r
            g = torch.sigmoid(model.fuse_mlp(torch.cat([stats, delta], dim=-1)))
            fused = (1.0 - g) * out_o + g * out_r
            if args.dataset in ('questions'):
                loss_value = criterion(fused[train_idx], target)
            else:
                loss_value = criterion(
                    F.log_softmax(fused, dim=1)[train_idx], target)
            new_optimizer.zero_grad()
            loss_value.backward()
            new_optimizer.step()
            del out_o, out_r, stats, delta, g, fused
        else:
            base_optimizer.zero_grad()

            out_base = model.official(
                dataset.graph['node_feat'], dataset.graph['edge_index'])
            if args.dataset in ('questions'):
                if dataset.label.shape[1] == 1:
                    true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
                else:
                    true_label = dataset.label
                target = true_label.squeeze(1)[train_idx].to(torch.float)
                base_loss = criterion(out_base[train_idx], target)
            else:
                target = dataset.label.squeeze(1)[train_idx]
                base_loss = criterion(
                    F.log_softmax(out_base, dim=1)[train_idx], target)
            base_loss.backward()
            base_optimizer.step()
            loss_value = base_loss.detach()
            del out_base, base_loss

            new_optimizer.zero_grad()
            out_struct = model.redesigned(
                dataset.graph['node_feat'], dataset.graph['edge_index'])
            if args.dataset in ('questions'):
                struct_loss = criterion(out_struct[train_idx], target)
            else:
                struct_loss = criterion(
                    F.log_softmax(out_struct, dim=1)[train_idx], target)
            if epoch < args.local_epochs and args.route_lambda > 0:
                aux_loss = model.route_aux_loss(
                    dataset.graph['node_feat'], dataset.graph['edge_index'],
                    dataset.label.squeeze(1), train_idx)
                struct_loss = struct_loss + args.route_lambda * aux_loss
            struct_loss.backward()
            new_optimizer.step()
            loss_value = loss_value + struct_loss.detach()
            del out_struct, struct_loss

        model.eval()
        with torch.no_grad():
            if use_gate and epoch >= args.local_epochs:
                fused, _ = model.fused_logits(
                    dataset.graph['node_feat'], dataset.graph['edge_index'])
                result = evaluate(
                    model, dataset, split_idx, eval_func, criterion, args,
                    result=fused)
                selected_alpha = float('nan')
            else:
                eval_base = model.official(
                    dataset.graph['node_feat'], dataset.graph['edge_index'])
                eval_struct = model.redesigned(
                    dataset.graph['node_feat'], dataset.graph['edge_index'])
                candidates = []
                for alpha in fusion_grid:
                    fused = (1.0 - alpha) * eval_base + alpha * eval_struct
                    candidate = evaluate(
                        model, dataset, split_idx, eval_func, criterion, args,
                        result=fused)
                    candidates.append((candidate[1], alpha, candidate))
                _, selected_alpha, result = max(candidates, key=lambda item: item[0])

        if epoch < args.local_epochs:
            base_valid = candidates[0][0]
            new_valid = candidates[-1][0]
            if base_valid > best_local_base:
                best_local_base = base_valid
                torch.save({'model': model.official.state_dict(),
                            'optimizer': base_optimizer.state_dict()}, base_local_path)
            if new_valid > best_local_new:
                best_local_new = new_valid
                torch.save({'model': model.redesigned.state_dict(),
                            'optimizer': new_optimizer.state_dict()}, new_local_path)

        logger.add_result(run, result[:-1])

        if result[1] > best_val:
            best_val = result[1]
            best_test = result[2]
            if not (use_gate and epoch >= args.local_epochs):
                best_official_test = candidates[0][2][2]
                best_pure_test = candidates[-1][2][2]
            best_alpha = selected_alpha
            if args.save_model:
                torch.save({'model_state_dict': model.state_dict(),
                            'official_optimizer': base_optimizer.state_dict(),
                            'redesigned_optimizer': new_optimizer.state_dict(),
                            'alpha': selected_alpha}, combined_path)

        if epoch % args.display_step == 0:
            print(f'Epoch: {epoch:02d}, '
                  f'Loss: {loss_value:.4f}, '
                  f'Train: {100 * result[0]:.2f}%, '
                  f'Valid: {100 * result[1]:.2f}%, '
                  f'Test: {100 * result[2]:.2f}%, '
                  f'Best Valid: {100 * best_val:.2f}%, '
                  f'Best Test: {100 * best_test:.2f}%, '
                  f'Alpha: {selected_alpha:.2f}')
    logger.print_statistics(run)
    run_summaries.append({
        'run': run, 'seed': run_seed,
        'fused': 100.0 * best_test,
        'official': 100.0 * best_official_test,
        'pure': 100.0 * best_pure_test,
        'alpha': best_alpha,
    })
    print(f'RUN_SUMMARY run={run} seed={run_seed} '
          f'fused={100.0 * best_test:.4f} '
          f'official={100.0 * best_official_test:.4f} '
          f'pure={100.0 * best_pure_test:.4f} '
          f'alpha={best_alpha:.4f}')

results = logger.print_statistics()
### Save results ###
save_result(args, results)

### Save three-branch + alpha summary ###
summary_path = f'results/{args.dataset}/{args.method}_3branch.csv'
os.makedirs(f'results/{args.dataset}', exist_ok=True)
with open(summary_path, 'w') as f:
    f.write('run,seed,fused,official,pure,alpha\n')
    for s in run_summaries:
        f.write(f"{s['run']},{s['seed']},{s['fused']:.4f},"
                f"{s['official']:.4f},{s['pure']:.4f},{s['alpha']:.4f}\n")
    fused_vals = [s['fused'] for s in run_summaries]
    official_vals = [s['official'] for s in run_summaries]
    pure_vals = [s['pure'] for s in run_summaries]
    alpha_vals = [s['alpha'] for s in run_summaries]
    f.write(f"MEAN,{args.seed},{np.mean(fused_vals):.4f},{np.mean(official_vals):.4f},"
            f"{np.mean(pure_vals):.4f},{np.mean(alpha_vals):.4f}\n")
    f.write(f"STD,{args.seed},{np.std(fused_vals):.4f},{np.std(official_vals):.4f},"
            f"{np.std(pure_vals):.4f},{np.std(alpha_vals):.4f}\n")
