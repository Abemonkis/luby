#!/bin/bash
set -u
source /home/lby/miniconda3/etc/profile.d/conda.sh
conda activate polynormer_official
cd /home/lby/Polynormer-r
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

run_case() {
  label="$1"; shift
  echo "=== START ${label} $(date -Is) ==="
  python main_v37_slim.py "$@"
  status=$?
  echo "=== END ${label} status=${status} $(date -Is) ==="
  return 0
}

# 双专家（官方 + SlimFinal）验证集 α 融合：9 数据集 x 5 seed（42-46）
# 新分支为清理后的最佳 slim 分支（删 route，保留差异滤波+逐节点 β）
# 超参对齐 v37_single_full（官方 Polynormer 协议）
COMMON="--data_dir ./data_official_pyg23/ --seed 42 --device 0 --fusion_grid 0,0.25,0.5,0.75,1 --runs 5 --slim_backbone final"

run_case roman     $COMMON --method v37_duo_final_full --dataset roman-empire    --hidden_channels 64  --local_epochs 100 --global_epochs 2500 --lr 0.001 --local_layers 10 --global_layers 2 --weight_decay 0    --dropout 0.3 --global_dropout 0.5 --in_dropout 0.15 --num_heads 8 --beta 0.5
run_case ratings   $COMMON --method v37_duo_final_full --dataset amazon-ratings  --hidden_channels 256 --local_epochs 200 --global_epochs 2500 --lr 0.001 --local_layers 10 --global_layers 1 --weight_decay 0    --dropout 0.3 --in_dropout 0.2 --num_heads 2
run_case mine      $COMMON --method v37_duo_final_full --dataset minesweeper     --hidden_channels 64  --local_epochs 100 --global_epochs 2000 --lr 0.001 --local_layers 10 --global_layers 3 --weight_decay 0    --dropout 0.3 --in_dropout 0.2 --num_heads 8 --metric rocauc
run_case questions $COMMON --method v37_duo_final_full --dataset questions       --hidden_channels 64  --local_epochs 200 --global_epochs 1500 --lr 3e-5  --local_layers 5  --global_layers 3 --weight_decay 0    --dropout 0.2 --global_dropout 0.5 --num_heads 8 --metric rocauc --in_dropout 0.15 --beta 0.4 --pre_ln
run_case computer  $COMMON --method v37_duo_final_full --dataset amazon-computer --hidden_channels 64  --local_epochs 200 --global_epochs 1000 --lr 0.001 --local_layers 5  --global_layers 1 --weight_decay 5e-5 --dropout 0.7 --in_dropout 0.2 --num_heads 8
run_case photo     $COMMON --method v37_duo_final_full --dataset amazon-photo    --hidden_channels 64  --local_epochs 200 --global_epochs 1000 --lr 0.001 --local_layers 7  --global_layers 2 --weight_decay 5e-5 --dropout 0.7 --in_dropout 0.2 --num_heads 8
run_case cs        $COMMON --method v37_duo_final_full --dataset coauthor-cs     --hidden_channels 64  --local_epochs 100 --global_epochs 1500 --lr 0.001 --local_layers 5  --global_layers 2 --weight_decay 5e-4 --dropout 0.3 --in_dropout 0.1 --num_heads 8
run_case physics   $COMMON --method v37_duo_final_full --dataset coauthor-physics --hidden_channels 32  --local_epochs 100 --global_epochs 1500 --lr 0.001 --local_layers 5  --global_layers 4 --weight_decay 5e-4 --dropout 0.5 --in_dropout 0.1 --num_heads 8
run_case wikics    $COMMON --method v37_duo_final_full --dataset wikics          --hidden_channels 512 --local_epochs 100 --global_epochs 1000 --lr 0.001 --local_layers 7  --global_layers 2 --weight_decay 0    --dropout 0.5 --in_dropout 0.5 --num_heads 1

echo "=== V37 DUO FINAL FULL ALL COMPLETE $(date -Is) ==="
