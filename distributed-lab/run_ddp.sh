#!/usr/bin/env bash
# 项目 2 · 阶段 1：在多卡 GPU 机器上跑数据并行实验，结果追加到 results_ddp.csv。
#
# 用法：
#   bash run_ddp.sh                 # 自动检测有几张显卡
#   NGPU=2 bash run_ddp.sh          # 只用前 2 张卡
#   BATCH=8 bash run_ddp.sh         # 每张卡的 batch（默认 16，即项目 0 实验 7 的最佳单卡配置）
# 跑完后执行：python report_ddp.py

NGPU=${NGPU:-$(nvidia-smi -L | wc -l)}
BATCH=${BATCH:-16}
STEPS=${STEPS:-50}
RESULTS=${RESULTS:-results_ddp.csv}

# 和项目 0 实验 7 相同的单卡配置：打开全部单卡优化
OPTS="--model gpt2 --data shakespeare --steps $STEPS --skip_steps 10 --batch_size $BATCH \
--tf32 --bf16 --compile --attn sdpa --vocab_size 50304 --adamw fused --results $RESULTS"

PORT=29500
run() {
  local n=$1 impl=$2
  local tag="${n}gpu_${impl}"
  PORT=$((PORT + 1))  # 每次换一个端口，避免上一个实验的端口还没释放
  echo
  echo "################ $tag ################"
  torchrun --nproc_per_node="$n" --master_addr=127.0.0.1 --master_port="$PORT" \
    train_ddp.py $OPTS --ddp_impl "$impl" --tag "$tag" || echo "[$tag] 运行失败，继续下一个实验"
}

# 单卡 baseline：用来计算扩展效率
run 1 torch

# 多卡：三种梯度同步方式对比
for n in 2 4 8; do
  [ "$n" -le "$NGPU" ] || continue
  for impl in naive flat torch; do
    run "$n" "$impl"
  done
done

echo
python report_ddp.py "$RESULTS"
