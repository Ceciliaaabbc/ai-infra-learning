#!/usr/bin/env bash
# 优化阶梯：每一步在上一步的基础上多加一项优化，结果全部追加到 results.csv。
#
# 用法：
#   bash run_experiments.sh                          # 默认 gpt2 (124M)，B=4，T=1024，适合 24GB 显卡
#   BATCH=8 BIG_BATCH=32 bash run_experiments.sh     # A100 80G 等大显存卡可以调大
#   DATA=synthetic bash run_experiments.sh           # 不下载数据，只测速度
# 跑完后执行：python report.py

MODEL=${MODEL:-gpt2}
DATA=${DATA:-shakespeare}
BATCH=${BATCH:-4}
BIG_BATCH=${BIG_BATCH:-16}
STEPS=${STEPS:-50}
RESULTS=${RESULTS:-results.csv}

COMMON="--model $MODEL --data $DATA --steps $STEPS --skip_steps 10 --results $RESULTS"
OPTS="--tf32 --bf16 --compile --attn sdpa --vocab_size 50304 --adamw fused"

run() {
  local tag=$1
  shift
  echo
  echo "################ $tag ################"
  python train.py $COMMON --tag "$tag" "$@" || echo "[$tag] 运行失败（可能是 OOM），继续下一个实验"
}

# 第一组：batch 不变，逐项打开优化
run 0_baseline_fp32 --batch_size $BATCH
run 1_tf32          --batch_size $BATCH --tf32
run 2_bf16          --batch_size $BATCH --tf32 --bf16
run 3_compile       --batch_size $BATCH --tf32 --bf16 --compile
run 4_flash_attn    --batch_size $BATCH --tf32 --bf16 --compile --attn sdpa
run 5_vocab_50304   --batch_size $BATCH --tf32 --bf16 --compile --attn sdpa --vocab_size 50304
run 6_fused_adamw   --batch_size $BATCH $OPTS

# 第二组：显存和速度的取舍
run 7_big_batch            --batch_size $BIG_BATCH $OPTS
run 8_big_batch_ckpt       --batch_size $BIG_BATCH $OPTS --act_ckpt
run 9_bigger_batch         --batch_size $((BIG_BATCH * 2)) $OPTS
run 10_bigger_batch_ckpt   --batch_size $((BIG_BATCH * 2)) $OPTS --act_ckpt

echo
python report.py "$RESULTS"
