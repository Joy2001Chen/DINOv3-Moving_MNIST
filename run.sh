#!/usr/bin/env bash
set -euo pipefail

############################
# 可调参数（按需改）
############################
: "${CUDA_VISIBLE_DEVICES:=0}"           # 选 GPU
BACKBONE="${BACKBONE:-facebook/dinov2-small}"   # dinov2-small/base/large/giant
TEMPORAL="${TEMPORAL:-lstm}"             # lstm | transformer
EPOCHS="${EPOCHS:-40}"                   # 充分训练可 30–50
TRAIN_SEQS="${TRAIN_SEQS:-80000}"
VAL_SEQS="${VAL_SEQS:-10000}"
TEST_SEQS="${TEST_SEQS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SEED="${SEED:-42}"

# 线程数（多任务并发时可调小以更稳）
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

############################
# 路径与派生参数
############################
timestamp() { date +"%Y%m%d_%H%M%S"; }
TS="$(timestamp)"

# 根据 backbone 自动设置 token dim
TOKENDIM=384
case "$BACKBONE" in
  *small* ) TOKENDIM=384 ;;
  *base*  ) TOKENDIM=768 ;;
  *large* ) TOKENDIM=1024 ;;
  *giant* ) TOKENDIM=1536 ;;
esac

CKPT_DIR="checkpoints/${TS}"
OUT_DIR="outputs/${TS}"
LOG_DIR="logs"
mkdir -p "$CKPT_DIR" "$OUT_DIR" "$LOG_DIR"

# venv（如已有环境可注释掉）
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
conda activate bmic
pip install --upgrade pip
pip install -r requirements.txt

echo "=== MAIN EXPERIMENT CONFIG ==="
echo "GPU: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "BACKBONE=${BACKBONE}  TOKENDIM=${TOKENDIM}"
echo "TEMPORAL=${TEMPORAL}"
echo "EPOCHS=${EPOCHS}  TRAIN_SEQS=${TRAIN_SEQS}  VAL_SEQS=${VAL_SEQS}  TEST_SEQS=${TEST_SEQS}"
echo "BATCH_SIZE=${BATCH_SIZE}  SEED=${SEED}"
echo "CKPT_DIR=${CKPT_DIR}"
echo "OUT_DIR=${OUT_DIR}"
echo "=============================="

############################
# 1) 训练主实验
############################
CKPT="${CKPT_DIR}/latent_dino_${TEMPORAL}.pt"
SAMPLES="${OUT_DIR}/latent_${TEMPORAL}_samples"

if [ "$TEMPORAL" = "lstm" ]; then
  D_MODEL="${D_MODEL:-512}"
  LAYERS="${LAYERS:-2}"
  DROPOUT="${DROPOUT:-0.1}"

  python train_predict.py \
    --epochs "${EPOCHS}" \
    --train-seqs "${TRAIN_SEQS}" \
    --val-seqs "${VAL_SEQS}" \
    --batch-size "${BATCH_SIZE}" \
    --seq-len 20 --cond-len 10 --num-digits 2 \
    --backbone "${BACKBONE}" \
    --temporal lstm \
    --token-dim "${TOKENDIM}" \
    --d-model "${D_MODEL}" \
    --layers "${LAYERS}" \
    --dropout "${DROPOUT}" \
    --seed "${SEED}" \
    --save "${CKPT}" \
    --samples-out "${SAMPLES}" \
    2>&1 | tee "${LOG_DIR}/train_main_${TEMPORAL}_${TS}.log"

else
  # transformer
  D_MODEL="${D_MODEL:-512}"
  LAYERS="${LAYERS:-4}"
  HEADS="${HEADS:-8}"
  FF="${FF:-1024}"
  DROPOUT="${DROPOUT:-0.1}"

  python train_predict.py \
    --epochs "${EPOCHS}" \
    --train-seqs "${TRAIN_SEQS}" \
    --val-seqs "${VAL_SEQS}" \
    --batch-size "${BATCH_SIZE}" \
    --seq-len 20 --cond-len 10 --num-digits 2 \
    --backbone "${BACKBONE}" \
    --temporal transformer \
    --token-dim "${TOKENDIM}" \
    --d-model "${D_MODEL}" \
    --layers "${LAYERS}" \
    --transformer-heads "${HEADS}" \
    --transformer-ff "${FF}" \
    --dropout "${DROPOUT}" \
    --seed "${SEED}" \
    --save "${CKPT}" \
    --samples-out "${SAMPLES}" \
    2>&1 | tee "${LOG_DIR}/train_main_${TEMPORAL}_${TS}.log"
fi

############################
# 2) 测试评估
############################
python eval_predict.py \
  --checkpoint "${CKPT}" \
  --test-seqs "${TEST_SEQS}" \
  --batch-size "${BATCH_SIZE}" \
  --seq-len 20 --cond-len 10 --num-digits 2 \
  --outdir "${OUT_DIR}/latent_${TEMPORAL}_eval" \
  --num-samples 16 \
  2>&1 | tee "${LOG_DIR}/eval_main_${TEMPORAL}_${TS}.log"

echo "Finished MAIN experiment."
echo "Checkpoint: ${CKPT}"
echo "Outputs:   ${OUT_DIR}"
echo "Logs:      ${LOG_DIR}"
