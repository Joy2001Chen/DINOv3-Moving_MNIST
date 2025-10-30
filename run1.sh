#!/usr/bin/env bash
#SBATCH --job-name=moving_mnist_dinov3
#SBATCH --output=./sbatch_log/moving_mnist.out
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1

#SBATCH --nodelist=bmicgpu03
#SBATCH --cpus-per-task=4
#SBATCH --mem=35GB

set -euo pipefail

########## 用户需确认的路径 ##########
# 本地 DINOv3 权重（你下载的 .pth）
DINO3_VITS16_WEIGHTS="./checkpoints/dinov3_vits16_pretrain.pth"
DINO3_VITB16_WEIGHTS="。/checkpoints/dinov3_vitb16_pretrain.pth"

# dinov3 源码目录（可 import）
# DINO3_REPO="$HOME/dinov3"

# 数据目录（MovingMNIST 会在这里找 MNIST；若没下载会尝试联网下载）
DATA_ROOT="./data"

########## 环境变量：离线 + 限制线程 + 安静 ##########
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=3

# 让 Python 能 import dinov3 仓库
#export PYTHONPATH="${PYTHONPATH}:${DINO3_REPO}"


#conda activate bmic
# pip install --upgrade pip
# pip install -r requirements.txt

########## 快速配置（小数据 + 1 epoch） ##########
EPOCHS=1
TRAIN_SEQS=512
VAL_SEQS=128
TEST_SEQS=256
BATCH_SIZE=8
TEMPORAL="lstm"           # 也可改为 transformer
COND_LEN=10
SEQ_LEN=20
NUM_DIGITS=2

# 选择已有的权重与对应 token 维度
USE_ARCH=""
WEIGHTS=""
TOKEN_DIM=384

if [ -f "$DINO3_VITS16_WEIGHTS" ]; then
  USE_ARCH="vits16"
  WEIGHTS="$DINO3_VITS16_WEIGHTS"
  TOKEN_DIM=384
elif [ -f "$DINO3_VITB16_WEIGHTS" ]; then
  USE_ARCH="vitb16"
  WEIGHTS="$DINO3_VITB16_WEIGHTS"
  TOKEN_DIM=768
else
  echo "[Error] 未找到本地 DINOv3 权重，请检查路径："
  echo "  $DINO3_VITS16_WEIGHTS"
  echo "  $DINO3_VITB16_WEIGHTS"
  exit 1
fi

# 检查 MNIST 是否已在本地（避免离线失败）
if [ ! -d "${DATA_ROOT}/MNIST/raw" ] && [ ! -d "${DATA_ROOT}/MNIST/processed" ]; then
  echo "[Warn] 本地未检测到 MNIST 数据，若计算节点无外网，下载会失败。"
  echo "       请在有外网的登录节点先运行一次下载："
  echo "         python -c 'from torchvision import datasets; datasets.MNIST(root=\"${DATA_ROOT}\", train=True, download=True); datasets.MNIST(root=\"${DATA_ROOT}\", train=False, download=True)'"
fi

timestamp() { date +"%Y%m%d_%H%M%S"; }
TS="$(timestamp)"
CKPT_DIR="checkpoints/${TS}"
OUT_DIR="outputs/${TS}"
LOG_DIR="logs"
mkdir -p "$CKPT_DIR" "$OUT_DIR" "$LOG_DIR"

echo "=== QUICK RUN CONFIG ==="
echo "ARCH=${USE_ARCH}  WEIGHTS=${WEIGHTS}"
echo "TOKEN_DIM=${TOKEN_DIM}  TEMPORAL=${TEMPORAL}"
echo "EPOCHS=${EPOCHS}  TRAIN_SEQS=${TRAIN_SEQS}  VAL_SEQS=${VAL_SEQS}  TEST_SEQS=${TEST_SEQS}"
echo "BATCH_SIZE=${BATCH_SIZE}  DATA_ROOT=${DATA_ROOT}"
echo "========================"

CKPT="${CKPT_DIR}/latent_dino3_${USE_ARCH}_${TEMPORAL}.pt"
SAMPLES="${OUT_DIR}/samples"

########## 训练（极简一轮，快速出样例图） ##########
python train_predict.py \
  --use-dino3 \
  --dino3-arch "${USE_ARCH}" \
  --dino3-weights "${WEIGHTS}" \
  --epochs "${EPOCHS}" \
  --train-seqs "${TRAIN_SEQS}" \
  --val-seqs "${VAL_SEQS}" \
  --batch-size "${BATCH_SIZE}" \
  --seq-len "${SEQ_LEN}" \
  --cond-len "${COND_LEN}" \
  --num-digits "${NUM_DIGITS}" \
  --temporal "${TEMPORAL}" \
  --token-dim "${TOKEN_DIM}" \
  --d-model 256 \
  --layers 1 \
  --dropout 0.0 \
  --save "${CKPT}" \
  --samples-out "${SAMPLES}" \
  2>&1 | tee "${LOG_DIR}/train_quick_${TS}.log"

########## 评估（输出 MSE/PSNR/SSIM + 保存若干样例） ##########
python eval_predict.py \
  --checkpoint "${CKPT}" \
  --test-seqs "${TEST_SEQS}" \
  --batch-size "${BATCH_SIZE}" \
  --seq-len "${SEQ_LEN}" \
  --cond-len "${COND_LEN}" \
  --num-digits "${NUM_DIGITS}" \
  --outdir "${OUT_DIR}/eval" \
  --num-samples 12 \
  2>&1 | tee "${LOG_DIR}/eval_quick_${TS}.log"

echo "Quick run finished."
echo "Checkpoint: ${CKPT}"
echo "Samples:    ${SAMPLES}"
echo "Eval out:   ${OUT_DIR}/eval"
echo "Logs:       ${LOG_DIR}"
