# DINOv3-Moving_MNIST

## 项目简介

本项目基于 Moving MNIST 数据集，探索使用 DINOv2/DINOv3 自监督视觉 Transformer 作为冻结特征提取器，并通过轻量级时序模块（LSTM 或 Transformer）完成未来帧预测与碰撞事件判别。仓库同时保留了 ConvLSTM 等经典时空基线，便于对比实验。

典型工作流：

1. 基于 torchvision/MNIST 动态合成 Moving MNIST 序列；
2. 使用 DINO 特征抽取每个时间步的 patch/CLS token；
3. 通过时序编码器学习潜在轨迹并输出未来帧；
4. 评估生成序列 (MSE/PSNR/SSIM)，保存可视化与检查点。

## 主要特性

- ✅ DINOv2 (HuggingFace) 与 DINOv3（本地权重）双后端支持；
- ✅ LSTM 与 Transformer 两种时序建模方式，可独立调参；
- ✅ 自动生成 Moving MNIST，支持随机/确定性采样与碰撞标签；
- ✅ 预测任务、碰撞分类任务及 ConvLSTM 基线脚本；
- ✅ 训练/评估流程统一，产出日志、样本图与模型检查点；
- ✅ 提供面向集群的 `run.sh`（本地）与 `run1.sh`（Slurm）范例。

## 目录结构

- `data/`：Moving MNIST 数据生成脚本；
- `models/`：DINO 后端、时序模块、解码器与基线模型；
- `utils/`：度量指标、随机种子、可视化工具；
- `train_predict.py`：主任务（未来帧预测）训练脚本；
- `eval_predict.py`：预测任务的量化评估与可视化；
- `train_classify_collision.py`：碰撞事件二分类；
- `train_predict_convlstm.py`：ConvLSTM 基线训练；
- `run.sh` / `run1.sh`：快速实验证脚本（本地 / Slurm）；
- `logs/`、`checkpoints/`、`outputs/`：默认日志、权重与样本输出目录。

## 环境准备

1. 建议使用 Python 3.10+，并与 PyTorch 2.1 / CUDA 兼容的环境。
2. 创建虚拟环境并安装依赖：

```bash
python -m venv .venv        # 或使用 conda
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

3. 若使用 HuggingFace 版 DINOv2，可直接联网下载；离线场景下请提前缓存模型或设置：

```bash
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
```

4. 若需启用 DINOv3，本地准备：

```bash
git clone https://github.com/facebookresearch/dinov3.git
export PYTHONPATH=$PYTHONPATH:/path/to/dinov3
# 将预训练权重 *.pth 放入 ./checkpoints/ 或设置 DINO3_WEIGHTS=/path/to/file.pth
```

## 数据准备

首次运行时 `torchvision.datasets.MNIST` 会自动下载数据到 `./data/MNIST/`。离线集群环境请预先运行一次：

```bash
python -c "from torchvision import datasets; datasets.MNIST(root='./data', train=True, download=True); datasets.MNIST(root='./data', train=False, download=True)"
```

Moving MNIST 序列在训练期间按需即时生成，无需额外存储。

## 快速上手

### 1. 体验版（10 epoch，DINOv3，单 GPU）

```bash
bash run1.sh
```

脚本会自动：
- 选择本地可用的 DINOv3 权重；
- 运行 `train_predict.py` 进行 10 epoch 快速训练；
- 调用 `eval_predict.py` 输出 `[Test] MSE/PSNR/SSIM` 并保存样本图。

### 2. 标准实验

```bash
# 训练
python train_predict.py \
  --backbone facebook/dinov2-small \
  --temporal lstm \
  --epochs 40 \
  --train-seqs 80000 --val-seqs 10000 \
  --batch-size 32 \
  --save checkpoints/latest.pt \
  --samples-out outputs/latest_samples

# 评估
python eval_predict.py \
  --checkpoint checkpoints/latest.pt \
  --test-seqs 10000 \
  --batch-size 32 \
  --outdir outputs/latest_eval
```

常用调参入口：
- `--temporal {lstm, transformer}`、`--d-model`、`--layers`、`--transformer-heads`；
- `--use-dino3`、`--dino3-arch {vits16, vitb16}`、`--token-dim`；
- `--seq-len`、`--cond-len`、`--num-digits` 控制任务难度。

## 其他脚本

- `train_classify_collision.py`：基于 CLS token + 时序编码器的碰撞事件检测（0/1 标签），指标为准确率。
- `train_predict_convlstm.py`：无 DINO 特征的 ConvLSTM 基线，可用来评估纯时空卷积效果。
- `quickcheck/`：包含轻量调试脚本与用于 sanity check 的样例配置。

## 日志与结果

- 训练与评估日志默认写入 `logs/`。例如 `logs/train_quick_20251031_150358.log` 展示了 10 epoch 快速实验，验证集 MSE 约 0.041。
- 评估日志 `logs/eval_quick_20251029_221809.log` 报告测试集指标：`MSE=0.0426 / PSNR=13.86 / SSIM=0.7201`。
- 样本帧对比图保存在 `outputs/<timestamp>/`，命名为 `epochXX_sampleY.png`。

## 已知问题 & 提示

- **加载检查点时报 decoder key 错误**：由于 `TokenGridDecoder` 会在首次前向传播时按需初始化，`eval_predict.py` 可能在 `load_state_dict` 前尚未构建解码器。临时解决方式是在加载前做一次哑输入前向，例如：

  ```python
  dummy = torch.zeros(1, cfg["cond_len"], 1, 64, 64, device=device)
  model(dummy)
  model.load_state_dict(ckpt["model"])
  ```

- **大规模并行**：建议限制 `OMP_NUM_THREADS` 等环境变量（`run.sh` 已给出示例），避免数据加载争用。

## 许可证

请在发布或开源前补充明确的许可证条款；若遵循原始 DINOv3/DINOv2 许可，请同时保留其版权声明。
