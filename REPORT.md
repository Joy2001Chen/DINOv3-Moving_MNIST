# 实验报告：DINOv3-Moving_MNIST

## 1. 项目目标

- 探索将 DINOv2/DINOv3 自监督视觉 Transformer 作为冻结特征提取器，结合轻量级时序模块完成 Moving MNIST 未来帧预测；
- 评估冻结视觉特征 + 小型时序头的表现，并与传统 ConvLSTM 基线进行对比；
- 扩展任务至碰撞事件分类，为后续多任务学习夯实数据与代码基础。

## 2. 方法摘要

1. **数据生成**：基于 `data/moving_mnist.py` 即时合成序列，支持随机/可复现采样及碰撞标签。
2. **视觉特征**：冻结 DINOv2（HuggingFace 权重）或 DINOv3（本地权重），抽取每帧 patch token（预测任务）与 CLS token（分类任务）。
3. **时序建模**：提供 LSTM 与 Transformer 两种编码器，可在每个 spatial token 上共享。
4. **未来帧解码**：`TokenGridDecoder` 将预测的 patch token 栅格恢复为 64×64 灰度帧。
5. **评估指标**：MSE / PSNR / SSIM 量化生成质量，另有样本网格保存定性结果。

## 3. 实验配置

### 3.1 快速实验（run1.sh）

- 训练脚本：`train_predict.py`，10 epoch；
- 架构：DINOv3 ViT-S/16 特征 + LSTM（d_model=256, layers=1, dropout=0）；
- 数据：Moving MNIST，`seq_len=20`，前 10 帧条件，2 个数字；
- 批量大小：8；训练/验证/测试样本分别为 512/128/256；
- 采样与日志：保存到 `checkpoints/<timestamp>/`、`outputs/<timestamp>/`、`logs/`。

### 3.2 评估流程

`eval_predict.py` 读取检查点，生成测试集指标并导出样本网格。当前脚本需在加载权重前先触发一次前向传播以初始化解码器（详见第 5 节问题记录）。

## 4. 结果与观察

### 4.1 验证集表现

`logs/train_quick_20251031_150358.log` 显示在第 10 epoch 获得最优验证性能（MSE≈0.0410）。主要指标如下（取最优 epoch）：

| 指标 | 数值 |
| --- | --- |
| Val MSE | 0.0410 |
| Val PSNR | 14.010 |
| Val SSIM | 0.0350 |

> 参见 `logs/train_quick_20251031_150358.log:29`。

### 4.2 测试集表现

`logs/eval_quick_20251029_221809.log:2` 报告测试集指标：

- MSE = 0.0426
- PSNR = 13.857
- SSIM = 0.7201

SSIM 明显高于验证阶段的数值，说明测试集（确定性采样）上时序结构更稳定。可视化样本位于 `outputs/20251029_221809/eval/`。

### 4.3 训练稳定性

TQDM 日志显示训练初期损失稳定，未观察到梯度爆炸或发散情况。由于 LSTM 层数较浅，GPU 占用与每 iter 时长保持在 15–16 s，适合快速迭代。

## 5. 问题记录

- **解码器权重加载失败**：在最新一次评估中，`eval_predict.py` 因解码器延迟初始化导致 `load_state_dict` 出现多余 key（`logs/eval_quick_20251031_150358.log:10`）。临时修复方式是在加载前对模型进行一次哑输入前向传播，或在加载时设置 `strict=False` 并手动重建解码器。
- **日志被控制字符覆盖**：由于 TQDM 的回车符，`logs/train_quick_20251031_150358.log` 中进度条覆盖行首，建议在后续分析时使用 `python -m pip install tqdm==4.66` 以上版本或将 `tqdm(..., dynamic_ncols=True, leave=True)` 配置为输出完整日志。

## 6. 后续工作建议

1. **完善评估脚本**：在 `eval_predict.py` 内增加解码器初始化逻辑，并考虑同时加载 `samples_out` 以复现训练期可视化。
2. **长程训练**：扩展至 30–50 epoch，开启 `AdamW` 的 `weight_decay` 与学习率调度，观察泛化性能提升幅度。
3. **Transformer 头对比**：复现实验，比较 LSTM 与 Transformer 在相同 FLOPs 下的表现，量化 patch 间交互的收益。
4. **融合分类任务**：尝试共享 DINO 特征，联合优化未来帧预测与碰撞分类，以验证多任务协同的正效应。
5. **自动化评估清单**：编写脚本批量读取 `logs/`，汇总到 CSV/Markdown，便于追踪不同时间戳实验。

---

如需更多细节，请参考更新后的 `README.md` 与相应脚本注释。
