# DINOv3-Moving\_MNIST: Future Frame Prediction via Frozen ViT Features

## Project Overview

This project explores and validates the potential of **self-supervised visual foundation models** (DINOv2/DINOv3) in the **video prediction** task. We utilize a frozen DINO Transformer as an efficient feature extractor, combined with lightweight temporal modules (LSTM/Transformer), to perform future frame prediction and dynamic event classification on the Moving MNIST dataset.

By adopting this **"feature-temporal" decoupled** architecture, we aim to significantly reduce model training cost and complexity while maintaining predictive quality. The repository also retains classic spatiotemporal baselines, such as ConvLSTM, for comparative experiments.

### Core Objectives

1.  **Feature Efficiency Exploration:** Evaluate the performance of frozen DINOv2/DINOv3 features combined with a small temporal head (LSTM/Transformer) on Moving MNIST.
2.  **High-Performance Prediction:** Achieve high accuracy in future frame prediction (both qualitative and quantitative metrics: MSE/PSNR/SSIM).
3.  **Multi-Task Extension:** Extend the task to collision event classification, building the foundation for subsequent multi-task learning and world model development.

-----

## 2. Methodology and Architecture

### 2.1 Overall Architecture

This project adopts the **latent feature prediction** paradigm, comprising the following main components:

1.  **Data Generation:** `data/moving_mnist.py` dynamically synthesizes Moving MNIST sequences, supporting random/reproducible sampling and collision label generation.
2.  **Visual Feature Extraction (Frozen):** A frozen DINOv2 (HuggingFace weights) or DINOv3 (local weights) is used as the **frozen** backbone.
      * **Prediction Task:** **Patch Token grid** from each frame is extracted as input to the temporal encoder.
      * **Classification Task:** **CLS Token** from each frame is extracted as input to the temporal encoder.
3.  **Temporal Modeling:** A lightweight encoder is used to learn the latent dynamics.
      * **Available Encoders:** **LSTM** (suited for sequence modeling) or **Transformer** (suited for long-range dependencies within the sequence). The encoder weights are shared across all Patch Tokens.
4.  **Future Frame Decoding:** The **`TokenGridDecoder`** receives the predicted future Patch Token grid and reconstructs it into a **64×64 grayscale image frame**.

### 2.2 Evaluation Metrics

  * **Generation Quality:** **MSE** (Mean Squared Error), **PSNR** (Peak Signal-to-Noise Ratio), **SSIM** (Structural Similarity Index Measure).
  * **Qualitative Assessment:** Export sample grid images for visual comparison.
  * **Classification Task:** Accuracy.

-----

## 3. Experimental Configuration and Results Analysis

### 3.1 Quick Experiment Configuration (`run1.sh`)

This configuration is used for quickly validating the model's learning capability and the training pipeline:

| Configuration Item | Details |
| :--- | :--- |
| **Script** | `train_predict.py` |
| **Training Epochs** | 10 |
| **Backbone** | DINOv3 ViT-S/16 (Frozen) |
| **Temporal Encoder** | LSTM (d\_model=256, layers=1, dropout=0) |
| **Sequence Length** | `seq_len=20` (10 frames condition, predict next 10 frames) |
| **Number of Digits** | 2 digits |
| **Sample Size** | Train 512 / Validation 128 / Test 256 |
| **Batch Size** | 8 |
| **GPU Utilization** | Approx. 15–16 s per iteration, high stability |

### 3.2 Performance Results (Best Epoch)

| Phase | Metric | Value | Notes |
| :--- | :--- | :--- | :--- |
| **Validation Set** | Val MSE | | |
| **Validation Set** | Val PSNR | | |
| **Test Set** | Test MSE | | |
| **Test Set** | Test PSNR | | |
| **Test Set** | Test SSIM | | |

### 3.3 Training Stability Observation

TQDM logs indicate that due to the shallow LSTM layer count and the use of frozen features, the loss was stable in the initial training phase, and **no gradient explosion or divergence was observed**. The model has achieved preliminary convergence within 10 epochs.

-----

## 4. Suggested Future Work

| Priority | Task | Goal |
| :--- | :--- | :--- |
| **High** | Refine Evaluation Script | Implement decoder initialization logic within `eval_predict.py` to ensure the decoder is constructed before loading the checkpoint. |
| **High** | Long-Term Training | Extend training to 30–50 epochs, enable `weight_decay` for `AdamW` and learning rate scheduling (e.g., `ReduceLROnPlateau`) to improve generalization. |
| **Medium** | Transformer Head Comparison | Reproduce experiments comparing **LSTM** vs. **Transformer** (with added Patch interaction) at similar FLOPs to quantify the benefit of inter-patch communication. |
| **Medium** | Integrate Classification Task | Attempt shared DINO features, jointly optimizing **future frame prediction** and **collision classification**, to verify the positive effect of multi-task synergy. |
| **Low** | Automated Evaluation Checklist | Write a script to batch read `logs/` and summarize results into CSV/Markdown for easy tracking of experiments across different timestamps. |

-----

## 5. Repository Structure and Quick Start

### Directory Structure

  * `data/`: Moving MNIST data generation scripts.
  * `models/`: DINO backend, temporal modules, decoder, and baseline models (e.g., ConvLSTM).
  * `utils/`: Metric calculation, random seeding, visualization tools.
  * `train_predict.py`: **Main task (future frame prediction)** training script.
  * `eval_predict.py`: Quantitative evaluation and visualization for the prediction task.
  * `train_classify_collision.py`: Collision event binary classification script.
  * `train_predict_convlstm.py`: ConvLSTM baseline training.
  * `run.sh` / `run1.sh`: Quick experiment scripts (Local / Slurm).
  * `logs/`, `checkpoints/`, `outputs/`: Default directories for logs, weights, and sample outputs.

### Environment Setup

1.  It is recommended to use Python 3.10+ in an environment compatible with PyTorch 2.1 / CUDA.
2.  Create a virtual environment and install dependencies:

<!-- end list -->

```bash
python -m venv .venv        # Or use conda
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

3.  To enable DINOv3, manually clone the repository and set the environment variable:

<!-- end list -->

```bash
git clone https://github.com/facebookresearch/dinov3.git
export PYTHONPATH=$PYTHONPATH:/path/to/dinov3
# Place pre-trained weights *.pth into ./checkpoints/ or set DINO3_WEIGHTS=/path/to/file.pth
```

### Quick Start

Run the quick experiment:

```bash
bash run1.sh
```

Standard training and evaluation examples:

```bash
# Training (using DINOv2 small, 40 epochs)
python train_predict.py \
  --backbone facebook/dinov2-small \
  --temporal lstm \
  --epochs 40 \
  --train-seqs 80000 --val-seqs 10000 \
  --batch-size 32 \
  --save checkpoints/latest.pt \
  --samples-out outputs/latest_samples

# Evaluation
python eval_predict.py \
  --checkpoint checkpoints/latest.pt \
  --test-seqs 10000 \
  --batch-size 32 \
  --outdir outputs/latest_eval
```
