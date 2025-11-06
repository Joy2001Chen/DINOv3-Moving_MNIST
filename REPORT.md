
# DINOv3-Moving\_MNIST: Future Frame Prediction via Frozen ViT Features
### Core Goal

- **Future Frame Prediction:**  
  The model receives a sequence of dynamic frames (e.g., the first 5 frames) and predicts the next few frames (e.g., the next 5).  

- **High Precision Requirement:**  
  The model’s predictions must be **extremely close** to the ground truth. The target metric on small-scale tests is a **Mean Squared Error (MSE)** below **1e-2 (0.01)** — a very high level of accuracy.  
  - **Mean Squared Error (MSE):** Measures the average squared difference between predictions and true values. The smaller, the better.  

- **Avoid “Gray/Black Collapse”:**  
  A common failure mode where the model outputs uniform gray or black frames with no discernible features. The model must avoid collapsing into this state.

---

### Architecture: DINO-Based Latent Predictor

Your model is a hybrid system combining state-of-the-art computer vision and sequence modeling techniques:

1. **DINO v2/v3 Backbone (Vision Transformer):**
   - **DINO:** A self-supervised **Vision Transformer (ViT)** that learns image representations without labeled data. It decomposes raw images into meaningful high-dimensional **tokens**.
   - **Role:** Receives 64×64 image frames, splits them into **patches**, and extracts per-patch latent **features**.

2. **Per-Patch Spatial Mixer:**
   - **Role:** Processes and mixes DINO’s patch-level features to capture spatial relationships within a frame (e.g., the relationship between different parts of a digit).

3. **Temporal Encoder (LSTM/Transformer):**
   - **Role:** The core **predictive module**. It takes feature sequences (tokens from past frames) and models **temporal dependencies** to predict future tokens.
   - **LSTM (Long Short-Term Memory):** A type of **RNN** specialized for sequential data with long-range dependencies.
   - **Transformer:** A modern alternative that uses **self-attention** to model temporal relations more efficiently.

4. **MLP Head (Multi-Layer Perceptron):**
   - **Role:** Refines the temporal encoder’s predicted tokens to produce higher-quality latent representations for the next frames.

5. **Convolutional Decoder:**
   - **Role:** Converts the refined latent tokens back into **64×64 pixel frames** using convolutional layers, restoring human-interpretable images.

6. **Cached Tokens (Optional):**
   - **Role:** Pre-computed DINO features can be loaded directly to **skip the heavy backbone pass**, significantly accelerating experiments.

---

## Training Pipeline Details

### Data Preparation

- **MovingMNISTDataset:**  
  Generates synthetic sequences from the **Moving-MNIST** dataset, containing one or more handwritten digits moving and bouncing within a 64×64 frame.

- **FeatureCache:**  
  Manages retrieval of pre-computed **DINO tokens**.  
  **Important:** Data generation must be **deterministic** (fixed seed, identical order) to ensure cached features align with current samples.

---

### 🧠 Model Execution

- **LatentDynViTForecaster:**  
  The end-to-end model responsible for feature extraction, spatial mixing, temporal forecasting, and decoding predicted frames.

- **Debug Hooks:**  
  Internal monitoring tools that log token or logit statistics, helping diagnose problems such as saturation or collapse.

---

### 🏃 Training Loop

- **Device Resolution:**  
  Automatically selects the optimal device (preferably **GPU/CUDA**, fallback to **CPU**).

- **AMP (Automatic Mixed Precision):**  
  Uses both 16-bit and 32-bit floats to **speed up training** and **reduce memory usage** without major accuracy loss (enabled only on CUDA).

- **GradScaler:**  
  Works with AMP to prevent **gradient underflow** in half precision by scaling and unscaling gradients safely.

- **Validation:**  
  Mirrors the training path for consistent evaluation between train and test phases.

---

### Debugging Aids

- **`--debug-checks`:**  
  Enables detailed logging of **decoder activations** and **logit statistics** to detect **Sigmoid collapse** (when logits saturate into −∞ or +∞, causing gradients to vanish).

- **Overfit Test:**  
  Trains on **a single small batch** for multiple iterations to verify the model’s ability to **fit simple data**.  
  If the model cannot overfit a small batch, its architecture or optimization is fundamentally broken.

---

## Key Findings & Fixes

1. **Decoder Collapse:**
   - **Cause:** The decoder was not properly **registered in the optimizer**, so its parameters were never updated. Additionally, the **learning rate (LR)** was **too high**.
   - **Optimizer:** Updates model parameters based on the loss gradient; missing modules lead to “frozen” parts of the model.
   - **Fix:** Instantiate the decoder in `__init__` so it’s included in the optimizer, and **lower the LR**.

2. **Learning Rate Adjustment:**
   - Original LR (0.01) is too high; it drives decoder logits to extreme negatives, causing Sigmoid outputs to collapse to zero.
   - **Recommended LR:** `3e-4` (0.0003) for stable training.

3. **Cached Feature Alignment:**
   - If cached features are used but data sampling is **non-deterministic**, gradients vanish due to feature/label mismatch.  
     Ensure identical random seeds and deterministic sampling.

4. **GPU Confirmation:**
   - Verified that CUDA must be available in PyTorch for proper GPU acceleration.

---

## Next Steps

1. **Lock Parameters and Re-run Tests:**
   - Fix LR at **3e-4**, enable **deterministic caching**, and re-run **single-batch overfitting tests**.  
     Expect **MSE < 1e-2** within a few hundred iterations.

2. **Scale to Full Dataset:**
   - After successful overfitting, resume training on the full dataset (`train_seqs` / `val_seqs`) for large-scale performance.

3. **Continuous Monitoring:**
   - Track **decoder logits** throughout training to ensure they remain within a healthy range.  
   - Adjust **dropout** and **regularization** if overfitting occurs.
     - **Dropout:** Randomly disables neurons during training to improve generalization.
     - **Regularization:** Adds penalties on model parameters to prevent over-complexity.

4. **Automation & Regression Tests:**
   - Automate **feature-cache validation** and implement **regression tests** to ensure all components (data, backbone, cache, model) remain compatible across future changes.
