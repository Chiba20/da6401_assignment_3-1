# DA6401 - Assignment 3: Implementing the Transformer for Machine Translation


- 📊 [W&B Project Dashboard](https://wandb.ai/ge26z812-iitm-india/da6401-a3)
- 🐙 [GitHub Repository](https://github.com/yourusername/da6401-assignment-3)

## Overview


In this assignment, you will implement the landmark architecture from the paper "Attention Is All You Need" from scratch using PyTorch. The goal is to develop a Neural Machine Translation (NMT) system capable of translating text from German to English using the Multi30k dataset.

This assignment includes ablation studies and detailed analysis of transformer mechanisms including:
- Scaling factor in scaled dot-product attention
- Gradient flow during training
- Attention head specialization and redundancy
- Learned positional encodings
- Label smoothing effects

## Project Structure

```
assignment3/
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
├── model.py                          # Core Transformer architecture
├── dataset.py                        # Multi30k dataset loading
├── lr_scheduler.py                   # Noam learning rate scheduler
├── train.py                          # Main training script with greedy decoding
│
├── Qn1_train_fixed_lr.py             # Qn 2.1: Training with fixed learning rate
├── Qn2_no_scaling.py                 # Qn 2.2: Ablation - Scaling Factor (no 1/√dk)
├── Qn3_attention_visualization.py    # Qn 2.3: Attention head specialization analysis
├── Qn4_learned_positional.py         # Qn 2.4: Learned positional encodings
├── Qn5_label_smoothing.py            # Qn 2.5: Label smoothing analysis
│
└── wandb/                            # W&B run logs
```

## Submission Checklist

1. Train the model with `python train.py`. This creates `transformer_checkpoint.pt`.
2. Upload `transformer_checkpoint.pt` to Google Drive and make it accessible by link.
3. Put the Drive file id in `model.py` by replacing `CHECKPOINT_GDRIVE_ID`, or set the environment variable `DA6401_A3_CHECKPOINT_ID`.
4. Do not upload the checkpoint to Gradescope. Submit the code files only.
5. Before submitting, verify the autograder path:

```python
from model import Transformer

model = Transformer().to(device)
model.eval()
english_sentence = model.infer(german_sentence)
```

## Setup & Requirements

### Installation

```bash
pip install -r requirements.txt
```

### Required Packages
- torch
- datasets (HuggingFace)
- spacy (for tokenization)
- matplotlib, seaborn (visualization)
- scipy (for cosine similarity)
- wandb (experiment tracking)

```bash
# Download spacy tokenizer
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```

## Baseline Training

```bash
# Train the standard transformer with Noam scheduler
python train.py
```

**Output:**
- `transformer_checkpoint.pt` - Best model checkpoint
- W&B logs with training curves

---

## Question-Specific Experiments

### Qn 2.1: Fixed Learning Rate Training

**File:** `Qn1_train_fixed_lr.py`

**Purpose:** Compare Noam scheduler (with warmup) vs. fixed learning rate

**Run:**
```bash
python Qn1_train_fixed_lr.py
```

**Expected Output:**
- Training/validation loss curves
- Analysis of learning rate impact on convergence
- W&B logs comparing both LR strategies

---

### Qn 2.2: Ablation Study - Scaling Factor 1/√dk (10 Marks)

**File:** `Qn2_no_scaling.py`

**Paper Reference:** "Attention Is All You Need" Section 3.2.1 - Scaled Dot-Product Attention

**Question Requirements:**
- Train transformer **WITHOUT** the 1/√dk scaling factor
- Log gradient norms of Query (Q) and Key (K) weights during first 1,000 training steps
- Analyze the "vanishing gradient" problem

**Key Insight:**
The paper argues that for large values of $d_k$, dot products grow large in magnitude, pushing softmax into saturation regions with extremely small gradients. The 1/√dk scaling stabilizes this.

**Run:**
```bash
python Qn2_no_scaling.py
```

**What It Does:**
1. Creates Transformer with `use_attention_scaling=False`
2. Logs per-layer Q/K gradient norms every step (first 1000)
3. Logs to W&B: `train/grad_norm_Q_mean`, `train/grad_norm_K_mean`, layer-specific norms
4. Prints gradient norm summary at end

**Expected Output:**
```
Using device: cuda
✓ Dataset loaded
✓ Model loaded
--- Epoch 1/20 ---
train_loss=5.1234 val_loss=4.9876 val_bleu=0.25
✓ Best checkpoint saved

Gradient Norm Summary (First 1000 steps):
Q mean norms: min=0.0012, max=0.0456
K mean norms: min=0.0008, max=0.0389
```

**Analysis Points:**
- Compare gradient norms WITH vs WITHOUT scaling
- Observe if gradients vanish (become very small) without scaling
- Relate empirical findings to paper's theory

**W&B Metrics:**
- `train/grad_norm_Q_mean` - Average Q gradient norm
- `train/grad_norm_K_mean` - Average K gradient norm
- `train/grad_norm_Q_layer_1-3` - Per-layer Q norms
- `train/grad_norm_K_layer_1-3` - Per-layer K norms

---

### Qn 2.3: Attention Rollout & Head Specialization (10 Marks)

**File:** `Qn3_attention_visualization.py`

**Question Requirements:**
- Extract attention weights from **LAST encoder layer** for a sample sentence
- Visualize heatmaps for each individual Multi-Head Attention head
- Analyze: Which heads perform distinct tasks?
- Detect: Head redundancy?

**Key Questions:**
1. Do some heads attend to next tokens (sequential)?
2. Do some heads capture long-range dependencies?
3. Are heads redundant (similar patterns)?

**Run:**
```bash
python Qn3_attention_visualization.py
```

**What It Does:**
1. Loads trained model and sentence: "ein mann spielt gitarre"
2. Extracts attention weights from last encoder layer using forward hook
3. Analyzes 8 attention heads using 3 metrics:
   - **Next-Token Score**: Focus on adjacent positions (i→i+1)
   - **Long-Range Score**: Focus on distant tokens (distance > 2)
   - **Diagonal Score**: Self-attention focus (position i→i)
4. Classifies each head: `Next-Token` | `Long-Range` | `Self-Attention` | `General`
5. Computes **Head Redundancy**: Average cosine similarity between all head pairs
6. Generates heatmap visualization for each head

**Expected Output:**
```
Qn 2.3: Attention Head Specialization Analysis
======================================================================
✓ Dataset loaded
✓ Model loaded
✓ Hook registered on last encoder layer
✓ Input sentence: ein mann spielt gitarre
✓ Tokens: ['<sos>', 'ein', 'mann', 'spielt', 'gitarre', '<eos>']
✓ Attention shape: torch.Size([1, 8, 6, 6])

ATTENTION HEAD ANALYSIS (Last Encoder Layer)
======================================================================

Head 1:
  Next-Token Score:  0.4231
  Long-Range Score:  0.1856
  Diagonal Score:    0.2145
  Specialization:    Next-Token

Head 2:
  Next-Token Score:  0.1245
  Long-Range Score:  0.3892
  Diagonal Score:    0.1654
  Specialization:    Long-Range

...

Head Redundancy Analysis:
  Average Cosine Similarity: 0.3421
  Interpretation: Low Redundancy (heads are diverse)
======================================================================

Generating attention heatmaps...
  ✓ Saved attention_head_1.png
  ✓ Saved attention_head_2.png
  ...
  ✓ Results logged to W&B

✓ Attention head analysis complete!
```

**Outputs:**
- 8 PNG files: `attention_head_1.png` through `attention_head_8.png`
- Console analysis with per-head scores
- Head redundancy score
- W&B heatmaps and metrics

**Analysis Insights to Look For:**
- **Head Specialization**: Different heads should show different patterns
- **Next-Token Heads**: Strong diagonal at position i→i+1
- **Long-Range Heads**: Strong connections between distant tokens
- **Redundancy Score**: 
  - 0.0-0.3: High specialization (good)
  - 0.5+: High redundancy (some heads are similar)

**W&B Metrics:**
- `attention/head_1` through `attention/head_8` - Heatmaps
- `analysis/head_redundancy` - Redundancy score (0-1)
- `analysis/num_next_token_heads` - Count
- `analysis/num_long_range_heads` - Count
- `heads/head_N/next_token_score` - Per-head metrics
- `heads/head_N/long_range_score`
- `heads/head_N/diagonal_score`

---

### Qn 2.4: Learned Positional Encodings (10 Marks)

**File:** `Qn4_learned_positional.py`

**Run:**
```bash
python Qn4_learned_positional.py
```

---

### Qn 2.5: Label Smoothing (10 Marks)

**File:** `Qn5_label_smoothing.py`

**Run:**
```bash
python Qn5_label_smoothing.py
```

---

## Model Architecture

### Transformer Components

1. **Embeddings**: Token + Positional (either sinusoidal or learned)
2. **Encoder**: 3 layers of multi-head attention + feed-forward
3. **Decoder**: 3 layers with self-attention + cross-attention + feed-forward
4. **Multi-Head Attention**: 8 heads, scaled dot-product attention with optional 1/√dk scaling

### Key Hyperparameters

```python
{
    "d_model": 256,           # Model dimension
    "num_heads": 8,           # Number of attention heads
    "d_ff": 1024,             # Feed-forward hidden dimension
    "N": 3,                   # Number of encoder/decoder layers
    "dropout": 0.1,           # Dropout rate
    "warmup_steps": 4000,     # Noam scheduler warmup
    "batch_size": 64,
    "num_epochs": 20,
    "learning_rate": 1.0,
}
```

## Experiment Tracking with W&B

All experiments log to Weights & Biases for reproducibility and analysis.

**Setup:**
```bash
wandb login
```

**Key Metrics Tracked:**
- Training loss & validation loss
- BLEU score
- Learning rate schedule
- Gradient norms (Qn 2.2)
- Attention visualizations (Qn 2.3)

**View Results:**
```bash
wandb online  # Sync offline runs
```

Access project at: https://wandb.ai/[username]/da6401-a3

## Analysis & Comparison

To compare experiments (e.g., with vs without scaling):

1. **W&B Dashboard**: View side-by-side training curves
2. **Gradient Analysis**: Compare gradient norms across runs
3. **Attention Patterns**: Visualize head specialization

### Expected Findings

**Qn 2.2 (Scaling Factor):**
- Without scaling: Gradients should be smaller/more unstable
- With scaling: More stable gradient flow
- Demonstrates vanishing gradient problem

**Qn 2.3 (Head Specialization):**
- Not all heads are identical
- Some specialize in next-token attention
- Others capture long-range dependencies
- Low redundancy indicates good head diversity

## Inference / Evaluation

```python
from model import Transformer
import torch

model = Transformer().to(device)
model.eval()

# Inference on German sentence
german_sentence = "ein mann spielt gitarre"
english_translation = model.infer(german_sentence)
print(english_translation)  # Expected: "a man plays a guitar"
```

## Troubleshooting

### Q: Model checkpoint not found
**A:** Ensure `transformer_checkpoint.pt` exists or set checkpoint path in config

### Q: W&B not logging
**A:** Run `wandb login` and ensure `use_wandb=True` in config

### Q: Attention weights not captured
**A:** Ensure model has `attention_weights` attribute in MultiHeadAttention (check model.py)

### Q: Import errors
**A:** Install all dependencies: `pip install -r requirements.txt`

## References

- "Attention Is All You Need" - Vaswani et al. (2017)
- Original Paper: https://arxiv.org/abs/1706.03762
- Multi30k Dataset: https://github.com/multi30k/dataset

## Author Notes

- All experiments use the same trained checkpoint for consistency
- Gradient analysis requires raw model (not finetuned)
- Attention visualization uses last encoder layer (most refined representations)
- Head analysis reveals transformer's internal mechanisms

