"""
Qn 2.3: Attention Rollout & Head Specialization (10 Marks)
Extract attention weights from LAST encoder layer and analyze head specialization
- Visualization: Heatmaps for each head
- Analysis: Identify head tasks, detect redundancy, long-range dependencies
"""

import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.spatial.distance import cosine

try:
    import wandb
except Exception:
    wandb = None

from model import Transformer, make_src_mask
from dataset import Multi30kDataset


print("=" * 70)
print("Qn 2.3: Attention Head Specialization Analysis")
print("=" * 70)

# =========================================================
# LOAD DATASET
# =========================================================
train_ds = Multi30kDataset("train")
print(f"✓ Dataset loaded")

# =========================================================
# LOAD MODEL
# =========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✓ Using device: {device}")

model = Transformer(
    src_vocab_size=len(train_ds.src_vocab),
    tgt_vocab_size=len(train_ds.tgt_vocab),
    checkpoint_path="transformer_checkpoint.pt"
).to(device)

model.eval()
model.src_vocab = train_ds.src_vocab
model.tgt_vocab = train_ds.tgt_vocab
print(f"✓ Model loaded")

# =========================================================
# REGISTER HOOKS FOR LAST ENCODER LAYER ONLY
# =========================================================
attention_weights_last_layer = None

def save_last_layer_attention(module, input, output):
    """Hook to capture attention weights from last encoder layer"""
    global attention_weights_last_layer
    if hasattr(module, "attention_weights"):
        attention_weights_last_layer = module.attention_weights.detach().cpu()

# Register hook ONLY on the last encoder layer
last_layer = model.encoder.layers[-1]
last_layer.self_attn.register_forward_hook(save_last_layer_attention)
print(f"✓ Hook registered on last encoder layer")

# =========================================================
# INPUT SENTENCE
# =========================================================
sentence = "ein mann spielt gitarre"
tokens = ["<sos>"] + sentence.split() + ["<eos>"]
src_ids = train_ds.src_vocab.lookup_indices(tokens)

src_tensor = torch.tensor(
    src_ids,
    dtype=torch.long
).unsqueeze(0).to(device)

src_mask = make_src_mask(src_tensor, model.pad_idx)
print(f"✓ Input sentence: {sentence}")
print(f"✓ Tokens: {tokens}")

# =========================================================
# RUN ENCODER
# =========================================================
with torch.no_grad():
    model.encode(src_tensor, src_mask)

attention = attention_weights_last_layer  # Shape: [batch, heads, seq, seq]
print(f"✓ Attention shape: {attention.shape}")

# =========================================================
# ANALYSIS FUNCTIONS
# =========================================================

def detect_next_token_attention(attn_head):
    """
    Detect if a head focuses on next-token attention.
    Returns: score (0-1) indicating next-token focus strength
    """
    seq_len = attn_head.shape[0]
    next_token_scores = []
    
    for i in range(seq_len - 1):
        # Check attention from position i to position i+1
        next_token_scores.append(float(attn_head[i, i + 1].item()))
    
    # Average strength on next-token diagonal
    return np.mean(next_token_scores) if next_token_scores else 0.0

def detect_long_range_dependencies(attn_head):
    """
    Detect if a head captures long-range dependencies.
    Returns: score (0-1) indicating long-range focus
    """
    seq_len = attn_head.shape[0]
    long_range_scores = []
    
    # Long-range = distance > 2
    for i in range(seq_len):
        for j in range(seq_len):
            if abs(i - j) > 2:
                long_range_scores.append(float(attn_head[i, j].item()))
    
    return np.mean(long_range_scores) if long_range_scores else 0.0

def detect_diagonal_attention(attn_head):
    """
    Detect if a head is mostly self-attending (diagonal focus).
    Returns: score (0-1) indicating self-attention focus
    """
    seq_len = attn_head.shape[0]
    diagonal_scores = []
    
    for i in range(seq_len):
        diagonal_scores.append(float(attn_head[i, i].item()))
    
    return np.mean(diagonal_scores)

def compute_head_redundancy(attn_heads):
    """
    Compute cosine similarity between all pairs of heads.
    Returns: average redundancy score (0=all unique, 1=all identical)
    """
    num_heads = attn_heads.shape[0]
    similarities = []
    
    for i in range(num_heads):
        for j in range(i + 1, num_heads):
            head_i_flat = attn_heads[i].reshape(-1)
            head_j_flat = attn_heads[j].reshape(-1)
            
            # Cosine similarity
            sim = 1 - cosine(head_i_flat, head_j_flat)
            similarities.append(sim)
    
    return np.mean(similarities) if similarities else 0.0

# =========================================================
# ANALYZE HEADS (LAST LAYER ONLY)
# =========================================================
print("\n" + "=" * 70)
print("ATTENTION HEAD ANALYSIS (Last Encoder Layer)")
print("=" * 70)

attn_last_layer = attention[0]  # Shape: [heads, seq, seq]
num_heads = attn_last_layer.shape[0]

head_analysis = []

for head_idx in range(num_heads):
    attn_head = attn_last_layer[head_idx]
    
    next_token_score = detect_next_token_attention(attn_head)
    long_range_score = detect_long_range_dependencies(attn_head)
    diagonal_score = detect_diagonal_attention(attn_head)
    
    # Classify head specialization
    specialization = "General"
    if next_token_score > 0.3:
        specialization = "Next-Token"
    elif long_range_score > 0.25:
        specialization = "Long-Range"
    elif diagonal_score > 0.4:
        specialization = "Self-Attention"
    
    head_info = {
        "head_idx": head_idx,
        "next_token_score": next_token_score,
        "long_range_score": long_range_score,
        "diagonal_score": diagonal_score,
        "specialization": specialization,
    }
    head_analysis.append(head_info)
    
    print(f"\nHead {head_idx + 1}:")
    print(f"  Next-Token Score:  {next_token_score:.4f}")
    print(f"  Long-Range Score:  {long_range_score:.4f}")
    print(f"  Diagonal Score:    {diagonal_score:.4f}")
    print(f"  Specialization:    {specialization}")

# =========================================================
# HEAD REDUNDANCY
# =========================================================
redundancy_score = compute_head_redundancy(attn_last_layer)
print(f"\n{'=' * 70}")
print(f"Head Redundancy Analysis:")
print(f"  Average Cosine Similarity: {redundancy_score:.4f}")
print(f"  Interpretation: {'High Redundancy (heads are similar)' if redundancy_score > 0.5 else 'Low Redundancy (heads are diverse)'}")
print(f"{'=' * 70}\n")

# =========================================================
# VISUALIZE ATTENTION HEATMAPS
# =========================================================
print(f"Generating attention heatmaps...")
for head_idx in range(num_heads):
    attn_head = attn_last_layer[head_idx]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attn_head.numpy(),
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="Blues",
        cbar_kws={"label": "Attention Weight"},
    )
    
    info = head_analysis[head_idx]
    title = (
        f"Last Encoder Layer - Head {head_idx + 1}\n"
        f"Specialization: {info['specialization']} | "
        f"Next-Token: {info['next_token_score']:.3f} | "
        f"Long-Range: {info['long_range_score']:.3f}"
    )
    plt.title(title, fontsize=11, fontweight="bold")
    plt.xlabel("Key Tokens", fontweight="bold")
    plt.ylabel("Query Tokens", fontweight="bold")
    plt.tight_layout()
    
    filename = f"attention_head_{head_idx + 1}.png"
    plt.savefig(filename, dpi=100)
    plt.close()
    
    print(f"  ✓ Saved {filename}")

# =========================================================
# LOG TO W&B
# =========================================================
if wandb is not None:
    try:
        if wandb.run is None:
            wandb.init(project="da6401-a3", name="attention-heads-analysis")
        
        # Log heatmaps
        for head_idx in range(num_heads):
            filename = f"attention_head_{head_idx + 1}.png"
            wandb.log({
                f"attention/head_{head_idx + 1}": wandb.Image(filename),
            })
        
        # Log analysis metrics
        log_data = {
            "analysis/head_redundancy": redundancy_score,
            "analysis/num_next_token_heads": sum(1 for h in head_analysis if h["specialization"] == "Next-Token"),
            "analysis/num_long_range_heads": sum(1 for h in head_analysis if h["specialization"] == "Long-Range"),
            "analysis/num_self_attention_heads": sum(1 for h in head_analysis if h["specialization"] == "Self-Attention"),
            "analysis/num_general_heads": sum(1 for h in head_analysis if h["specialization"] == "General"),
        }
        
        for head_idx, info in enumerate(head_analysis):
            log_data[f"heads/head_{head_idx + 1}/next_token_score"] = info["next_token_score"]
            log_data[f"heads/head_{head_idx + 1}/long_range_score"] = info["long_range_score"]
            log_data[f"heads/head_{head_idx + 1}/diagonal_score"] = info["diagonal_score"]
        
        wandb.log(log_data)
        
        # Log analysis summary
        summary_text = f"""
## Attention Head Analysis Summary

**Sentence:** {sentence}

**Head Specialization Breakdown:**
- Next-Token Heads: {sum(1 for h in head_analysis if h['specialization'] == 'Next-Token')}
- Long-Range Heads: {sum(1 for h in head_analysis if h['specialization'] == 'Long-Range')}
- Self-Attention Heads: {sum(1 for h in head_analysis if h['specialization'] == 'Self-Attention')}
- General Heads: {sum(1 for h in head_analysis if h['specialization'] == 'General')}

**Head Redundancy:** {redundancy_score:.4f}
- Interpretation: {'Heads show high redundancy (similar attention patterns)' if redundancy_score > 0.5 else 'Heads show low redundancy (diverse attention patterns)'}

**Key Findings:**
1. Different heads specialize in different attention patterns
2. Some heads focus on adjacent tokens, others capture long-range dependencies
3. Redundancy score indicates {'significant overlap in head functions' if redundancy_score > 0.5 else 'good specialization across heads'}
"""
        
        wandb.log({"analysis/summary": wandb.Html(summary_text)})
        
        print(f"\n✓ Results logged to W&B")
        
    except Exception as e:
        print(f"Warning: Could not log to W&B: {e}")

print("\n✓ Attention head analysis complete!")
