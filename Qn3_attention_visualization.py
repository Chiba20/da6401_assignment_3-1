"""
<<<<<<< HEAD
Qn 2.3: Attention Rollout & Head Specialization (10 Marks)
Extract attention weights from LAST encoder layer and analyze head specialization
- Visualization: Heatmaps for each head
- Analysis: Identify head tasks, detect redundancy, long-range dependencies
=======
Qn2.2: Ablation Study - Scaling Factor (No Scaling Version)
Training script for transformer WITHOUT 1/√dk scaling to study vanishing gradients
>>>>>>> 97afd926d08462c5893bcb5e568f9bc0284bfaa8
"""

from collections import Counter
import math
import os
from typing import Optional

import torch
<<<<<<< HEAD
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.spatial.distance import cosine
=======
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import Multi30kDataset, collate_batch
from lr_scheduler import NoamScheduler
from model import Transformer, make_src_mask, make_tgt_mask, MultiHeadAttention

>>>>>>> 97afd926d08462c5893bcb5e568f9bc0284bfaa8

try:
    import wandb
except ImportError:
    wandb = None


def get_attention_modules(model: Transformer):
    return [
        module
        for module in model.modules()
        if isinstance(module, MultiHeadAttention)
    ]


<<<<<<< HEAD
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
=======
def compute_query_key_grad_norms(model: Transformer):
    """Compute L2 norms of Query and Key weight gradients"""
    q_norms = []
    k_norms = []
    for module in get_attention_modules(model):
        if module.w_q.weight.grad is not None:
            q_norms.append(float(module.w_q.weight.grad.norm(2).item()))
        else:
            q_norms.append(0.0)
        
        if module.w_k.weight.grad is not None:
            k_norms.append(float(module.w_k.weight.grad.norm(2).item()))
        else:
            k_norms.append(0.0)
    return q_norms, k_norms


class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=-1)
        with torch.no_grad():
            true_dist = torch.full_like(log_probs, self.smoothing / (self.vocab_size - 2))
            true_dist[:, self.pad_idx] = 0.0
            target_clamped = target.masked_fill(target == self.pad_idx, 0)
            true_dist.scatter_(1, target_clamped.unsqueeze(1), self.confidence)
            true_dist.masked_fill_((target == self.pad_idx).unsqueeze(1), 0.0)
        denom = (target != self.pad_idx).sum().clamp_min(1)
        return torch.sum(-true_dist * log_probs) / denom


def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
    grad_history: dict | None = None,
    global_step_start: int = 0,
    max_grad_steps: int = 1000,
    use_wandb: bool = False,
) -> tuple[float, int]:
    model.train(is_train)
    total_loss = 0.0
    total_tokens = 0
    global_step = global_step_start

    for src, tgt in data_iter:
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_input = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        src_mask = make_src_mask(src, model.pad_idx)
        tgt_mask = make_tgt_mask(tgt_input, model.pad_idx)

        logits = model(src, tgt_input, src_mask, tgt_mask)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

        if is_train:
            if optimizer is None:
                raise ValueError("optimizer is required when is_train=True")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            global_step += 1

            # Log gradient norms during first 1000 steps
            if grad_history is not None and global_step <= max_grad_steps:
                q_norms, k_norms = compute_query_key_grad_norms(model)
                q_mean = sum(q_norms) / max(len(q_norms), 1)
                k_mean = sum(k_norms) / max(len(k_norms), 1)
                
                grad_history["step"].append(global_step)
                grad_history["q_mean"].append(q_mean)
                grad_history["k_mean"].append(k_mean)
                
                if use_wandb and wandb is not None:
                    log_data = {
                        "step": global_step,
                        "train/grad_norm_Q_mean": q_mean,
                        "train/grad_norm_K_mean": k_mean,
                    }
                    for layer_idx, q_norm in enumerate(q_norms, start=1):
                        log_data[f"train/grad_norm_Q_layer_{layer_idx}"] = q_norm
                    for layer_idx, k_norm in enumerate(k_norms, start=1):
                        log_data[f"train/grad_norm_K_layer_{layer_idx}"] = k_norm
                    wandb.log(log_data)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        non_pad = (tgt_out != model.pad_idx).sum().item()
        total_loss += float(loss.item()) * max(non_pad, 1)
        total_tokens += max(non_pad, 1)
    return total_loss / max(total_tokens, 1), global_step


def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    model.eval()
    src = src.to(device)
    src_mask = src_mask.to(device)
    ys = torch.tensor([[start_symbol]], dtype=torch.long, device=device)
    with torch.no_grad():
        memory = model.encode(src, src_mask)
        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys, model.pad_idx)
            logits = model.decode(memory, src_mask, ys, tgt_mask)
            next_word = torch.argmax(logits[:, -1, :], dim=-1).view(1, 1)
            ys = torch.cat([ys, next_word], dim=1)
            if int(next_word.item()) == end_symbol:
                break
    return ys


def _lookup_token(vocab, idx: int) -> str:
    if hasattr(vocab, "lookup_token"):
        return vocab.lookup_token(int(idx))
    if hasattr(vocab, "itos"):
        return vocab.itos[int(idx)]
    return str(idx)


def _ids_to_tokens(ids, vocab, specials={1, 2, 3}):
    return [_lookup_token(vocab, int(i)) for i in ids if int(i) not in specials]


def _ngram_counts(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(max(len(tokens) - n + 1, 0)))


def _corpus_bleu(references: list[list[str]], hypotheses: list[list[str]], max_n: int = 4) -> float:
    matches = [0] * max_n
    totals = [0] * max_n
    ref_len = 0
    hyp_len = 0
    for ref, hyp in zip(references, hypotheses):
        ref_len += len(ref)
        hyp_len += len(hyp)
        for n in range(1, max_n + 1):
            ref_counts = _ngram_counts(ref, n)
            hyp_counts = _ngram_counts(hyp, n)
            matches[n - 1] += sum((hyp_counts & ref_counts).values())
            totals[n - 1] += sum(hyp_counts.values())
    if hyp_len == 0:
        return 0.0
    precisions = [(matches[i] + 1) / (totals[i] + 1) for i in range(max_n)]
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / hyp_len)
    return 100.0 * bp * math.exp(sum(math.log(p) for p in precisions) / max_n)


def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    model.eval()
    references = []
    hypotheses = []
    for src, tgt in test_dataloader:
        for i in range(src.size(0)):
            src_i = src[i : i + 1].to(device)
            src_mask = make_src_mask(src_i, model.pad_idx)
            pred = greedy_decode(model, src_i, src_mask, max_len, model.sos_idx, model.eos_idx, device)
            hypotheses.append(_ids_to_tokens(pred.squeeze(0).tolist(), tgt_vocab))
            references.append(_ids_to_tokens(tgt[i].tolist(), tgt_vocab))
    return _corpus_bleu(references, hypotheses)


def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "model_config": {
            "src_vocab_size": model.src_vocab_size,
            "tgt_vocab_size": model.tgt_vocab_size,
            "d_model": model.d_model,
            "N": model.N,
            "num_heads": model.num_heads,
            "d_ff": model.d_ff,
            "dropout": model.dropout_p,
            "max_len": model.max_len,
        },
        "src_stoi": getattr(model.src_vocab, "stoi", None),
        "tgt_stoi": getattr(model.tgt_vocab, "stoi", None),
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return int(checkpoint.get("epoch", 0))


def run_training_experiment() -> None:
    """Main training loop WITHOUT scaling factor (1/√dk)"""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    config = {
        "project": "da6401-a3",
        "run_name": "qn2_no_scaling",
        "batch_size": 64,
        "num_epochs": 20,
        "d_model": 256,
        "N": 3,
        "num_heads": 8,
        "d_ff": 1024,
        "dropout": 0.1,
        "warmup_steps": 4000,
        "lr": 1.0,
        "min_freq": 2,
        "checkpoint_path": "checkpoint_no_scaling.pt",
        "use_wandb": True,
        "use_scaling": False,  # KEY: disable 1/√dk scaling
    }

    use_wandb = (
        config["use_wandb"]
        and wandb is not None
        and os.environ.get("WANDB_MODE") != "disabled"
    )

    # Initialize W&B
    if use_wandb:
        wandb.init(
            project=config["project"],
            name=config["run_name"],
            config=config,
        )
        wandb.define_metric("step")
        wandb.define_metric("train/loss", step_metric="epoch")
        wandb.define_metric("val/loss", step_metric="epoch")
        wandb.define_metric("val/bleu", step_metric="epoch")
        wandb.define_metric("train/grad_norm_Q_mean", step_metric="step")
        wandb.define_metric("train/grad_norm_K_mean", step_metric="step")

    # Load datasets
    train_ds = Multi30kDataset("train", min_freq=config["min_freq"])
    val_ds = Multi30kDataset("validation", src_vocab=train_ds.src_vocab, tgt_vocab=train_ds.tgt_vocab)
    test_ds = Multi30kDataset("test", src_vocab=train_ds.src_vocab, tgt_vocab=train_ds.tgt_vocab)

    # Create dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_batch,
    )

    # Create model WITHOUT scaling (use_attention_scaling=False)
    model = Transformer(
        src_vocab_size=len(train_ds.src_vocab),
        tgt_vocab_size=len(train_ds.tgt_vocab),
        d_model=config["d_model"],
        N=config["N"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
        use_attention_scaling=False,  # KEY: No 1/√dk scaling
        checkpoint_path=None,
    ).to(device)

    model.src_vocab = train_ds.src_vocab
    model.tgt_vocab = train_ds.tgt_vocab

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        betas=(0.9, 0.98),
        eps=1e-9,
    )

    # Learning rate scheduler (Noam)
    scheduler = NoamScheduler(
        optimizer,
        d_model=config["d_model"],
        warmup_steps=config["warmup_steps"],
    )

    # Loss function
    loss_fn = LabelSmoothingLoss(
        len(train_ds.tgt_vocab),
        model.pad_idx,
        smoothing=0.1,
    )

    best_val = float("inf")
    
    # Gradient history for first 1000 steps
    grad_history = {
        "step": [],
        "q_mean": [],
        "k_mean": [],
    }
    global_step = 0

    # Training loop
    for epoch in range(config["num_epochs"]):
        print(f"\n--- Epoch {epoch + 1}/{config['num_epochs']} ---")
        
        # Training
        train_loss, global_step = run_epoch(
            train_loader,
            model,
            loss_fn,
            optimizer,
            scheduler,
            epoch,
            is_train=True,
            device=device,
            grad_history=grad_history,
            global_step_start=global_step,
            max_grad_steps=1000,
            use_wandb=use_wandb,
        )

        # Validation
        val_loss, _ = run_epoch(
            val_loader,
            model,
            loss_fn,
            optimizer=None,
            scheduler=None,
            epoch_num=epoch,
            is_train=False,
            device=device,
            grad_history=None,
            global_step_start=global_step,
            use_wandb=False,
        )

        # BLEU evaluation
        val_bleu = evaluate_bleu(model, val_loader, train_ds.tgt_vocab, device=device)

        print(f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_bleu={val_bleu:.2f}")

        # Log to W&B
        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "val/bleu": val_bleu,
                "train/lr": optimizer.param_groups[0]["lr"],
            })

        # Save best model
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                config["checkpoint_path"],
            )
            print("✓ Best checkpoint saved")
            
            if use_wandb:
                artifact = wandb.Artifact("best-no-scaling-checkpoint", type="model")
                artifact.add_file(config["checkpoint_path"])
                wandb.log_artifact(artifact)

    # Final test evaluation
    test_bleu = evaluate_bleu(model, test_loader, train_ds.tgt_vocab, device=device)
    print(f"\n✓ Final test BLEU: {test_bleu:.2f}")

    if use_wandb:
        wandb.log({
            "test/bleu": test_bleu,
            "best/val_loss": best_val,
        })
        wandb.finish()

    # Print gradient norm summary
    if grad_history["step"]:
        print(f"\nGradient Norm Summary (First 1000 steps):")
        print(f"Q mean norms: min={min(grad_history['q_mean']):.6f}, max={max(grad_history['q_mean']):.6f}")
        print(f"K mean norms: min={min(grad_history['k_mean']):.6f}, max={max(grad_history['k_mean']):.6f}")

if __name__ == "__main__":
    run_training_experiment()
>>>>>>> 97afd926d08462c5893bcb5e568f9bc0284bfaa8
