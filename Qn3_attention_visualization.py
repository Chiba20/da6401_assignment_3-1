"""
Question 2.3
Attention Head Visualization
"""

import math
import torch
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import wandb
except Exception:
    wandb = None

from model import Transformer, make_src_mask
from dataset import Multi30kDataset


# =========================================================
# LOAD DATASET
# =========================================================

train_ds = Multi30kDataset("train")


# =========================================================
# LOAD MODEL
# =========================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Transformer(
    src_vocab_size=len(train_ds.src_vocab),
    tgt_vocab_size=len(train_ds.tgt_vocab),
    checkpoint_path="transformer_checkpoint.pt"
).to(device)

model.eval()

model.src_vocab = train_ds.src_vocab
model.tgt_vocab = train_ds.tgt_vocab


# =========================================================
# STORE ATTENTION WEIGHTS
# =========================================================

attention_maps = []


def save_attention_hook(module, input, output):

    if hasattr(module, "attention_weights"):

        attention_maps.append(
            module.attention_weights.detach().cpu()
        )


# =========================================================
# REGISTER HOOKS
# =========================================================

for layer in model.encoder.layers:

    layer.self_attn.register_forward_hook(
        save_attention_hook
    )


# =========================================================
# INPUT SENTENCE
# =========================================================

sentence = "ein mann spielt gitarre"

tokens = (
    ["<sos>"]
    + sentence.split()
    + ["<eos>"]
)

src_ids = train_ds.src_vocab.lookup_indices(tokens)

src_tensor = torch.tensor(
    src_ids,
    dtype=torch.long
).unsqueeze(0).to(device)

src_mask = make_src_mask(
    src_tensor,
    model.pad_idx
)


# =========================================================
# RUN ENCODER
# =========================================================

with torch.no_grad():

    model.encode(
        src_tensor,
        src_mask
    )


# =========================================================
# VISUALIZE ATTENTION
# =========================================================

for layer_idx, attn in enumerate(attention_maps):

    # shape:
    # [batch, heads, seq, seq]

    num_heads = attn.shape[1]

    for head in range(num_heads):

        plt.figure(figsize=(8, 6))

        sns.heatmap(

            attn[0, head],

            xticklabels=tokens,

            yticklabels=tokens,

            cmap="viridis"
        )

        plt.title(
            f"Encoder Layer {layer_idx + 1} "
            f"Head {head + 1}"
        )

        plt.xlabel("Key Tokens")

        plt.ylabel("Query Tokens")

        plt.tight_layout()

        filename = (
            f"encoder_layer"
            f"{layer_idx+1}_head"
            f"{head+1}.png"
        )

        plt.savefig(filename)

        plt.close()

        print(f"saved: {filename}")
        # log to wandb if available
        if wandb is not None and wandb.run is None:
            try:
                wandb.init(project="da6401-a3", name="attention-visualization")
            except Exception:
                pass

        if wandb is not None and wandb.run is not None:
            try:
                wandb.log({
                    f"encoder/layer{layer_idx+1}/head{head+1}": wandb.Image(filename)
                })
            except Exception:
                pass


print("\nAttention visualization complete.")