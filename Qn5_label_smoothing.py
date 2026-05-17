"""
Qn5: Decoder Sensitivity - Label Smoothing
Train two runs with label smoothing = 0.1 and 0.0 and log prediction confidence to W&B.
"""

import os
import math
from collections import deque

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset import Multi30kDataset, collate_batch
from model import Transformer, make_src_mask, make_tgt_mask
from lr_scheduler import NoamScheduler

try:
    import wandb
except ImportError:
    wandb = None


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


def prediction_confidence(model: Transformer, dataloader: DataLoader, device: str = "cpu") -> float:
    model.eval()
    total_conf = 0.0
    total_tokens = 0
    with torch.no_grad():
        for src, tgt in dataloader:
            src = src.to(device)
            tgt = tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            src_mask = make_src_mask(src, model.pad_idx)
            tgt_mask = make_tgt_mask(tgt_input, model.pad_idx)
            logits = model(src, tgt_input, src_mask, tgt_mask)
            probs = torch.softmax(logits, dim=-1)
            # gather probability of the correct token
            correct_probs = probs.gather(2, tgt_out.unsqueeze(2)).squeeze(2)
            mask = (tgt_out != model.pad_idx)
            total_conf += float((correct_probs * mask).sum().item())
            total_tokens += int(mask.sum().item())
    return total_conf / max(total_tokens, 1)


def run_experiment(smoothing: float, config: dict) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # datasets
    train_ds = Multi30kDataset("train", min_freq=config["min_freq"]) 
    val_ds = Multi30kDataset("validation", src_vocab=train_ds.src_vocab, tgt_vocab=train_ds.tgt_vocab)

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_batch)

    # model
    model = Transformer(src_vocab_size=len(train_ds.src_vocab), tgt_vocab_size=len(train_ds.tgt_vocab), d_model=config["d_model"], N=config["N"], num_heads=config["num_heads"], d_ff=config["d_ff"], dropout=config["dropout"], checkpoint_path=None).to(device)
    model.src_vocab = train_ds.src_vocab
    model.tgt_vocab = train_ds.tgt_vocab

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, d_model=config["d_model"], warmup_steps=config["warmup_steps"]) if config.get("use_noam", True) else None

    loss_fn = LabelSmoothingLoss(len(train_ds.tgt_vocab), model.pad_idx, smoothing=smoothing)

    history_conf = []

    run_name = f"label_smoothing_{smoothing}"
    use_wandb = config.get("use_wandb", True) and wandb is not None and os.environ.get("WANDB_MODE") != "disabled"

    if use_wandb:
        wandb.init(project=config.get("project", "da6401-a3"), name=run_name, config=dict(config, smoothing=smoothing))

    for epoch in range(config["num_epochs"]):
        model.train()
        for src, tgt in train_loader:
            src = src.to(device)
            tgt = tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            src_mask = make_src_mask(src, model.pad_idx)
            tgt_mask = make_tgt_mask(tgt_input, model.pad_idx)
            logits = model(src, tgt_input, src_mask, tgt_mask)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        val_conf = prediction_confidence(model, val_loader, device)
        history_conf.append(val_conf)
        print(f"smoothing={smoothing} epoch={epoch+1} val_conf={val_conf:.4f}")

        if use_wandb:
            wandb.log({"epoch": epoch+1, "val/pred_confidence": val_conf})

    # plot confidence curve
    plt.figure()
    plt.plot(range(1, len(history_conf)+1), history_conf, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Prediction Confidence")
    plt.title(f"Label Smoothing={smoothing}")
    plt.grid(True)
    fname = f"pred_conf_smoothing_{smoothing}.png"
    plt.savefig(fname)
    plt.close()

    if use_wandb:
        wandb.log({"plots/pred_confidence": wandb.Image(fname)})
        wandb.finish()


if __name__ == "__main__":
    config = {
        "project": "da6401-a3",
        "batch_size": 64,
        "num_epochs": 5,
        "d_model": 256,
        "N": 3,
        "num_heads": 8,
        "d_ff": 1024,
        "dropout": 0.1,
        "warmup_steps": 4000,
        "lr": 1.0,
        "min_freq": 2,
        "use_wandb": True,
        "use_noam": True,
    }
    # run both experiments
    run_experiment(0.1, config)
    run_experiment(0.0, config)
