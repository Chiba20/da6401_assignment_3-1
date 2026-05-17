"""
Question 2 experiment: train the Transformer without attention scaling.

This script keeps the baseline Noam learning-rate schedule, but replaces
scaled dot-product attention with unscaled dot-product attention.
"""

import os
from typing import Optional

import torch
from torch.utils.data import DataLoader

from dataset import Multi30kDataset, collate_batch
from lr_scheduler import NoamScheduler
from model import Transformer, MultiHeadAttention
from train import LabelSmoothingLoss, evaluate_bleu, run_epoch, save_checkpoint

try:
    import wandb
except ImportError:
    wandb = None


def no_scaling_forward(
    self: MultiHeadAttention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    q = self._split_heads(self.w_q(query))
    k = self._split_heads(self.w_k(key))
    v = self._split_heads(self.w_v(value))

    scores = torch.matmul(q, k.transpose(-2, -1))
    if mask is not None:
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)

    attn_weights = torch.softmax(scores, dim=-1)
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
    if self.training:
        attn_weights = self.dropout(attn_weights)

    attn_out = torch.matmul(attn_weights, v)
    attn_out = attn_out.transpose(1, 2).contiguous()
    batch_size, seq_len, _, _ = attn_out.shape
    return self.w_o(attn_out.view(batch_size, seq_len, self.d_model))


MultiHeadAttention.forward = no_scaling_forward


def run_training_experiment() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = {
        "project": "da6401-a3",
        "run_name": "q2_no_attention_scaling",
        "experiment": "no_attention_scaling",
        "attention_scaling": False,
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
        "checkpoint_path": "q2_no_scaling_checkpoint.pt",
        "use_wandb": True,
    }

    use_wandb = (
        config["use_wandb"]
        and wandb is not None
        and os.environ.get("WANDB_MODE") != "disabled"
    )

    if use_wandb:
        wandb.init(
            project=config["project"],
            name=config["run_name"],
            config=config,
        )
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("val/*", step_metric="epoch")
        wandb.define_metric("test/*")

    train_ds = Multi30kDataset("train", min_freq=config["min_freq"])
    val_ds = Multi30kDataset(
        "validation",
        src_vocab=train_ds.src_vocab,
        tgt_vocab=train_ds.tgt_vocab,
    )
    test_ds = Multi30kDataset(
        "test",
        src_vocab=train_ds.src_vocab,
        tgt_vocab=train_ds.tgt_vocab,
    )

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

    model = Transformer(
        src_vocab_size=len(train_ds.src_vocab),
        tgt_vocab_size=len(train_ds.tgt_vocab),
        d_model=config["d_model"],
        N=config["N"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
        checkpoint_path=None,
    ).to(device)
    model.src_vocab = train_ds.src_vocab
    model.tgt_vocab = train_ds.tgt_vocab

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    scheduler = NoamScheduler(
        optimizer,
        d_model=config["d_model"],
        warmup_steps=config["warmup_steps"],
    )
    loss_fn = LabelSmoothingLoss(
        len(train_ds.tgt_vocab),
        model.pad_idx,
        smoothing=0.1,
    )

    best_val = float("inf")

    for epoch in range(config["num_epochs"]):
        train_loss = run_epoch(
            train_loader,
            model,
            loss_fn,
            optimizer,
            scheduler,
            epoch,
            True,
            device,
        )
        val_loss = run_epoch(
            val_loader,
            model,
            loss_fn,
            None,
            None,
            epoch,
            False,
            device,
        )
        val_bleu = evaluate_bleu(
            model,
            val_loader,
            train_ds.tgt_vocab,
            device=device,
        )

        print(
            f"epoch={epoch + 1} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_bleu={val_bleu:.2f}"
        )

        if use_wandb:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "val/bleu": val_bleu,
                    "train/lr": optimizer.param_groups[0]["lr"],
                }
            )

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                config["checkpoint_path"],
            )
            print(f"best checkpoint saved to {config['checkpoint_path']}")

            if use_wandb:
                artifact = wandb.Artifact(
                    "q2-no-scaling-checkpoint",
                    type="model",
                    metadata={
                        "epoch": epoch + 1,
                        "val_loss": val_loss,
                        "attention_scaling": False,
                    },
                )
                artifact.add_file(config["checkpoint_path"])
                wandb.log_artifact(artifact, aliases=["best", f"epoch-{epoch + 1}"])

    test_bleu = evaluate_bleu(
        model,
        test_loader,
        train_ds.tgt_vocab,
        device=device,
    )
    print(f"test_bleu={test_bleu:.2f}")

    if use_wandb:
        wandb.log(
            {
                "test/bleu": test_bleu,
                "best/val_loss": best_val,
            }
        )
        wandb.finish()


if __name__ == "__main__":
    run_training_experiment()
