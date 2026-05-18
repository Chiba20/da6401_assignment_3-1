"""
Scaling factor ablation (with and without 1/sqrt(d_k)) for DA6401 Assignment 3.
This script trains two Transformer variants, logs gradient norms for Q and K weights,
and produces comparison plots for W&B.
"""

from collections import Counter
import math
import os
from typing import Optional

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import Multi30kDataset, collate_batch
from lr_scheduler import NoamScheduler
from model import Transformer, make_src_mask, make_tgt_mask, MultiHeadAttention

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


def compute_query_key_grad_norms(model: Transformer):
    q_norms = []
    k_norms = []
    for module in get_attention_modules(model):
        q_grad = module.w_q.weight.grad
        k_grad = module.w_k.weight.grad
        q_norms.append(float(q_grad.norm(2).item()) if q_grad is not None else 0.0)
        k_norms.append(float(k_grad.norm(2).item()) if k_grad is not None else 0.0)
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

            if grad_history is not None and global_step <= max_grad_steps:
                q_norms, k_norms = compute_query_key_grad_norms(model)
                q_mean = sum(q_norms) / max(len(q_norms), 1)
                k_mean = sum(k_norms) / max(len(k_norms), 1)
                grad_history["step"].append(global_step)
                grad_history["q_mean"].append(q_mean)
                grad_history["k_mean"].append(k_mean)
                if use_wandb:
                    log_data = {
                        "global_step": global_step,
                        "grad_norm/Q/mean": q_mean,
                        "grad_norm/K/mean": k_mean,
                    }
                    for layer_index, norm in enumerate(q_norms, start=1):
                        log_data[f"grad_norm/Q/layer_{layer_index}"] = norm
                    for layer_index, norm in enumerate(k_norms, start=1):
                        log_data[f"grad_norm/K/layer_{layer_index}"] = norm
                    wandb.log(log_data, step=global_step)

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


def run_single_variant(config: dict, use_scaling: bool, run_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    use_wandb = (
        config["use_wandb"]
        and wandb is not None
        and os.environ.get("WANDB_MODE") != "disabled"
    )

    if use_wandb:
        wandb.init(
            project=config["project"],
            name=run_name,
            config=dict(config, use_attention_scaling=use_scaling),
        )
        wandb.define_metric("epoch")
        wandb.define_metric("global_step", step_metric=True)
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("val/*", step_metric="epoch")
        wandb.define_metric("grad_norm/*", step_metric="global_step")
        wandb.define_metric("test/*")

    train_ds = Multi30kDataset("train", min_freq=config["min_freq"])
    val_ds = Multi30kDataset("validation", src_vocab=train_ds.src_vocab, tgt_vocab=train_ds.tgt_vocab)
    test_ds = Multi30kDataset("test", src_vocab=train_ds.src_vocab, tgt_vocab=train_ds.tgt_vocab)

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_batch)

    model = Transformer(
        src_vocab_size=len(train_ds.src_vocab),
        tgt_vocab_size=len(train_ds.tgt_vocab),
        d_model=config["d_model"],
        N=config["N"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
        checkpoint_path=None,
        use_attention_scaling=use_scaling,
    ).to(device)

    model.src_vocab = train_ds.src_vocab
    model.tgt_vocab = train_ds.tgt_vocab

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, d_model=config["d_model"], warmup_steps=config["warmup_steps"])
    loss_fn = LabelSmoothingLoss(len(train_ds.tgt_vocab), model.pad_idx, smoothing=0.1)

    grad_history = {"step": [], "q_mean": [], "k_mean": []}
    val_bleu_history = []
    global_step = 0
    best_val = float("inf")

    for epoch in range(config["num_epochs"]):
        train_loss, global_step = run_epoch(
            train_loader,
            model,
            loss_fn,
            optimizer,
            scheduler,
            epoch,
            True,
            device,
            grad_history,
            global_step,
            max_grad_steps=1000,
            use_wandb=use_wandb,
        )

        val_loss, _ = run_epoch(
            val_loader,
            model,
            loss_fn,
            None,
            None,
            epoch,
            False,
            device,
        )

        val_bleu = evaluate_bleu(model, val_loader, train_ds.tgt_vocab, device=device)
        val_bleu_history.append(val_bleu)

        print(
            f"[{run_name}] epoch={epoch + 1} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_bleu={val_bleu:.2f}"
        )

        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "val/bleu": val_bleu,
                "train/lr": optimizer.param_groups[0]["lr"],
            }, step=epoch + 1)

        if val_loss < best_val:
            best_val = val_loss
            checkpoint_name = f"{run_name}_checkpoint.pt"
            save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_name)
            print("best checkpoint saved")
            if use_wandb:
                artifact = wandb.Artifact(f"best-{run_name}-checkpoint", type="model")
                artifact.add_file(checkpoint_name)
                wandb.log_artifact(artifact)

    test_bleu = evaluate_bleu(model, test_loader, train_ds.tgt_vocab, device=device)
    print(f"[{run_name}] test_bleu={test_bleu:.2f}")

    if use_wandb:
        wandb.log({"test/bleu": test_bleu, "best/val_loss": best_val})
        wandb.finish()

    return {
        "run_name": run_name,
        "val_bleu": val_bleu_history,
        "grad_history": grad_history,
        "test_bleu": test_bleu,
    }


def plot_ablation_results(scaling_data: dict, no_scaling_data: dict, project_name: str):
    import matplotlib.pyplot as plt

    epochs = list(range(1, len(scaling_data["val_bleu"]) + 1))
    plt.figure()
    plt.plot(epochs, scaling_data["val_bleu"], marker="o", label="With scaling")
    plt.plot(epochs, no_scaling_data["val_bleu"], marker="o", label="Without scaling")
    plt.xlabel("Epoch")
    plt.ylabel("Validation BLEU")
    plt.title("Scaling Factor Ablation: Validation BLEU")
    plt.legend()
    plt.grid(True)
    bleu_path = "scaling_ablation_bleu.png"
    plt.savefig(bleu_path)
    plt.close()

    steps = scaling_data["grad_history"]["step"]
    plt.figure(figsize=(10, 6))
    plt.plot(steps, scaling_data["grad_history"]["q_mean"], marker=".", label="Q norm with scaling")
    plt.plot(steps, no_scaling_data["grad_history"]["q_mean"], marker=".", label="Q norm without scaling")
    plt.plot(steps, scaling_data["grad_history"]["k_mean"], marker=".", linestyle="--", label="K norm with scaling")
    plt.plot(steps, no_scaling_data["grad_history"]["k_mean"], marker=".", linestyle="--", label="K norm without scaling")
    plt.xlabel("Training Step")
    plt.ylabel("Gradient Norm")
    plt.title("Gradient Norms for Q and K Weight Matrices (First 1000 Steps)")
    plt.legend()
    plt.grid(True)
    grad_path = "scaling_ablation_gradnorm.png"
    plt.savefig(grad_path)
    plt.close()

    if wandb is not None:
        try:
            wandb.init(project=project_name, name="scaling_ablation_comparison")
            wandb.log({
                "ablation/bleu_comparison": wandb.Image(bleu_path),
                "ablation/gradnorm_comparison": wandb.Image(grad_path),
            })
            wandb.finish()
        except Exception:
            pass

    print("Saved ablation comparison plots:", bleu_path, grad_path)


def run_training_experiment() -> None:
    config = {
        "project": "da6401-a3",
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
        "use_wandb": True,
    }

    scaling_data = run_single_variant(config, True, "scaling")
    no_scaling_data = run_single_variant(config, False, "no_scaling")
    plot_ablation_results(scaling_data, no_scaling_data, config["project"])


if __name__ == "__main__":
    run_training_experiment()
