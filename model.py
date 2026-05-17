"""
Transformer architecture for DA6401 Assignment 3.

The default Transformer() constructor is intentionally usable by the
Gradescope-style inference contract. For meaningful translations, train the
model, upload the saved checkpoint/artifacts, and set CHECKPOINT_GDRIVE_ID.
"""

import copy
import math
import os
import re
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import gdown
except ImportError:  # Allows local architecture tests before requirements are installed.
    gdown = None


# After training, paste your Google Drive file id here, or set the env var.
CHECKPOINT_GDRIVE_ID = os.environ.get("DA6401_A3_CHECKPOINT_ID", "https://drive.google.com/file/d/1Uo31AvMjCPdRpZq8M39akwfeqJXbcup7/view?usp=drive_link")
DEFAULT_CHECKPOINT_PATH = os.environ.get("DA6401_A3_CHECKPOINT_PATH", "transformer_checkpoint.pt")


class SimpleVocab:
    def __init__(self, token_to_idx: Optional[dict[str, int]] = None) -> None:
        token_to_idx = token_to_idx or {"<unk>": 0, "<pad>": 1, "<sos>": 2, "<eos>": 3}
        self.stoi = dict(token_to_idx)
        self.itos = [None] * len(self.stoi)
        for tok, idx in self.stoi.items():
            if 0 <= idx < len(self.itos):
                self.itos[idx] = tok
        for i, tok in enumerate(self.itos):
            if tok is None:
                self.itos[i] = "<unk>"

    def __len__(self) -> int:
        return len(self.itos)

    def __getitem__(self, token: str) -> int:
        return self.lookup_indices([token])[0]

    def lookup_indices(self, tokens: list[str]) -> list[int]:
        unk = self.stoi.get("<unk>", 0)
        return [self.stoi.get(tok, unk) for tok in tokens]

    def lookup_token(self, idx: int) -> str:
        if 0 <= int(idx) < len(self.itos):
            return self.itos[int(idx)]
        return "<unk>"


def _tokenize_text(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)


def _state_dict_from_checkpoint(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
    attn_w = F.softmax(scores, dim=-1)
    attn_w = torch.nan_to_num(attn_w, nan=0.0)
    return torch.matmul(attn_w, V), attn_w


def make_src_mask(src: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    batch_size, tgt_len = tgt.shape
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)
    causal_mask = torch.triu(
        torch.ones((tgt_len, tgt_len), dtype=torch.bool, device=tgt.device),
        diagonal=1,
    ).unsqueeze(0).unsqueeze(0)
    return pad_mask.expand(batch_size, 1, tgt_len, tgt_len) | causal_mask


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self._split_heads(self.w_q(query))
        k = self._split_heads(self.w_k(key))
        v = self._split_heads(self.w_v(value))
        attn_out, attn_weights = scaled_dot_product_attention(q, k, v, mask)
        attn_out = torch.matmul(self.dropout(attn_weights), v)
        attn_out = attn_out.transpose(1, 2).contiguous()
        batch_size, seq_len, _, _ = attn_out.shape
        return self.w_o(attn_out.view(batch_size, seq_len, self.d_model))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)].to(dtype=x.dtype))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.dropout1(self.self_attn(x, x, x, src_mask)))
        return self.norm2(x + self.dropout2(self.feed_forward(x)))


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.norm1(x + self.dropout1(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout2(self.cross_attn(x, memory, memory, src_mask)))
        return self.norm3(x + self.dropout3(self.feed_forward(x)))


class Encoder(nn.Module):
    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):

    def __init__(
        self,
<<<<<<< HEAD
        src_vocab_size=10000,
        tgt_vocab_size=10000,
        d_model=256,
        N=6,
        num_heads=8,
        d_ff=1024,
        dropout=0.1,
        max_len=100,
        pad_idx=1,
        sos_idx=2,
        eos_idx=3,
        max_decode_len=50,
        checkpoint_path=DEFAULT_CHECKPOINT_PATH,
        checkpoint_gdrive_id=CHECKPOINT_GDRIVE_ID,
=======
        src_vocab_size: int = 10000,
        tgt_vocab_size: int = 10000,
        d_model: int = 512,
        N: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        checkpoint_path: str = 'transformer_checkpoint.pt',
        checkpoint_gdrive_id: str = "https://drive.google.com/file/d/1Uo31AvMjCPdRpZq8M39akwfeqJXbcup7/view?usp=drive_link",
        max_len: int = 5000,
        pad_idx: int = 1,
        sos_idx: int = 2,
        eos_idx: int = 3,
        max_decode_len: int = 100,
>>>>>>> ae621f8d7401fb01a32ef3902e3c8cf4cecb8e88
    ) -> None:

        super().__init__()

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.N = N
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_p = dropout
        self.max_len = max_len

        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.max_decode_len = max_decode_len

        # embeddings
        self.src_embed = nn.Embedding(
            src_vocab_size,
            d_model,
            padding_idx=pad_idx
        )

        self.tgt_embed = nn.Embedding(
            tgt_vocab_size,
            d_model,
            padding_idx=pad_idx
        )

        # positional encoding
        self.positional_encoding = PositionalEncoding(
            d_model,
            dropout,
            max_len
        )

        # encoder
        self.encoder = Encoder(
            EncoderLayer(
                d_model,
                num_heads,
                d_ff,
                dropout
            ),
            N
        )

        # decoder
        self.decoder = Decoder(
            DecoderLayer(
                d_model,
                num_heads,
                d_ff,
                dropout
            ),
            N
        )

        # output layer
        self.generator = nn.Linear(
            d_model,
            tgt_vocab_size
        )

        # vocab
        self.src_vocab = SimpleVocab()
        self.tgt_vocab = SimpleVocab()

        # initialize weights
        self._reset_parameters()

        # download checkpoint if needed
        if checkpoint_path and (
            not os.path.exists(checkpoint_path)
        ) and checkpoint_gdrive_id:

            self._download_checkpoint_if_needed(
                checkpoint_path,
                checkpoint_gdrive_id
            )

        # load checkpoint
        if checkpoint_path and os.path.exists(checkpoint_path):

            loaded = torch.load(
                checkpoint_path,
                map_location="cpu"
            )

            # load vocab
            self._load_artifacts(loaded)

            # load weights
            self.load_state_dict(
                _state_dict_from_checkpoint(loaded),
                strict=False
            )

    def _reset_parameters(self):

        for p in self.parameters():

            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _download_checkpoint_if_needed(
        self,
        checkpoint_path,
        drive_id
    ):

        if gdown is None:

            raise ImportError(
                "Install gdown first."
            )

        gdown.download(
            id=drive_id,
            output=checkpoint_path,
            quiet=False
        )

    def _load_artifacts(self, checkpoint):

        if not isinstance(checkpoint, dict):
            return

        src_stoi = checkpoint.get("src_stoi")
        tgt_stoi = checkpoint.get("tgt_stoi")

        if src_stoi:
            self.src_vocab = SimpleVocab(src_stoi)

        if tgt_stoi:
            self.tgt_vocab = SimpleVocab(tgt_stoi)

    def encode(
        self,
        src,
        src_mask
    ):

        x = self.src_embed(src) * math.sqrt(self.d_model)

        x = self.positional_encoding(x)

        return self.encoder(x, src_mask)

    def decode(
        self,
        memory,
        src_mask,
        tgt,
        tgt_mask
    ):

        x = self.tgt_embed(tgt) * math.sqrt(self.d_model)

        x = self.positional_encoding(x)

        dec = self.decoder(
            x,
            memory,
            src_mask,
            tgt_mask
        )

        return self.generator(dec)

    def forward(
        self,
        src,
        tgt,
        src_mask=None,
        tgt_mask=None
    ):

        if src_mask is None:
            src_mask = make_src_mask(
                src,
                self.pad_idx
            )

        if tgt_mask is None:
            tgt_mask = make_tgt_mask(
                tgt,
                self.pad_idx
            )

        memory = self.encode(
            src,
            src_mask
        )

        return self.decode(
            memory,
            src_mask,
            tgt,
            tgt_mask
        )

    def infer(
        self,
        src_sentence
    ):

        was_training = self.training

        self.eval()

        device = next(self.parameters()).device

        # tokenize
        tokens = (
            ["<sos>"]
            + _tokenize_text(src_sentence)
            + ["<eos>"]
        )

        # numericalize
        src_ids = self.src_vocab.lookup_indices(tokens)

        src = torch.tensor(
            src_ids,
            dtype=torch.long,
            device=device
        ).unsqueeze(0)

        # source mask
        src_mask = make_src_mask(
            src,
            self.pad_idx
        )

        # encoder output
        with torch.no_grad():

            memory = self.encode(
                src,
                src_mask
            )

        # decoder input starts with <sos>
        ys = torch.tensor(
            [[self.sos_idx]],
            dtype=torch.long,
            device=device
        )

        # greedy decoding
        with torch.no_grad():

            for _ in range(
                self.max_decode_len
            ):

                tgt_mask = make_tgt_mask(
                    ys,
                    self.pad_idx
                )

                out = self.decode(
                    memory,
                    src_mask,
                    ys,
                    tgt_mask
                )

                next_word = torch.argmax(
                    out[:, -1],
                    dim=-1
                ).item()

                ys = torch.cat(
                    [
                        ys,
                        torch.tensor(
                            [[next_word]],
                            device=device
                        )
                    ],
                    dim=1
                )

                if next_word == self.eos_idx:
                    break

        # ids -> tokens
        output_tokens = []

        for idx in ys.squeeze(0).tolist():

            if idx in (
                self.sos_idx,
                self.eos_idx,
                self.pad_idx
            ):
                continue

            output_tokens.append(
                self.tgt_vocab.lookup_token(idx)
            )

        if was_training:
            self.train()

        return self._detokenize(output_tokens)

    @staticmethod
    def _detokenize(tokens):

        text = " ".join(tokens)

        text = re.sub(
            r"\s+([.,!?;:%)\]}])",
            r"\1",
            text
        )

        return text.strip()
