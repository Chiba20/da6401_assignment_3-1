from collections import Counter
from typing import Iterable

import torch
from torch.utils.data import Dataset

from model import SimpleVocab, _tokenize_text


SPECIALS = ["<unk>", "<pad>", "<sos>", "<eos>"]


class Multi30kDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        src_vocab: SimpleVocab | None = None,
        tgt_vocab: SimpleVocab | None = None,
        min_freq: int = 2,
        max_size: int | None = None,
    ) -> None:
        self.split = split
        self.min_freq = min_freq
        self.max_size = max_size
        self.raw_data = self._load_split(split)
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        if self.src_vocab is None or self.tgt_vocab is None:
            self.build_vocab()
        self.examples = self.process_data()

    def _load_split(self, split: str) -> list[tuple[str, str]]:
        from datasets import load_dataset

        ds = load_dataset("bentrevett/multi30k", split=split)
        pairs = []
        for row in ds:
            de = row.get("de") or row.get("deu") or row.get("german")
            en = row.get("en") or row.get("eng") or row.get("english")
            if de is None and isinstance(row.get("translation"), dict):
                de = row["translation"].get("de")
                en = row["translation"].get("en")
            if de is None or en is None:
                values = list(row.values())
                de, en = values[0], values[1]
            pairs.append((str(de), str(en)))
        return pairs

    def _make_vocab(self, token_iter: Iterable[list[str]]) -> SimpleVocab:
        counter = Counter()
        for tokens in token_iter:
            counter.update(tokens)
        stoi = {tok: idx for idx, tok in enumerate(SPECIALS)}
        words = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
        for word, freq in words:
            if freq < self.min_freq:
                continue
            if word in stoi:
                continue
            if self.max_size is not None and len(stoi) >= self.max_size:
                break
            stoi[word] = len(stoi)
        return SimpleVocab(stoi)

    def build_vocab(self):
        self.src_vocab = self._make_vocab(_tokenize_text(de) for de, _ in self.raw_data)
        self.tgt_vocab = self._make_vocab(_tokenize_text(en) for _, en in self.raw_data)
        return self.src_vocab, self.tgt_vocab

    def process_data(self):
        examples = []
        for de, en in self.raw_data:
            src_tokens = ["<sos>"] + _tokenize_text(de) + ["<eos>"]
            tgt_tokens = ["<sos>"] + _tokenize_text(en) + ["<eos>"]
            examples.append(
                (
                    torch.tensor(self.src_vocab.lookup_indices(src_tokens), dtype=torch.long),
                    torch.tensor(self.tgt_vocab.lookup_indices(tgt_tokens), dtype=torch.long),
                )
            )
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        return self.examples[idx]


def collate_batch(batch, pad_idx: int = 1):
    src_batch, tgt_batch = zip(*batch)
    src = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    tgt = torch.nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)
    return src, tgt
