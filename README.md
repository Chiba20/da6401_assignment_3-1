# DA6401 - Assignment 3: Implementing the Transformer for Machine Translation

## Overview

In this assignment, you will implement the landmark architecture from the paper "Attention Is All You Need" from scratch using PyTorch. The goal is to develop a Neural Machine Translation (NMT) system capable of translating text from German to English using the Multi30k dataset.

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

## Project Structure

```text
assignment3/
├── requirements.txt
├── README.md
├── model.py           # Core Transformer architecture (Encoders, Decoders, Multi-Head Attention)
├── utils.py           # Label Smoothing, Noam Scheduler, Masking Utilities
├── dataset.py         # Multi30k dataset loading and spacy tokenization
├── train.py           # Training loops and Greedy Decoding inference
```
