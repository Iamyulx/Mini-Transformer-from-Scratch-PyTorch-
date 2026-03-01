# Mini-Transformer-from-Scratch-PyTorch-
Educational PyTorch implementation of a mini Transformer Encoder, built from scratch following the architecture introduced in Attention Is All You Need to demonstrate attention mechanisms and encoder design.
This repository contains a **from-scratch implementation** of a Transformer Encoder using PyTorch.

## Features
- Scaled Dot-Product Attention
- Multi-Head Attention
- Positional Encoding (sinusoidal)
- Encoder Block (Attention + Feed Forward)
- Full Transformer Encoder stack

## Model Output
The encoder returns a tensor with shape:

(batch_size, sequence_length, d_model)

Example output:
torch.Size([4, 10, 128])

## Example Usage

```python
import torch
from mini_transformer import TransformerEncoder

vocab_size = 1000
batch_size = 4
seq_len = 10

model = TransformerEncoder(vocab_size)
x = torch.randint(0, vocab_size, (batch_size, seq_len))
output = model(x)

print(output.shape)
