#!/usr/bin/env python3
"""
Train a tiny character-level language model with LogNN.

Model:
  token embedding -> causal transformer encoder -> linear vocab head

Usage (from repo root, after building lognn):
  PYTHONPATH=. python3 sample_tiny_lm.py
  PYTHONPATH=. python3 sample_tiny_lm.py --device cpu --steps 600
"""

from __future__ import annotations

import argparse
import random
import sys

import lognn


def build_vocab(text: str) -> tuple[dict[str, int], list[str]]:
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = chars
    return stoi, itos


def encode(text: str, stoi: dict[str, int]) -> list[int]:
    return [stoi[ch] for ch in text]


def decode(ids: list[int], itos: list[str]) -> str:
    return "".join(itos[i] for i in ids)


def make_batch(
    token_ids: list[int],
    seq_len: int,
    device: str,
    device_index: int,
) -> tuple[lognn.Tensor, lognn.Tensor]:
    if len(token_ids) <= seq_len + 1:
        raise ValueError("Corpus too small for requested --seq-len")
    start = random.randint(0, len(token_ids) - seq_len - 1)
    x_ids = token_ids[start : start + seq_len]
    y_ids = token_ids[start + 1 : start + seq_len + 1]
    x = lognn.Tensor.from_data([seq_len], [float(t) for t in x_ids], device, device_index)
    y = lognn.Tensor.from_data([seq_len], [float(t) for t in y_ids], device, device_index)
    return x, y


def argmax(values: list[float]) -> int:
    best_i = 0
    best_v = values[0]
    for i in range(1, len(values)):
        if values[i] > best_v:
            best_v = values[i]
            best_i = i
    return best_i


def generate_text(
    embed: lognn.nn.Embedding,
    encoder: lognn.nn.CausalTransformerEncoder,
    lm_head: lognn.nn.Linear,
    stoi: dict[str, int],
    itos: list[str],
    prompt: str,
    max_new_tokens: int,
    device: str,
    device_index: int,
) -> str:
    if not prompt:
        prompt = itos[0]
    ids = [stoi[ch] for ch in prompt if ch in stoi]
    if not ids:
        ids = [0]

    embed.eval()
    encoder.eval()
    lm_head.eval()

    for _ in range(max_new_tokens):
        # stop generation if generation repeats a token too many times
        if len(ids) > 5:
            if ids[-1] == ids[-2] and ids[-2] == ids[-3] and ids[-3] == ids[-4] and ids[-4] == ids[-5]:
                break
        x = lognn.Tensor.from_data([len(ids)], [float(t) for t in ids], device, device_index)
        h = embed.forward_from_indices(x)
        h = encoder.forward(h)
        logits = lm_head.forward(h).data().get_data()
        vocab_size = len(itos)
        last_row = logits[-vocab_size:]
        next_id = argmax(last_row)
        ids.append(next_id)

    return decode(ids, itos)


def main() -> None:
    p = argparse.ArgumentParser(description="Train a tiny LogNN language model.")
    p.add_argument("--device", type=str, default="cpu", help="cpu, cuda, or mlx")
    p.add_argument("--device-index", type=int, default=0)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--d-model", type=int, default=48)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--prompt", type=str, default="what is lognn?")
    p.add_argument("--generate", type=int, default=120, help="Number of new tokens to generate")
    args = p.parse_args()

    random.seed(args.seed)

    corpus = (
        "lognn is a tiny deep learning library.\n"
        "it has tensors, autograd, modules, and optimizers.\n"
        "this sample trains a very small language model.\n"
        "the model learns to predict the next character.\n"
    )
    # Repeat corpus to create enough training windows while staying lightweight.
    train_text = corpus * 80

    stoi, itos = build_vocab(train_text)
    token_ids = encode(train_text, stoi)
    vocab_size = len(itos)

    embed = lognn.nn.Embedding(vocab_size, args.d_model, args.device, args.device_index)
    encoder = lognn.nn.CausalTransformerEncoder(
        args.layers, args.d_model, args.dropout, args.device, args.device_index
    )
    lm_head = lognn.nn.Linear(args.d_model, vocab_size, args.device, args.device_index)

    params = embed.parameters() + encoder.parameters() + lm_head.parameters()
    opt = lognn.optim.Adam(params, args.lr)

    embed.train()
    encoder.train()
    lm_head.train()

    first_loss = None
    last_loss = None
    for step in range(args.steps):
        x_t, y_t = make_batch(token_ids, args.seq_len, args.device, args.device_index)
        h = embed.forward_from_indices(x_t)
        h = encoder.forward(h)
        logits = lm_head.forward(h)
        loss = lognn.Variable.cross_entropy_next_token_lm(logits, y_t)

        if first_loss is None:
            first_loss = loss.data().get_data()[0]
        last_loss = loss.data().get_data()[0]

        opt.zero_grad()
        loss.backward()
        opt.step()

        if args.log_every > 0 and (step + 1) % args.log_every == 0:
            print(f"step {step + 1:4d}  loss: {last_loss:.6f}")

    print("initial loss:", first_loss)
    print("final loss:  ", last_loss)

    if first_loss is None or last_loss is None or not (last_loss < first_loss):
        print("Loss did not decrease. Try more --steps or a smaller --lr.", file=sys.stderr)

    generated = generate_text(
        embed=embed,
        encoder=encoder,
        lm_head=lm_head,
        stoi=stoi,
        itos=itos,
        prompt=args.prompt,
        max_new_tokens=args.generate,
        device=args.device,
        device_index=args.device_index,
    )
    print("\n--- generated text ---")
    print(generated)


if __name__ == "__main__":
    main()
