"""Predict each spatial center's neighborhood composition from its expression alone."""

import argparse
from pathlib import Path

import numpy as np
import torch
from eval_niche import N_TYPES, read_spatial_rows

from brainwm_unified import infer
from brainwm_unified.config import DataConfig
from brainwm_unified.data import Subset


@torch.inference_mode()
def predict(model, data, rows, token_budget, seed):
    head = getattr(model, "niche_head", None)
    if head is None:
        raise ValueError("checkpoint does not contain a trained niche head")
    device = next(model.parameters()).device
    ds = Subset(str(data))
    embeddings = infer.predict(
        model,
        ds,
        rows,
        DataConfig(root=str(data)),
        token_budget,
        num_workers=8 if device.type == "cuda" else 0,
        seed=seed,
    )
    if not np.array_equal(embeddings["cells"], rows):
        raise ValueError("checkpoint embeddings do not match the ordered spatial rows")
    logits = head(torch.as_tensor(embeddings["mu"], device=device)).float()
    if logits.shape != (len(rows), N_TYPES) or not torch.isfinite(logits).all():
        raise ValueError(f"niche head must produce {N_TYPES} finite logits per spatial center")
    return logits.softmax(dim=-1).cpu().numpy()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--token-budget", type=int, default=1 << 19)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.token_budget <= 0 or args.seed < 0:
        parser.error("token budget must be positive and seed must be nonnegative")
    rows, _, source = read_spatial_rows(args.data)
    model, state = infer.load_model(args.checkpoint, device=args.device)
    probabilities = predict(model, args.data, rows, args.token_budget, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            task="niche",
            rows=rows,
            source_rows=source,
            probabilities=probabilities,
            seed=args.seed,
            step=state["step"],
        )
    temporary.replace(args.out)
    print(f"niche: {len(rows)} centers (step {state['step']}) -> {args.out}")


if __name__ == "__main__":
    main()
