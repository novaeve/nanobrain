"""Independent candidate training; only an identified candidate's own run can resume."""

import json
from pathlib import Path

from .contracts import digest, resolved_document, save


def identity_for(bundle, cfg, seed, *, device="cuda"):
    if seed not in cfg["seeds"]:
        raise ValueError("seed is outside this campaign")
    document = resolved_document(bundle, cfg, seed)
    return {
        "version": 1,
        "protocol_id": cfg["protocol_id"],
        "candidate_id": digest(bundle),
        "seed": seed,
        "device": device,
        "weight_policy": cfg["weight_policy"],
        "training_document_sha256": digest(document),
        "external_weight_dependencies": [],
    }


def validate_identity(receipt, bundle, cfg, seed, *, device="cuda"):
    if (
        receipt.get("weight_policy") != cfg["weight_policy"]
        or receipt.get("external_weight_dependencies") != []
        or receipt.get("training_identity") != identity_for(bundle, cfg, seed, device=device)
    ):
        raise ValueError("cached training does not attest independent training of this candidate")


def prepare_training(directory, bundle, cfg, seed, *, device="cuda"):
    directory = Path(directory)
    identity = identity_for(bundle, cfg, seed, device=device)
    path = directory / "training-identity.json"
    checkpoint = directory / "train/latest.pt"
    if path.exists():
        if json.loads(path.read_text()) != identity:
            raise ValueError("resume checkpoint belongs to a different candidate, seed or training recipe")
    elif checkpoint.exists():
        raise ValueError("refuse an unidentified checkpoint; independent training must start from random weights")
    else:
        save(path, identity)
    return ["--resume"] if checkpoint.exists() else []
