"""Mask as task: every pretraining task is a mask over the Cell x Gene table.

A cell's row holds one token per gene it became (non-zero genes, capped, plus sampled
measured zeros). Each token is in one of three states:

  VISIBLE  its value is input
  TARGET   its value is hidden from the input (the gene id stays), predicted and scored
  HIDDEN   dropped: neither input nor scored

Genes the cell's panel did not measure (MISSING) never become tokens. Label columns
(cell type, anatomy) are always targets and the batch column is hidden from the encoder;
those live in the objectives. Spatial groups get a row mask: which member cells are
targets for the neighbourhood model.

A spec decides the states of one row (CellSpec) or one group (GroupSpec). A Mixture draws
one spec per row or group, with weights that may change by training stage (curriculum).
Plans live in configs/masks/*.toml and name registered specs. The "cajal_v0" plan
reproduces the original batches exactly: its specs consume the random generator in the
same order, and a one-spec mixture draws nothing.
"""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .registry import Registry

# Row specs decide one cell's token states; group specs decide which members of a spatial group are targets.
ROW_MASKS = Registry("row mask")
GROUP_MASKS = Registry("group mask")
MASK_CONFIGS = Path(__file__).resolve().parent.parent / "configs" / "masks"

VISIBLE, TARGET, HIDDEN = 0, 1, 2


@dataclass
class Row:
    """One cell's tokens before masking (CLS excluded)."""

    genes: np.ndarray  # gene ids
    values: np.ndarray  # normalised input values (log1p CP10K for counts)
    raw: np.ndarray | None  # raw counts of the tokens (0 for sampled zeros); None if not counts
    lib: float  # the cell's total count
    whole_transcriptome: bool  # its panel measures (almost) every gene


@dataclass
class Resources:
    """Data the richer specs need; empty for cajal_v0."""

    gene_program: np.ndarray | None = None  # (n_genes,) program id per gene, -1 if none
    panels: list = field(default_factory=list)  # gene sets of targeted-panel platforms (e.g. MERFISH)


def _random_targets(n: int, ratio: float, rng) -> np.ndarray:
    """Exactly the original draw: max(1, round(ratio * n)) distinct positions."""
    states = np.zeros(n, np.int8)
    states[rng.choice(n, size=min(max(1, round(ratio * n)), n), replace=False)] = TARGET
    return states


# Row specs: (row, rng, res) -> (states, input values or None to keep row.values)


@ROW_MASKS.register("random_genes")
@dataclass
class RandomGenes:
    """Scattered genes: imputation and denoising."""

    ratio: float = 0.15

    def __call__(self, row: Row, rng, res: Resources):
        return _random_targets(len(row.genes), self.ratio, rng), None


@ROW_MASKS.register("gene_program")
@dataclass
class GeneProgram:
    """Whole co-expressed gene programs: infer one program from the others (the table's span mask)."""

    ratio: float = 0.15
    fallback: RandomGenes = field(default_factory=RandomGenes)

    def __call__(self, row: Row, rng, res: Resources):
        prog = res.gene_program[row.genes] if res.gene_program is not None else np.full(len(row.genes), -1)
        present = np.unique(prog[prog >= 0])
        if len(present) == 0:
            return self.fallback(row, rng, res)
        states = np.zeros(len(row.genes), np.int8)
        want = max(1, round(self.ratio * len(row.genes)))
        for p in rng.permutation(present):
            states[prog == p] = TARGET
            if (states == TARGET).sum() >= want:
                break
        return states, None


@ROW_MASKS.register("panel")
@dataclass
class Panel:
    """Keep only the genes of a targeted panel visible; predict a sample of the rest.

    Rehearses cross-platform imputation (a MERFISH-like panel -> the transcriptome). Applies
    to whole-transcriptome cells; non-panel genes not sampled as targets are HIDDEN.
    """

    target_ratio: float = 0.25  # of the non-panel tokens
    min_visible: int = 8
    fallback: RandomGenes = field(default_factory=RandomGenes)

    def __call__(self, row: Row, rng, res: Resources):
        if not (row.whole_transcriptome and res.panels):
            return self.fallback(row, rng, res)
        panel = res.panels[rng.integers(len(res.panels))]
        inside = np.isin(row.genes, panel)
        if inside.sum() < self.min_visible or inside.all():
            return self.fallback(row, rng, res)
        states = np.full(len(row.genes), HIDDEN, np.int8)
        states[inside] = VISIBLE
        out = np.flatnonzero(~inside)
        states[rng.choice(out, size=max(1, round(self.target_ratio * len(out))), replace=False)] = TARGET
        return states, None


@ROW_MASKS.register("depth")
@dataclass
class Depth:
    """Shallower sequencing as evidence: binomially thin the input counts, score scattered originals."""

    keep: tuple = (0.1, 0.6)
    ratio: float = 0.15

    def __call__(self, row: Row, rng, res: Resources):
        states = _random_targets(len(row.genes), self.ratio, rng)
        if row.raw is None:
            return states, None
        thinned = rng.binomial(row.raw.astype(np.int64), rng.uniform(*self.keep)).astype(np.float32)
        lib = max(float(thinned.sum()), 1.0)
        return states, np.log1p(thinned / lib * 1e4).astype(np.float32)


# Group specs: (rel, valid, rng) -> boolean target mask over the group's members


@GROUP_MASKS.register("neighbours")
@dataclass
class Neighbours:
    """Scattered member cells of a spatial group (spatial imputation)."""

    ratio: float = 0.3

    def __call__(self, rel: np.ndarray, valid: np.ndarray, rng) -> np.ndarray:
        masked = np.zeros(len(rel), bool)
        n = max(1, round(self.ratio * len(valid)))
        masked[rng.choice(valid, size=min(n, len(valid)), replace=False)] = True
        return masked


@GROUP_MASKS.register("region")
@dataclass
class Region:
    """A contiguous patch: the members nearest a random anchor (tissue gaps, lesions)."""

    ratio: float = 0.3

    def __call__(self, rel: np.ndarray, valid: np.ndarray, rng) -> np.ndarray:
        masked = np.zeros(len(rel), bool)
        n = min(max(1, round(self.ratio * len(valid))), len(valid))
        anchor = rel[rng.choice(valid)]
        near = valid[np.argsort(np.linalg.norm(rel[valid] - anchor, axis=1), kind="stable")[:n]]
        masked[near] = True
        return masked


@dataclass
class Mixture:
    """Draw one spec per row or group. `stages` are (end_progress, weights): weights in force until
    training progress (0..1) reaches end_progress; without stages, `weights` hold throughout."""

    specs: list
    weights: list | None = None
    stages: list | None = None

    def weights_at(self, progress: float) -> np.ndarray:
        if self.stages:
            for end, w in self.stages:
                if progress < end:
                    break
            weights = w
        else:
            weights = self.weights or [1.0] * len(self.specs)
        w = np.asarray(weights, float)
        return w / w.sum()

    def pick(self, rng, progress: float = 0.0):
        if len(self.specs) == 1:  # no draw: keeps single-spec presets bit-identical to the original
            return self.specs[0]
        return self.specs[rng.choice(len(self.specs), p=self.weights_at(progress))]


@dataclass
class MaskPlan:
    rows: Mixture
    groups: Mixture


def mixture_from_dict(d: dict, registry: Registry) -> Mixture:
    """{"specs": [{"name": ..., **params}], "weights": [...] | "stages": [[end, [weights]], ...]}"""
    specs = [registry.build(**spec) for spec in d["specs"]]
    for key in ("weights",):
        if key in d and len(d[key]) != len(specs):
            raise ValueError(f"{len(d[key])} weights for {len(specs)} specs")
    for end, w in d.get("stages", []):
        if len(w) != len(specs):
            raise ValueError(f"stage ending at {end}: {len(w)} weights for {len(specs)} specs")
    stages = [(float(end), list(w)) for end, w in d["stages"]] if "stages" in d else None
    return Mixture(specs, weights=d.get("weights"), stages=stages)


def plan_from_dict(d: dict) -> MaskPlan:
    unknown = set(d) - {"rows", "groups", "description"}
    if unknown:
        raise ValueError(f"mask plan has unknown sections {sorted(unknown)}; expected rows, groups")
    return MaskPlan(mixture_from_dict(d["rows"], ROW_MASKS), mixture_from_dict(d["groups"], GROUP_MASKS))


def load_plan(name_or_path: str) -> MaskPlan:
    """A mask plan from configs/masks/<name>.toml, or from a TOML path."""
    path = Path(name_or_path)
    if path.suffix != ".toml":
        path = MASK_CONFIGS / f"{name_or_path}.toml"
    if not path.exists():
        known = sorted(p.stem for p in MASK_CONFIGS.glob("*.toml"))
        raise FileNotFoundError(f"no mask plan {name_or_path!r}; known: {known}")
    with open(path, "rb") as fh:
        return plan_from_dict(tomllib.load(fh))


def preset(name: str) -> MaskPlan:
    """Kept for callers of the earlier API: the named plan in configs/masks/."""
    return load_plan(name)
