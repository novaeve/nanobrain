"""Human developmental task construction for population Flow Matching.

Intervals are cross-sectional and unpaired by contract. We never claim
cell-level lineage: source and target batches are population endpoints within
the same split/dataset/time stratum and are matched only through the task
manifest's runtime policy.
"""
from __future__ import annotations
import ast
import math
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import torch


@dataclass(frozen=True)
class DevelopmentInterval:
    interval_id: str
    split: str
    dataset: str
    assay_type: str
    time_stratum: str
    source_time_days: float
    target_time_days: float
    source_sample_keys: tuple[str, ...]
    target_sample_keys: tuple[str, ...]

    @property
    def delta_time_days(self) -> float:
        return self.target_time_days - self.source_time_days


def load_development_intervals(root: str | Path, split='train') -> list[DevelopmentInterval]:
    frame = pd.read_parquet(Path(root) / 'task_development_labels.parquet')
    samples = pd.read_parquet(Path(root) / 'samples.parquet').set_index('sample_key')
    frame = frame[(frame.split == split) & (frame.cell_correspondence == 'unpaired_cross_sectional')]
    result = []
    for i, row in frame.iterrows():
        parse = lambda x: tuple(ast.literal_eval(x)) if isinstance(x, str) else tuple(x)
        interval=DevelopmentInterval(str(row.get('interval_id', i)), str(row.split),
            str(row.dataset), str(row.assay_type), str(row.time_stratum),
            float(row.source_time_days), float(row.target_time_days),
            parse(row.source_sample_keys), parse(row.target_sample_keys))
        validate_interval_samples(interval,samples)
        result.append(interval)
    return result


def flow_pair_inputs(source_state: torch.Tensor, target_state: torch.Tensor,
                     source_time_days: float, target_time_days: float,
                     device=None) -> dict[str, torch.Tensor]:
    """Create a population endpoint training pair, not a lineage pair."""
    device = device or source_state.device
    delta = float(target_time_days - source_time_days)
    if not math.isfinite(delta) or delta <= 0:
        raise ValueError('observed developmental interval must have a finite positive duration')
    if source_state.shape != target_state.shape or source_state.ndim != 2:
        raise ValueError('coupled endpoint states must have matching [cells, features] shapes')
    if not torch.isfinite(source_state).all() or not torch.isfinite(target_state).all():
        raise ValueError('endpoint states must be finite')
    tau = torch.rand(source_state.shape[0], 1, device=device)
    x0, x1 = source_state.to(device), target_state.to(device)
    xt = (1 - tau) * x0 + tau * x1
    return {'x0': x0, 'x1': x1, 'xt': xt, 'tau': tau,
            'source_time': torch.full((len(x0), 1), source_time_days, device=device),
            'target_time': torch.full((len(x0), 1), target_time_days, device=device),
            'target_velocity': x1 - x0,
            'physical_velocity_per_day': (x1 - x0) / delta}


def validate_interval_pair(interval: DevelopmentInterval, *, min_delta_days: float = 1e-6) -> None:
    """Fail closed on invalid/degenerate developmental intervals."""
    if interval.split not in {'train', 'validation', 'test'}:
        raise ValueError(f'unknown split: {interval.split}')
    if not interval.source_sample_keys or not interval.target_sample_keys:
        raise ValueError(f'interval {interval.interval_id} has an empty endpoint')
    if not math.isfinite(interval.delta_time_days) or interval.delta_time_days < min_delta_days:
        raise ValueError(f'interval {interval.interval_id} has non-positive delta time')
    if set(interval.source_sample_keys) & set(interval.target_sample_keys):
        raise ValueError(f'interval {interval.interval_id} reuses source samples as target')


def validate_interval_samples(interval: DevelopmentInterval, samples: pd.DataFrame) -> None:
    """Validate both endpoint populations against the frozen sample context."""
    validate_interval_pair(interval)
    if not samples.index.is_unique:raise ValueError('sample index must be unique')
    for keys,time in [(interval.source_sample_keys,interval.source_time_days),
                      (interval.target_sample_keys,interval.target_time_days)]:
        if not set(keys)<=set(samples.index):raise ValueError('unknown developmental sample')
        part=samples.loc[list(keys)]
        for field,value in [('split',interval.split),('dataset',interval.dataset),
                            ('assay_type',interval.assay_type),('time_stratum',interval.time_stratum)]:
            if not part[field].eq(value).all():raise ValueError(f'development interval crosses {field}')
        ages=pd.to_numeric(part.time_point_days,errors='coerce')
        if not ages.map(lambda age:math.isfinite(age) and math.isclose(age,time,abs_tol=1e-6)).all():
            raise ValueError('development interval age differs from sample metadata')
        if not part.time_status.isin(['exact','approximate','range_midpoint']).all():
            raise ValueError('unreliable age cannot supervise development')


def build_development_task(interval: DevelopmentInterval, source_state: torch.Tensor,
                           target_state: torch.Tensor, *, device=None) -> dict[str, torch.Tensor]:
    """Validate and materialize one cross-sectional developmental task.

    The two endpoint populations are sampled independently within the manifest
    stratum; this function deliberately does not match cells by barcode.  The
    returned normalized velocity is used by Flow Matching, while the per-day
    velocity is retained for reporting and time-span stratification.
    """
    validate_interval_pair(interval)
    pair = flow_pair_inputs(source_state, target_state, interval.source_time_days,
                            interval.target_time_days, device=device)
    pair['interval_days'] = pair['tau'].new_full(pair['tau'].shape, interval.delta_time_days)
    pair['interval_id'] = interval.interval_id
    return pair
