"""Full-corpus convergence accounting shared by iterative benchmark trainers."""
from dataclasses import dataclass, asdict
import math


@dataclass
class FullTrainingProgress:
    expected_rows_per_pass: int
    minimum_passes: int = 5
    patience: int = 5
    direction: str = 'minimize'
    min_delta: float = 0.0
    completed_passes: int = 0
    rows_this_pass: int = 0
    best_metric: float | None = None
    stale_validations: int = 0
    pending_validation: bool = False
    converged: bool = False

    def __post_init__(self):
        if min(self.expected_rows_per_pass,self.minimum_passes,self.patience)<=0:
            raise ValueError('row count, minimum passes and patience must be positive')
        if self.direction not in {'minimize','maximize'} or self.min_delta<0:
            raise ValueError('invalid convergence policy')

    def record_batch(self, rows: int):
        if self.pending_validation or self.converged:
            raise RuntimeError('cannot train before pending validation or after convergence')
        if rows<=0 or self.rows_this_pass+rows>self.expected_rows_per_pass:
            raise ValueError('batch would violate full-pass coverage')
        self.rows_this_pass+=rows

    def finish_pass(self):
        if self.pending_validation or self.rows_this_pass!=self.expected_rows_per_pass:
            raise ValueError('cannot count a partial or already-finished pass')
        self.completed_passes+=1
        self.rows_this_pass=0
        self.pending_validation=True

    def record_validation(self, metric: float, *, rows: int, expected_rows: int) -> bool:
        if not self.pending_validation:
            raise RuntimeError('validation must follow a complete training pass')
        if rows!=expected_rows or expected_rows<=0 or not math.isfinite(metric):
            raise ValueError('validation must be finite and cover its full declared population')
        improved=self.best_metric is None or (
            metric<self.best_metric-self.min_delta if self.direction=='minimize'
            else metric>self.best_metric+self.min_delta)
        if improved:
            self.best_metric=metric;self.stale_validations=0
        else:
            self.stale_validations+=1
        self.pending_validation=False
        self.converged=self.completed_passes>=self.minimum_passes and self.stale_validations>=self.patience
        return improved

    def state_dict(self):
        return asdict(self)

    @classmethod
    def from_state_dict(cls, state, *, expected_rows_per_pass):
        if state['expected_rows_per_pass']!=expected_rows_per_pass:
            raise ValueError('checkpoint belongs to a different training population')
        return cls(**state)
