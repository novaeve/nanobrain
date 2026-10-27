"""Small, deterministic evolutionary controller for nanoBrain candidates."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable, Iterable

from .genome import CandidateGenome, default_genome


@dataclass(frozen=True)
class CandidateScore:
    candidate_id: str
    quality: float
    validation_loss: float
    runtime_seconds: float
    parameter_count: int
    receipt: dict
    task_objectives: tuple[float, ...] = ()

    @property
    def objective(self) -> tuple[float, ...]:
        # All values are converted to maximize form for Pareto comparison.
        if self.task_objectives:
            return (
                *self.task_objectives,
                -float(self.runtime_seconds),
                -float(self.parameter_count),
            )
        return (
            float(self.quality),
            -float(self.validation_loss),
            -float(self.runtime_seconds),
            -float(self.parameter_count),
        )


def score_receipt(receipt: dict) -> CandidateScore:
    validation = receipt.get("validation", {})
    losses = receipt.get("train_steps", [])
    last_loss = float(losses[-1].get("total", 0.0)) if losses else 0.0
    return CandidateScore(
        candidate_id=str(receipt["candidate_id"]),
        quality=float(receipt.get("quality", 0.0)),
        validation_loss=last_loss,
        runtime_seconds=float(receipt.get("runtime_seconds", float("inf"))),
        parameter_count=int(receipt.get("parameter_count", 0)),
        receipt=receipt,
        task_objectives=tuple(
            float(value)
            for value in receipt.get("promotion", {}).get("task_objectives", ())
        ),
    )


def dominates(left: CandidateScore, right: CandidateScore) -> bool:
    a, b = left.objective, right.objective
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))


def pareto_front(scores: Iterable[CandidateScore]) -> list[CandidateScore]:
    values = list(scores)
    return [candidate for candidate in values if not any(dominates(other, candidate) for other in values if other is not candidate)]


@dataclass
class EvolutionConfig:
    population_size: int = 4
    generations: int = 2
    mutation_rate: float = 0.35
    seed: int = 7


class EvolutionSearch:
    """Generate, evaluate, and retain candidates under one fixed protocol."""

    def __init__(self, config: EvolutionConfig, evaluator: Callable[[CandidateGenome], dict]):
        self.config = config
        self.evaluator = evaluator
        self.rng = random.Random(int(config.seed))

    def _initial_population(self) -> list[CandidateGenome]:
        population = [default_genome(seed=self.rng.randrange(1, 2**31 - 1))]
        while len(population) < max(1, int(self.config.population_size)):
            population.append(population[0].mutate(self.rng, rate=0.8))
        return population

    def run(self) -> tuple[list[dict], list[CandidateScore]]:
        receipts: list[dict] = []
        scores: list[CandidateScore] = []
        population = self._initial_population()
        for generation in range(max(1, int(self.config.generations))):
            generation_scores: list[CandidateScore] = []
            for genome in population:
                receipt = dict(self.evaluator(genome))
                receipt["generation"] = generation
                receipts.append(receipt)
                score = score_receipt(receipt)
                generation_scores.append(score)
                scores.append(score)
            survivors = pareto_front(generation_scores)
            survivors.sort(key=lambda value: value.objective, reverse=True)
            survivors = survivors[: max(1, int(self.config.population_size) // 2)]
            parent_genomes = [CandidateGenome.from_dict(score.receipt["genome"]) for score in survivors]
            next_population = list(parent_genomes)
            while len(next_population) < max(1, int(self.config.population_size)):
                left = self.rng.choice(parent_genomes)
                right = self.rng.choice(parent_genomes)
                child = left.crossover(right, self.rng).mutate(self.rng, rate=self.config.mutation_rate)
                next_population.append(child)
            population = next_population
        return receipts, pareto_front(scores)
