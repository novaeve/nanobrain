from nanobrain.evolution import EVOLUTION_TASKS, joint_score
from nanobrain.task_training import CORE_TASKS


def test_evolution_uses_all_four_core_tasks():
    assert set(EVOLUTION_TASKS) == set(CORE_TASKS)
    metrics = {
        "annotation.f1_macro": 0.3,
        "integration.integration_composite": 0.4,
        "spatial_niche.ari": 0.1,
        "development.population_mse": 0.6,
    }
    baseline = joint_score(metrics)
    changed = {**metrics, "development.population_mse": 0.3}
    assert joint_score(changed) > baseline
