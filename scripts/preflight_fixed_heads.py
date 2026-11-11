#!/usr/bin/env python3
"""Two-step real-data training and checkpoint-reload checks; no LLM calls."""
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch

from nanobrain.data import infer_label_cardinalities, load_v3_subset, verify_subset_manifest
from nanobrain.evolution import architecture_id
from nanobrain.fixed_heads import build_prompt, search_catalog, validate_search_scope
from nanobrain.genome import CandidateGenome
from nanobrain.model import NanoBrainModel, NanoBrainModelConfig
from nanobrain.training import TrainConfig, train_candidate
from scripts.run_nanobrain_evolution import evaluation_checkpoint


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main():
    torch.set_num_threads(2)
    protocol = json.loads((ROOT / "configs/nanobrain_graph_v2/backbone_training_fixed_heads_50g_20261030.json").read_text())
    genome = CandidateGenome.from_dict({**protocol["reference_genome"], "seed": protocol["evolution_seeds"][0]})
    validate_search_scope(genome.to_dict(), protocol["search_scope"])
    write(ROOT / "runtime/fixed_heads_search_catalog.json", search_catalog())
    parent = dict(genome=genome.to_dict(), architecture_id=architecture_id(genome.to_dict()))
    (ROOT / "runtime/fixed_heads_prompt_preview.txt").write_text(build_prompt(protocol, [parent], parent, {}, 12))
    # Verify the fixed heads against the tutorial project's actual definitions.
    tutorial_root = Path('/mnt/aliyunsb/zim_rider/nanobrain/nanobrain-github')
    def definitions(path):
        return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.ClassDef) and node.name in {'ClassificationHead', 'RegressionHead', 'FlowMatchingTransformerModule'}}
    assert definitions(ROOT / 'src/nanobrain/modules.py') == definitions(tutorial_root / 'src/nanobrain/modules.py')
    data = ROOT / protocol["data"]
    verify_subset_manifest(data, verify_hashes=True, data_splits=("train", "validation"))
    splits = load_v3_subset(data, include_test=False)
    assert splits.test is None and splits.development_test is None
    labels = infer_label_cardinalities(splits)
    model_config = NanoBrainModelConfig(splits.train.input_dim, labels['annotation'], labels['niche'], labels['integration'], splits.train.input_dim)
    output = ROOT / 'results/preflight-fixed-heads-20261030'
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for task in protocol['tasks']:
        print(f'preflight: {task}', flush=True)
        cfg = TrainConfig(steps=2, batch_size=16, validation_batch_size=2048, seed=genome.seed,
                          device='cpu', development_cells=8, validation_development_groups=1,
                          eval_interval=1, precision='fp32', task=task, checkpoint_policy='best_weights',
                          loss_balance_mode='fixed')
        receipt = train_candidate(splits, model_config, genome, cfg, output_dir=output / task)
        write(output / task / 'receipt.json', receipt)
        assert receipt['completed_steps'] == 2 and receipt['test_opened'] is False
        checkpoint = evaluation_checkpoint(receipt, 'best_validation')
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        restored = NanoBrainModel(model_config, CandidateGenome.from_dict(payload['genome']))
        restored.load_state_dict(payload['model'])
        restored.eval()
        with torch.no_grad():
            if task == 'development':
                batch = splits.development_validation.batch(0, max_cells=8, seed=17)
                prediction, _ = restored.predict_development_population(batch.source, batch.source_time_days, batch.target_time_days)
            else:
                batch = next(splits.validation.iter_batches(16, shuffle=False))
                z = restored.encode(batch)
                if task == 'integration':
                    logits, _ = restored.integration_adversarial_logits(z)
                    prediction = restored.integration_reconstruction_head(z)
                    assert torch.isfinite(logits).all()
                else:
                    prediction = (restored.annotation_head if task == 'annotation' else restored.niche_head)(z)
            assert torch.isfinite(prediction).all()
        records.append(dict(task=task, completed_steps=2, best_step=receipt['best_step'],
                            checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                            output_shape=list(prediction.shape), reload_forward_passed=True, test_opened=False))
    source = Path(protocol['forked_from'])
    fork_receipt = json.loads((ROOT / 'runtime/fork_receipt.json').read_text())
    original_unchanged = all(hashlib.sha256((source / name).read_bytes()).hexdigest() == digest
                             for name, digest in fork_receipt['original_sha256'].items())
    assert original_unchanged
    result = dict(search_scope=protocol['search_scope'], tutorial_task_adapter_definitions_match=True,
                  original_source_unchanged=original_unchanged, formal_campaign_started=False,
                  api_calls=0, real_data_preflight=records)
    write(ROOT / 'runtime/fixed_heads_verification.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
