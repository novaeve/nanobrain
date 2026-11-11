import json
import numpy as np
import pytest
import torch

import brainwm.training_observability as observability
from brainwm.training_observability import (ValidationAccumulator, append_jsonl,
    atomic_json, atomic_torch_save, classification_metrics)


def test_json_logs_are_strict_and_append_only(tmp_path):
    stream=tmp_path/'metrics.jsonl'
    append_jsonl(stream,{'step':1,'loss':2.0})
    append_jsonl(stream,{'step':2,'loss':1.0})
    assert [json.loads(line)['step'] for line in stream.read_text().splitlines()]==[1,2]
    manifest=tmp_path/'manifest.json'
    atomic_json(manifest,{'metric':None})
    assert json.loads(manifest.read_text())=={'metric':None}


def test_torch_checkpoint_replacement_is_atomic(tmp_path,monkeypatch):
    checkpoint=tmp_path/'model.pt'
    atomic_torch_save(checkpoint,{'step':1})
    assert torch.load(checkpoint,weights_only=True)=={'step':1}

    def interrupted_save(payload,path):
        path.write_bytes(b'partial')
        raise RuntimeError('interrupted')

    monkeypatch.setattr(observability.torch,'save',interrupted_save)
    with pytest.raises(RuntimeError,match='interrupted'):
        atomic_torch_save(checkpoint,{'step':2})
    assert torch.load(checkpoint,weights_only=True)=={'step':1}
    assert list(tmp_path.glob('.model.pt.*.tmp'))==[]


def test_classification_metrics_cover_macro_micro_and_weighted():
    metrics=classification_metrics(np.array([0,1,1,2]),np.array([0,1,0,2]),'annotation')
    assert metrics['annotation/count']==4
    for average in ('macro','micro','weighted'):
        assert f'annotation/precision_{average}' in metrics
        assert f'annotation/recall_{average}' in metrics
        assert f'annotation/f1_{average}' in metrics


def test_validation_accumulator_records_development_coverage_and_losses():
    accumulator=ValidationAccumulator()
    accumulator.add_development(8,torch.tensor(2.0),torch.tensor(3.0))
    metrics=accumulator.metrics()
    assert metrics['coverage/development_rows']==8
    assert metrics['coverage/development_groups']==1
    assert metrics['loss/flow_velocity']==2.0
    assert metrics['loss/flow_endpoint']==3.0
