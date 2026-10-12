"""Atomic full-training snapshots, bound to the immutable dataset and run config."""
import os
from pathlib import Path
import random
import tempfile
import numpy as np
import torch
from .full_training import FullTrainingProgress


def save_full_checkpoint(path, *, model, optimizer, progress, data_fingerprint,
                         config_fingerprint, optimizer_steps, scaler=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    state=dict(format='brainwm_full_checkpoint_v1',model=model.state_dict(),
        optimizer=optimizer.state_dict(),progress=progress.state_dict(),
        data_fingerprint=data_fingerprint,config_fingerprint=config_fingerprint,
        optimizer_steps=optimizer_steps,python_rng=random.getstate(),
        numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
        scaler=None if scaler is None else scaler.state_dict())
    fd,temporary=tempfile.mkstemp(prefix=path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as handle:
            torch.save(state,handle);handle.flush();os.fsync(handle.fileno())
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)


def load_full_checkpoint(path, *, model, optimizer, expected_rows_per_pass,
                         data_fingerprint, config_fingerprint, scaler=None):
    # This loader is for locally produced trusted training state, including RNG.
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state.get('format')!='brainwm_full_checkpoint_v1':
        raise ValueError('not a full-training checkpoint')
    if state['data_fingerprint']!=data_fingerprint or state['config_fingerprint']!=config_fingerprint:
        raise ValueError('checkpoint dataset/config differs from the current run')
    progress=FullTrainingProgress.from_state_dict(state['progress'],expected_rows_per_pass=expected_rows_per_pass)
    model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
    if scaler is not None and state['scaler'] is not None:scaler.load_state_dict(state['scaler'])
    random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
    torch.set_rng_state(state['torch_rng'])
    if state['cuda_rng'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    return progress,state['optimizer_steps']
