"""Read-only preparation inventory; never fits baselines or invokes an LLM."""
from __future__ import annotations
import dataclasses,json,shutil
from pathlib import Path
from .benchmark import NATIVE,CONTROLS,NOT_APPLICABLE,control_bundle


def audit(root,cfg):
    from nanobrain.data import load_v3_subset
    from nanobrain.baselines import build
    from nanobrain.baselines.base import AdapterContext
    from nanobrain.genome import CandidateGenome
    root=Path(root);data=load_v3_subset(root/cfg['data'],include_test=False)
    context=AdapterContext(data.train,data.validation,root/cfg['data'],root/'preparation_checks/audit',cfg['seeds'][0],'cpu',2,
        data.development_train,data.development_validation,cfg['native_options'])
    native=[]
    for name in NATIVE:
        try:
            adapter=build(name);available=adapter.availability(context)
            native.append(dict(method=name,tasks=adapter.tasks,**dataclasses.asdict(available),results=None))
        except Exception as exc:native.append(dict(method=name,status='blocked',reason=str(exc),results=None))
    controls=[] # Bundles remain search utilities, never independent benchmarks.
    requirements={key:str(Path(cfg['scaling'][key])) for key in ('source','contract','niche_cache')}
    return dict(status='prepared',native=native,component_controls=controls,not_applicable=NOT_APPLICABLE,
        sandbox_binary=shutil.which('bwrap'),scaling_paths={k:dict(path=p,exists=Path(p).exists()) for k,p in requirements.items()},
        formal_training_started=False,evolution_started=False,llm_called=False)
