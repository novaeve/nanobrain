"""Create a full-training experiment ledger without claiming jobs were launched."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    artifact = ROOT/'contracts/healthy_global_benchmark_v3'
    frozen = json.loads((artifact/'FROZEN_SHA256.json').read_text())
    for name, digest in frozen.items():
        if hashlib.sha256((artifact/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'frozen dataset changed: {name}')
    manifest = json.loads((artifact/'manifest.json').read_text())
    output = ROOT/'results/full_v3_campaign'
    output.mkdir(parents=True, exist_ok=True)
    if (output/'experiments.csv').exists():
        raise FileExistsError('campaign already exists; do not overwrite run state')
    # This initial roster includes the declared controls and reference methods;
    # adapters must be verified against their actual implementation before launch.
    methods = {
        'joint_foundation': ['BrainWM-Light','BrainWM-Normal','BrainWM-Ultra'],
        'integration': ['raw_log1p_pca','harmony','scanorama','scVI'],
        'annotation': ['raw_nearest_centroid','source_mnb','scBrain'],
        'development': ['identity','source_linear_mean','source_only_ot','scFiT'],
        'spatial_niche': ['neighbor_only','no_structure','coordinate_permutation','scUniFlow'],
    }
    rows = []
    for task, names in methods.items():
        for name in names:
            for seed in [20260901,20260902,20260903]:
                rows.append(dict(task=task,method=name,seed=seed,status='pending_adapter_verification',
                    split_id=manifest['split_id'],initialization='fresh',full_passes=0,
                    unique_train_rows=0,optimizer_steps=0,validation_metric='',test_metric='',
                    stop_reason='',checkpoint='',pid='',log=''))
    with (output/'experiments.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    policy=dict(split_id=manifest['split_id'],
        frozen_sha256=hashlib.sha256((artifact/'FROZEN_SHA256.json').read_bytes()).hexdigest(),
        train_cells=manifest['counts']['by_split']['train']['cells'],
        initialization='fresh; no v2 weights',
        corpus='all train cells eligible for each task; no row caps or silent subsampling',
        minimum_train_passes=5,validation_patience=5,maximum_train_passes=None,
        stopping='validation plateau after minimum complete passes; resource interruption is incomplete',
        non_iterative='fit on full eligible train data; identity controls have no optimization',
        shared_model='one BrainWM per size and seed, evaluated across tasks; not one separately trained foundation per task',
        test='freeze all predictions before one final evaluation',
        metrics=['task primary','micro','macro','label','anatomy','development stage','assay'],
        absent_metrics='report not applicable or unsupported explicitly; never invent values',
        blocked_tasks={'perturbation':'missing intervention outcomes'},
        job_status='No job is running merely because it appears in this ledger.')
    (output/'campaign.json').write_text(json.dumps(policy,indent=2)+'\n')
    print(f'Prepared {len(rows)} experiment rows: {output}')


if __name__=='__main__': main()
