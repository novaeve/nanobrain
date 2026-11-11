"""Audited baseline-only handoff; no mixed-source evolution can be adopted."""
from __future__ import annotations
import json,os,signal,time
from pathlib import Path
from .contracts import digest


def process_start(pid):
    try:return (Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[19]
    except (OSError,IndexError):return None


def adoption_plan(output):
    path=Path(output)/'benchmark-revision.json'
    if not path.exists():return {},None
    record=json.loads(path.read_text())
    return {r['job_id']:r for r in record.get('preserved_native_workers',[])},record


def adopt_receipt(root,output,job,worker,timeout):
    """Workers write atomic receipts themselves; the paused parent need not finish."""
    identity=digest(job);directory=Path(output)/'jobs'/identity;deadline=time.monotonic()+timeout
    effective=json.loads((directory/'input.json').read_text());expected=dict(job,device='cuda:0')
    if effective!=expected or worker['job_id']!=identity:raise ValueError('adopted worker recipe mismatch')
    while time.monotonic()<deadline:
        path=directory/'receipt.json'
        if path.exists():
            result=json.loads(path.read_text())
            if result.get('status')!='complete' or result.get('method')!=job['method'] or result.get('seed')!=job['seed']:
                raise ValueError('invalid adopted native receipt')
            if not result.get('configuration_frozen_before_test'):raise ValueError('unsealed adopted baseline')
            from .workflow import file_sha
            for artifact in result.get('model_artifacts',[]):
                local=(directory/artifact['path']).resolve()
                if not local.is_relative_to(directory.resolve()) or file_sha(local)!=artifact['sha256']:raise ValueError('adopted native artifact mismatch')
            result.update(job_id=identity,physical_gpu=worker['gpu'],data_manifest_sha256=job['data_manifest_sha256'],adopted_from_previous_baseline_contract=True)
            return result
        if process_start(worker['pid'])!=worker['process_start']:raise RuntimeError('preserved native worker exited without a receipt')
        time.sleep(5)
    raise TimeoutError('preserved native worker timed out')


def retire_old_controller(output):
    workers,record=adoption_plan(output)
    if not record:return
    for identity in workers:
        path=Path(output)/'jobs'/identity/'receipt.json'
        if not path.exists() or json.loads(path.read_text()).get('job_id')!=identity:return
    pid=record['old_controller_pid']
    if process_start(pid)==record.get('old_controller_start'):
        os.kill(pid,signal.SIGKILL)


def migrate(root,output,config,backup,allowed_changes):
    """Explicit user-requested protocol amendment before generation zero."""
    from .workflow import source_fingerprint,file_sha,runtime_versions,save
    from .benchmark import coverage_catalog,native_job
    import ast
    root=Path(root);output=Path(output);backup=Path(backup)
    old=json.loads((backup/'run-contract.json').read_text());handoff=json.loads((backup/'handoff.json').read_text())
    if (output/'run-contract.json').exists() and json.loads((output/'run-contract.json').read_text())!=old:raise ValueError('old contract changed before amendment')
    if list((output/'lineage/candidates').glob('*/bundle.json')) or list((output/'generations').glob('*.json')):raise ValueError('benchmark amendment requires no evolutionary candidates')
    if old['data_manifest_sha256']!=file_sha((root/config['data'])/'manifest.json'):raise ValueError('data contract changed')
    previous=json.loads((backup/'source-files.json').read_text())
    current={str(p.relative_to(root)):file_sha(p) for folder in ('src','scripts','reference_methods') for p in sorted((root/folder).rglob('*.py')) if '__pycache__' not in p.parts}
    changed={p for p in previous.keys()|current.keys() if previous.get(p)!=current.get(p)}
    if changed-set(allowed_changes):raise ValueError('unapproved scientific source changes: '+str(changed-set(allowed_changes)))
    def native_ast(path):
        return ast.dump(next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='native_job'),include_attributes=False)
    if native_ast(backup/'research/benchmark.py')!=native_ast(root/'src/nanobrain/research/benchmark.py'):raise ValueError('native scientific runner changed; cannot reuse its receipts')
    unchanged_config={k:v for k,v in config.items() if k!='transfer'}
    if unchanged_config!=old['config']:raise ValueError('only the transfer policy may amend configuration')
    contract=dict(config=config,data_manifest_sha256=old['data_manifest_sha256'],source_sha256=source_fingerprint(root),runtime_versions=runtime_versions(),baseline_catalog=coverage_catalog())
    record=dict(previous_contract=old,new_contract=contract,changed_files=sorted(changed),backup=str(backup),
        old_controller_pid=handoff['old_controller_pid'],old_controller_start=process_start(handoff['old_controller_pid']),
        preserved_native_workers=handoff['preserved_native_workers'],removed_component_workers=handoff['cancelled_component_workers'],
        reason='Explicit user request: full methods with missing-task transfer heads; remove standalone component controls',
        native_runner_ast_unchanged=True,common_evaluator_unchanged=True,existing_evolution_candidates=0)
    save(output/'benchmark-revision.json',record);save(output/'run-contract.json',contract)
    lock=output/'controller.lock'
    if lock.exists():lock.rename(backup/'paused-controller.lock')
    return record
