"""Fail-closed Linux bwrap launch for candidate code (no host-wide bind)."""
from __future__ import annotations
import hashlib,json, os, signal,subprocess, sys
from pathlib import Path


def sandbox_command(python, root, output, arguments, *, data=None, gpu=None, cache=None,baseline=None):
    python=Path(python).absolute();root=Path(root).resolve();output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    info=json.loads(subprocess.check_output([str(python),'-c',
        'import sys,json;print(json.dumps(dict(prefix=sys.base_prefix,paths=[p for p in sys.path if "site-packages" in p])))'],text=True))
    mounts={Path(p).resolve() for p in info['paths']}
    mounts|={Path(info['prefix']).resolve(),python.parent.parent,root/'src',root/'scripts',root/'reference_methods'}
    mounts|={Path(p) for p in ('/usr','/lib','/lib64','/bin','/sbin','/etc/ld.so.cache','/etc/alternatives') if Path(p).exists()}
    command=['bwrap','--die-with-parent','--new-session','--unshare-net','--unshare-pid','--unshare-ipc','--unshare-uts','--hostname','nanobrain','--cap-drop','ALL','--proc','/proc','--dev','/dev','--tmpfs','/tmp']
    mounted=[]
    for path in sorted(mounts,key=lambda p:len(str(p))):
        if path.exists() and not any(path.is_relative_to(parent) for parent in mounted):
            command+=['--ro-bind',str(path),str(path)];mounted.append(path)
    if data is not None:
        resolved=Path(data).resolve();command+=['--ro-bind',str(resolved),'/dataset','--ro-bind',str(resolved),str(Path(data).absolute())]
        for split in ('validation','test'):
            location=resolved/split;target=location.resolve()
            if location.is_symlink() and target.exists() and not target.is_relative_to(resolved):
                command+=['--ro-bind',str(target),str(target)]
        for name in ('foundation_models','human_v3_genes.parquet'):
            asset=root/'assets'/name
            if asset.exists():command+=['--ro-bind',str(asset.resolve()),str(asset)]
    if cache is not None:command+=['--ro-bind',str(Path(cache).resolve()),'/input-cache']
    if baseline is not None:command+=['--ro-bind',str(Path(baseline).resolve()),'/baseline']
    command+=['--bind',str(output),'/work','--chdir','/work']
    environment={'PATH':'/usr/bin:/bin','HOME':'/tmp','TMPDIR':'/tmp','PYTHONPATH':str(root/'src')+':'+str(root),
        'USER':'nanobrain','LOGNAME':'nanobrain','TORCHINDUCTOR_CACHE_DIR':'/tmp/torchinductor','NANOBRAIN_CPU_INDEX':str(int(hashlib.sha256(str(output).encode()).hexdigest()[:8],16)),'PYTHONHASHSEED':'0','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2',
        'NUMBA_CACHE_DIR':'/tmp/numba','JAX_COMPILATION_CACHE_DIR':'/tmp/jax','NUMBA_NUM_THREADS':'2','XLA_FLAGS':'--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2',
        'NANOBRAIN_COMPONENT_CACHE':'/work/component-cache' if cache is None else '/input-cache','NANOBRAIN_COMPONENT_CACHE_READONLY':'0' if cache is None else '1','JAX_PLATFORMS':'cpu','MPLCONFIGDIR':'/tmp/matplotlib','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
    if gpu is None:environment['CUDA_VISIBLE_DEVICES']=''
    else:
        if type(gpu)is not int or gpu not in range(5):raise ValueError('only physical GPU 0..4 is authorized')
        # Expose only the selected physical GPU and common CUDA control devices.
        for name in ('nvidiactl','nvidia-uvm','nvidia-uvm-tools',f'nvidia{gpu}'):
            device=Path('/dev')/name
            if device.exists():command+=['--dev-bind',str(device),str(device)]
        if Path('/sys').exists():command+=['--ro-bind','/sys','/sys']
        uuid=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
        if not uuid.startswith('GPU-') or '\n' in uuid:raise ValueError('unable to identify the selected GPU')
        environment['CUDA_VISIBLE_DEVICES']=uuid
    for key,value in environment.items():command+=['--setenv',key,value]
    return command+['--',str(python),'-m','nanobrain.research.worker',*arguments]


def launch(python,root,output,arguments,*,data=None,gpu=None,cache=None,baseline=None,timeout=300,on_started=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    command=sandbox_command(python,root,output,arguments,data=data,gpu=gpu,cache=cache,baseline=baseline)
    with (output/'worker.log').open('w') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
            env={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8'},start_new_session=True)
        if on_started:on_started(process.pid)
        try:code=process.wait(timeout=timeout)
        except BaseException:
            try:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
            raise
    if code:raise RuntimeError(f'sandbox worker failed ({code}); see {output}/worker.log')
    return json.loads((output/'receipt.json').read_text())
