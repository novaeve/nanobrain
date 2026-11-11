"""GPU leases with memory admission shared with other experiments on 0..4."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import os,subprocess,threading,time


def parent_pid(pid):
    try:return int((Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[1])
    except (OSError,ValueError,IndexError):return 0


class ResourcePool:
    def __init__(self,gpus,per_gpu,options=None):
        if set(gpus)-set(range(5)):raise ValueError('only physical GPUs 0..4 are authorized')
        self.gpus=list(gpus);self.per_gpu=per_gpu;self.options=options or {};self.condition=threading.Condition()
        self.leases={};self.snapshot={};self.snapshot_at=0.;self.pid_at={}

    def reserve_mib(self,job):
        if job['action']=='candidate':return self.options.get('candidate_reservation_mib',4096)
        if any(s in job['method'] for s in ('geneformer','scgpt','scuniflow')):return self.options.get('foundation_reservation_mib',12288)
        return self.options.get('native_reservation_mib',8192)

    def refresh(self):
        query=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True,timeout=15)
        devices={};uuids={}
        for line in query.splitlines():
            gpu,uuid,free=[x.strip() for x in line.split(',')];gpu=int(gpu)
            if gpu in self.gpus:devices[gpu]=dict(free_mib=int(free),own_mib=0);uuids[uuid]=gpu
        query=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=15)
        roots={lease.get('pid') for lease in self.leases.values()}-{None}
        for line in query.splitlines():
            uuid,pid,memory=[x.strip() for x in line.split(',')]
            if uuid not in uuids or not memory.isdigit():continue
            pid=int(pid);visited=set()
            while pid and pid not in visited:
                if pid in roots:
                    devices[uuids[uuid]]['own_mib']+=int(memory);break
                visited.add(pid);pid=parent_pid(pid)
        self.snapshot=devices;self.snapshot_at=time.monotonic()

    def acquire(self,key,job,needs_gpu=True):
        reserve=self.reserve_mib(job);margin=self.options.get('gpu_memory_margin_mib',4096)
        host_reserve=int(job.get('estimated_host_mib',2048))
        with self.condition:
            while True:
                if time.monotonic()-self.snapshot_at>5 or not self.snapshot:
                    try:self.refresh()
                    except (OSError,subprocess.SubprocessError,ValueError):self.snapshot={};self.condition.wait(5);continue
                counts=Counter(v['gpu'] for v in self.leases.values())
                meminfo={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
                host_reserved=sum(v.get('host_reserve_mib',0) for v in self.leases.values())
                host_ok=meminfo.get('MemAvailable',0)//1024-host_reserved>=self.options.get('minimum_host_available_mib',32768)+host_reserve
                if host_ok and not needs_gpu:
                    self.leases[key]=dict(gpu=None,reserve_mib=0,host_reserve_mib=host_reserve,method=job['method'],pid=None)
                    return None
                choices=[]
                if host_ok:
                    for gpu,s in self.snapshot.items():
                        reserved=sum(v['reserve_mib'] for v in self.leases.values() if v['gpu']==gpu)
                        capacity=s['free_mib']+s['own_mib']-margin
                        if counts[gpu]<self.per_gpu and reserved+reserve<=capacity:choices.append(gpu)
                if choices:
                    gpu=max(choices,key=lambda gpu:(self.snapshot[gpu]['free_mib']-sum(v['reserve_mib'] for v in self.leases.values() if v['gpu']==gpu),-counts[gpu]))
                    self.leases[key]=dict(gpu=gpu,reserve_mib=reserve,host_reserve_mib=host_reserve,method=job['method'],pid=None)
                    return gpu
                self.condition.wait(3)

    def started(self,key,pid):
        with self.condition:
            self.leases[key]['pid']=pid

    def release(self,key):
        with self.condition:
            self.leases.pop(key,None);self.snapshot_at=0.;self.condition.notify_all()

    def grow_reservation(self,job):
        kind='candidate_reservation_mib' if job['action']=='candidate' else 'foundation_reservation_mib' if any(s in job['method'] for s in ('geneformer','scgpt','scuniflow')) else 'native_reservation_mib'
        with self.condition:self.options[kind]=min(65536,max(self.reserve_mib(job)*2,8192))
