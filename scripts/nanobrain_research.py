#!/usr/bin/env python3
"""Prepare/inspect by default; formal work requires explicit --execute."""
from __future__ import annotations
import argparse,dataclasses,json,os,sys,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT))
from nanobrain.research.workflow import Workflow,load_config,save


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'configs/research_v3_react.json')
    p.add_argument('--output',type=Path,default=ROOT/'results/research-v3-react')
    p.add_argument('--execute',action='store_true');p.add_argument('--audit',action='store_true');args=p.parse_args()
    cfg=load_config(args.config)
    if args.audit:
        from nanobrain.research.audit import audit
        result=audit(ROOT,cfg);save(ROOT/'preparation_checks/availability.json',result);print(json.dumps(result,indent=2));return
    workflow=Workflow(ROOT,cfg,args.output)
    if not args.execute:print(json.dumps(workflow.plan(),indent=2));return
    # Credentials remain in controller only; they are never copied to workers.
    from nanobrain.llm_client import OpenAICompatibleClient
    from nanobrain.api_config import APIConfig
    config=dataclasses.replace(APIConfig.from_env(),model=cfg['model'],max_retries=-1,timeout_seconds=900)
    config.validate(require_credentials=True)
    client=OpenAICompatibleClient(config)
    import fcntl
    lock=(workflow.output/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    workflow.progress(status='starting',pid=os.getpid(),model=cfg['model'],reasoning_effort=cfg['reasoning_effort'],
                      gpu_ids=cfg['gpu_ids'],maximum_workers=workflow.maximum_workers)
    stop=threading.Event()
    def heartbeat():
        while not stop.wait(20):workflow.progress()
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
    try:print(json.dumps(workflow.execute(client),indent=2))
    except Exception as exc:
        message=str(exc).replace(config.api_key,'[redacted]') if config.api_key else str(exc)
        workflow.progress(status='failed',error=type(exc).__name__+': '+message[:3000])
        raise RuntimeError(message) from None
    finally:stop.set()

if __name__=='__main__':main()
