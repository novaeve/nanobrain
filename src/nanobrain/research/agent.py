"""Bounded ReAct tool loop: observe, inspect, propose, validate, submit."""
from __future__ import annotations
import json
from pathlib import Path
from .contracts import candidate_schema,digest,validate_bundle,semantic_id
from .store import Store

SYSTEM='''Design a nanoBrain backbone + independent task heads. You may recombine the component catalog or author restricted PyTorch modules/losses/optimizers. There is no numeric cap on custom plugins or source-file count; tensor contracts and total source/parameter/resource budgets still apply. The initial tree is the COMPLETE evolution DAG: every node and every parent edge, including unsuccessful and unselected branches. Only individual node contents are outlined. Use this global tree to reason about changes across branches and ancestors; inspect any relevant node with lineage/read_artifact and paginate until you have its full bundle, validation or result. The parent_ids in context are recommended archive parents; other fully evaluated valid historical nodes may also be parents after inspection. This tree view does not imply an MCTS/UCT search algorithm. Use tools to inspect details on demand. Submit only through submit_candidate after validate_candidate accepts the immutable bundle. Never invent metrics, ancestry or results. Data splits, masks, target semantics and official evaluation code are immutable. Validation chooses weights; test feedback chooses architecture, so all test feedback is exploratory. A useful valid non-winning candidate can remain in the novelty/Pareto archive. No arbitrary shell, network or data access exists. Code modules receive tensors, never labeled dataset objects. Keep useful state in artifacts. A final prose answer does not submit a candidate.'''


def loads(text):
    def pairs(items):
        result={}
        for k,v in items:
            if k in result:raise ValueError('duplicate JSON keys')
            result[k]=v
        return result
    return json.loads(text,object_pairs_hook=pairs,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite JSON')))


def tool_schema(name,description,properties=None,required=None):
    return dict(type='function',function=dict(name=name,description=description,parameters=dict(type='object',
        properties=properties or {},required=required or [],additionalProperties=False)))


TOOLS=[
 tool_schema('inspect_contract','Read strict candidate schema and tensor API contracts.'),
 tool_schema('list_components','Read a bounded page of existing component names, sources and factory parameters.',
   {'kind':{'type':'string'},'offset':{'type':'integer','minimum':0}},['kind','offset']),
 tool_schema('baseline_detail','Read frozen benchmark results for one method.',{'method':{'type':'string'}},['method']),
 tool_schema('read_baseline_artifact','Read one bounded page of a benchmark job receipt.',{'job_id':{'type':'string'},'offset':{'type':'integer','minimum':0}},['job_id','offset']),
 tool_schema('read_artifact','Read bounded candidate bundle/validation/result JSON.',
   {'candidate_id':{'type':'string'},'kind':{'enum':['bundle','validation','result']},'offset':{'type':'integer','minimum':0}},['candidate_id','kind','offset']),
 tool_schema('lineage','Read one generation/ancestor result.',{'candidate_id':{'type':'string'}},['candidate_id']),
 tool_schema('write_candidate','Stage a complete immutable JSON candidate artifact. This does not train.',{'bundle':candidate_schema()},['bundle']),
 tool_schema('validate_candidate','Run a disposable CPU-only synthetic acceptance probe, with no benchmark data.',{'candidate_id':{'type':'string'}},['candidate_id']),
 tool_schema('submit_candidate','Submit exactly one accepted nonduplicate candidate for trusted evaluation.',{'candidate_id':{'type':'string'}},['candidate_id'])]


class ToolSession:
    def __init__(self,store,baseline,validator,parents,seen,origin=None):
        self.store=store;self.baseline=baseline;self.validator=validator;self.parents=parents;self.seen=set(seen)
        self.staged=set();self.validated={};self.submitted=None;self.validation_attempts=0
        self.origin=origin

    def call(self,name,args):
        from jsonschema import Draft202012Validator
        schema=next((t['function']['parameters'] for t in TOOLS if t['function']['name']==name),None)
        if schema is None:raise ValueError('unknown tool')
        Draft202012Validator(schema).validate(args)
        if name=='inspect_contract':
            return dict(candidate_schema=candidate_schema(),factory_signatures={
                'encoder':'build(input_dim, hidden_dim, **declared_parameters) -> nn.Module; forward(features)',
                'head':'build(hidden_dim, output_dim, **declared_parameters) -> nn.Module; forward(state)',
                'flow':'build(hidden_dim, **declared_parameters) -> nn.Module; forward(state, context, time[N,3])',
                'operation_attention':'build(hidden_dim, **declared_parameters) -> nn.Module; forward(state)',
                'loss':'compute(prediction, target, **declared_parameters) -> finite nonnegative scalar with gradient',
                'optimizer':'build(parameters, lr, weight_decay, **declared_parameters) -> torch.optim.Optimizer'},
                policy='Approved torch tensor APIs only. No private introspection, I/O, shell or network. Source dirs modules/, losses/, optimizers/. Every plugin must be connected. See parent bundles for full graph schema.')
        if name=='list_components':
            from nanobrain.architecture import graph_catalog
            catalog=graph_catalog();entries=catalog.get(args['kind'],[]);offset=args['offset']
            return dict(kinds=list(catalog),total=len(entries),entries=entries[offset:offset+12],next_offset=offset+12 if len(entries)>offset+12 else None)
        if name=='baseline_detail':
            return [{k:v for k,v in r.items() if k not in {'training','bundle'}} for r in self.baseline if r['method']==args['method']]
        if name=='read_baseline_artifact':
            known={identifier for r in self.baseline for identifier in [r.get('job_id'),*r.get('source_job_ids',[])] if identifier}
            if args['job_id'] not in known:raise ValueError('unknown baseline job artifact')
            from .contracts import artifact_path
            path=artifact_path(self.store.root.parent/'jobs',args['job_id']+'/receipt.json');text=path.read_text();offset=args['offset']
            return dict(text=text[offset:offset+8000],offset=offset,total_chars=len(text),truncated=offset+8000<len(text),next_offset=offset+8000 if offset+8000<len(text) else None)
        if name=='read_artifact':return self.store.read_text(args['candidate_id'],args['kind'],args['offset'])
        if name=='lineage':
            identifier=self.store.resolve_id(args['candidate_id']);bundle=self.store.bundle(identifier);node=self.store.node(identifier)
            return dict(candidate_id=identifier,parent_ids=bundle['parent_ids'],hypothesis=bundle['hypothesis'],
                status=node['status'],generation=node.get('generation'),metrics=node.get('selection_metrics',{}),
                artifacts={kind:dict(candidate_id=identifier,kind=kind,offset=0) for kind in ('bundle','validation','result')
                    if (self.store.candidates/identifier/(kind+'.json')).exists()},
                instructions='Use read_artifact with next_offset until null to inspect full node code and all seed results.')
        if name=='write_candidate':
            bundle=args['bundle'];validate_bundle(bundle,parent_ids=self.parents)
            result=self.store.stage(bundle,parent_ids=self.parents,origin=self.origin);self.staged.add(result['candidate_id']);return result
        identifier=args['candidate_id']
        if identifier not in self.staged:raise ValueError('validate/submit only candidates staged during this proposal')
        if name=='validate_candidate':
            self.validation_attempts+=1
            if self.validation_attempts>4:raise ValueError('maximum 4 repair probes per proposal')
            result=self.validator(identifier,self.store.bundle(identifier))
            if result['status']=='accepted' and result['semantic_id'] in self.seen:raise ValueError('semantic duplicate: formatting, comments or ancestry are not mutations')
            self.validated[identifier]=result
            path=self.store.candidates/identifier/'validation.json';path.write_text(json.dumps(result,indent=2)+'\n')
            return result
        if name=='submit_candidate':
            result=self.validated.get(identifier,{})
            if result.get('status')!='accepted':raise ValueError('candidate needs accepted validation')
            if self.submitted:raise ValueError('one submission per proposal')
            self.submitted=identifier;return dict(status='submitted',candidate_id=identifier)
        raise ValueError('invalid tool')


def propose(client,session,context,*,max_turns=24,reasoning='max'):
    messages=[dict(role='system',content=SYSTEM),dict(role='user',content=json.dumps(context,ensure_ascii=False))]
    context_budget=max(150000,sum(len(json.dumps(m)) for m in messages)+150000)
    trace=[];usage={}
    for turn in range(max_turns):
        result=client.chat(messages,tools=TOOLS,tool_choice='required',reasoning_effort=reasoning,max_tokens=12000)
        for key,value in result.get('usage',{}).items():
            if isinstance(value,int):usage[key]=usage.get(key,0)+value
        message=result['choices'][0]['message'];calls=message.get('tool_calls',[])
        if not calls:raise ValueError('ReAct response must contain a tool call')
        messages.append(dict(role='assistant',content=message.get('content'),tool_calls=calls))
        for call in calls:
            try:
                name=call['function']['name'];args=loads(call['function']['arguments']);value=session.call(name,args)
            except Exception as exc:value=dict(status='rejected',error=type(exc).__name__,message=str(exc)[:1200])
            encoded=json.dumps(value,ensure_ascii=False,allow_nan=False)
            if len(encoded)>24000:encoded=json.dumps(dict(truncated=True,text=encoded[:22000]))
            trace.append(dict(turn=turn,tool=call['function']['name'],response=value))
            messages.append(dict(role='tool',tool_call_id=call['id'],content=encoded))
        if session.submitted:return dict(candidate_id=session.submitted,usage=usage,trace=trace)
        # Keep bounded history: artifacts preserve full candidates; tool arguments can be large.
        if sum(len(json.dumps(m)) for m in messages)>context_budget:raise ValueError('ReAct context budget exhausted')
    raise ValueError('ReAct tool-turn budget exhausted without valid submission')
