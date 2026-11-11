"""Append-only lineage and content-addressed candidate artifacts."""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
from .contracts import digest, semantic_id, validate_bundle, artifact_path


def change_outline(bundle,parent):
    if parent is None:return 'reference/root'
    def modules(value):
        architecture=value['genome']['architecture'];counts=Counter()
        for graph in ('backbone','flow'):
            definition=architecture.get(graph,{})
            if not isinstance(definition,dict):continue
            nodes=definition.get('nodes',[])
            if not isinstance(nodes,list):continue
            for node in nodes:
                if not isinstance(node,dict):continue
                repeat=node.get('repeat',1);repeat=repeat if type(repeat)is int and repeat>0 else 1
                counts[str(node.get('kind','invalid'))+'/'+str(node.get('name','invalid'))]+=repeat
        heads=architecture.get('heads',{})
        if isinstance(heads,dict):
            for head in heads.values():
                if isinstance(head,dict):counts['head/'+str(head.get('name','invalid'))]+=1
        return counts
    current,previous=modules(bundle),modules(parent);parts=[]
    parts.extend('+'+name+':'+str(n) for name,n in sorted((current-previous).items()))
    parts.extend('-'+name+':'+str(n) for name,n in sorted((previous-current).items()))
    if bundle['genome']['architecture']!=parent['genome']['architecture']:parts.append('graph/parameters')
    if bundle['genome']['architecture'].get('objectives')!=parent['genome']['architecture'].get('objectives'):parts.append('loss')
    if bundle['genome']['learning_rate']!=parent['genome']['learning_rate']:parts.append('lr='+str(bundle['genome']['learning_rate']))
    if bundle['optimizer']!=parent['optimizer']:parts.append('optimizer='+bundle['optimizer']['name'])
    if bundle['files']!=parent['files']:parts.append('custom-code')
    return ';'.join(parts)[:120] or 'metadata-only / inspect semantic validation'


class Store:
    def __init__(self, root):
        self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.candidates=self.root/'candidates';self.candidates.mkdir(exist_ok=True)

    def stage(self,bundle,parent_ids=None,origin=None):
        validate_bundle(bundle,parent_ids=parent_ids)
        semantic=semantic_id(bundle)
        identifier=digest(bundle);directory=self.candidates/identifier;directory.mkdir(exist_ok=True)
        path=directory/'bundle.json'
        if path.exists() and json.loads(path.read_text())!=bundle:raise ValueError('immutable artifact collision')
        if not path.exists():path.write_text(json.dumps(bundle,indent=2,sort_keys=True)+'\n')
        if origin is not None and not (directory/'origin.json').exists():
            (directory/'origin.json').write_text(json.dumps(origin,sort_keys=True)+'\n')
        return dict(candidate_id=identifier,semantic_id=semantic,bundle_sha256=digest(bundle))

    def resolve_id(self,identifier):
        if not isinstance(identifier,str) or not 12<=len(identifier)<=64 or any(c not in '0123456789abcdef' for c in identifier):
            raise ValueError('invalid candidate artifact reference')
        if len(identifier)==64:return identifier
        matches=[p.name for p in self.candidates.glob(identifier+'*') if (p/'bundle.json').is_file()]
        if len(matches)!=1:raise ValueError('unknown or ambiguous candidate artifact reference')
        return matches[0]

    def bundle(self,identifier):
        if len(identifier)!=64 or any(c not in '0123456789abcdef' for c in identifier):raise ValueError('invalid artifact identifier')
        path=artifact_path(self.candidates,identifier+'/bundle.json');data=json.loads(path.read_text())
        if digest(data)!=identifier:raise ValueError('candidate artifact was mutated')
        return data

    def node(self,identifier):
        identifier=self.resolve_id(identifier)
        self.bundle(identifier)
        p=self.candidates/identifier/'result.json'
        if p.exists():return json.loads(p.read_text())
        folder=p.parent;origin=folder/'origin.json';validation=folder/'validation.json'
        metadata=json.loads(origin.read_text()) if origin.exists() else {}
        status='not_evaluated'
        if validation.exists():
            status='validated' if json.loads(validation.read_text()).get('status')=='accepted' else 'rejected_validation'
        return dict(candidate_id=identifier,status=status,**metadata)

    def put_result(self,identifier,result):
        self.bundle(identifier);p=self.candidates/identifier/'result.json'
        if p.exists() and json.loads(p.read_text())!=result:raise ValueError('evaluation results are immutable')
        if not p.exists():p.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')

    def nodes(self):
        return [self.node(p.name) for p in sorted(self.candidates.iterdir()) if (p/'bundle.json').exists()]

    def tree_summary(self,limit=None):
        """Every node and parent edge; compact node contents, never a top-k tree.

        ``limit`` is retained for callers from the prepared version, but it does
        not truncate the lineage. Short IDs are unambiguous tool references.
        """
        nodes=self.nodes();ids={n['candidate_id'] for n in nodes};width=12
        while len({i[:width] for i in ids})!=len(ids):width+=1
        metric_keys=['annotation.f1_macro','integration.scib_composite',
                     'spatial_niche.composition_mae','development.wasserstein2']
        result=[];edges=[];bundles={identifier:self.bundle(identifier) for identifier in ids}
        for n in sorted(nodes,key=lambda n:(n.get('generation') if n.get('generation') is not None else 10**9,n['candidate_id'])):
            identifier=n['candidate_id'];bundle=bundles[identifier]
            parents=[p[:width] for p in bundle['parent_ids'] if p!='0'*64]
            edges.extend([p,identifier[:width]] for p in parents)
            metrics=n.get('selection_metrics',{})
            result.append(dict(id=identifier[:width],p=parents,g=n.get('generation'),s=n['status'],
                m=[round(metrics[k],6) if k in metrics else None for k in metric_keys],
                d=change_outline(bundle,bundles.get(bundle['parent_ids'][0])),
                h=' '.join(bundle['hypothesis'].split())[:48],
                plugins=len(bundle['plugins'])))
            if n.get('reason'):result[-1]['e']=n['reason'][:72]
        return dict(format='complete_lineage_outline_v2',complete=True,truncated=False,
            total_nodes=len(nodes),evaluated_nodes=sum(n.get('status')=='complete' for n in nodes),
            fields=dict(id='unique artifact reference; accepted by lineage/read_artifact',p='all parents',
                        g='generation (0=reference)',s='status',m=metric_keys,d='actual change outline versus first parent',
                        h='proposer hypothesis outline',plugins='custom plugin count; no numeric cap',e='rejection outline'),
            edges=edges,nodes=result,
            detail_tools=['lineage(candidate_id)','read_artifact(candidate_id, kind, offset)'])

    def read_text(self,identifier,kind='bundle',offset=0,limit=8000):
        if kind not in {'bundle','validation','result'} or type(offset)is not int or offset<0 or not 1<=limit<=16000:raise ValueError('invalid bounded artifact read')
        identifier=self.resolve_id(identifier)
        self.bundle(identifier);p=self.candidates/identifier/(kind+'.json')
        text=p.read_text();return dict(candidate_id=identifier,text=text[offset:offset+limit],offset=offset,
            next_offset=offset+limit if offset+limit<len(text) else None,total_chars=len(text),truncated=offset+limit<len(text))
