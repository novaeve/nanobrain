"""Relaxed archives: quality elites, Pareto alternatives and structural novelty."""
from __future__ import annotations
from .contracts import digest


def utility_vector(node):
    m=node['selection_metrics']
    return [m['annotation.f1_macro'],m['integration.scib_composite'],
        1/(1+m['spatial_niche.composition_mae']),1/(1+m['development.wasserstein2'])]


def select_parents(nodes,store,limit=12):
    valid=[n for n in nodes if n.get('status')=='complete' and n.get('all_seeds_passed')]
    if not valid:return []
    score=lambda n:sum(utility_vector(n))/4
    ordered=sorted(valid,key=score,reverse=True)
    pareto=[]
    for n in ordered:
        a=utility_vector(n)
        if not any(all(x>=y for x,y in zip(utility_vector(o),a)) and any(x>y for x,y in zip(utility_vector(o),a)) for o in valid):pareto.append(n)
    picked=ordered[:max(1,limit//3)]
    signatures=set()
    for n in picked:signatures.add(topology(store.bundle(n['candidate_id'])))
    for pool in (pareto,ordered[::-1],ordered):
        for n in pool:
            if len(picked)>=limit:break
            if n in picked:continue
            signature=topology(store.bundle(n['candidate_id']))
            if pool is not ordered and signature in signatures:continue
            picked.append(n);signatures.add(signature)
    return [n['candidate_id'] for n in picked]


def topology(bundle):
    a=bundle['genome']['architecture']
    return digest({k:[(n['kind'],n['name'],n['inputs']) for n in a[k]['nodes']] for k in ('backbone','flow')})
