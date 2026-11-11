"""Cria os controles positivos por trocas de arestas no contexto do dia."""
import copy
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from experimentos.piloto_sintetico.comum import legacy_imports
legacy_imports()
from preparar_gnn import FORWARD, REVERSE, separar
from features_contexto import features_contexto
from treinar_gnn import validar_divisao
from experimentos.gnn_v2.features import enriquecer
from experimentos.gnn_v2.detectores import ranking

protocol = json.loads(Path(__file__).with_name('protocolo.json').read_text())

torch.set_num_threads(2)

def save_case(folder, split, users, events, edges):
    base, _, _, _ = features_contexto(split, users, events, edges)
    enriched, _, stats = enriquecer(base)
    ids = base['context']['user'].original_id.numpy()
    ranking(ids, stats['surpresa_coparticipacao']).to_csv(folder/'relacional.csv', index=False)
    torch.save(enriched, folder/'rede.tmp')
    (folder/'rede.tmp').replace(folder/'rede.pt')

def intervene(out, split, users, events, edges, seed, index):
    rng = np.random.default_rng(seed)
    folder = out/str(seed)
    folder.mkdir(exist_ok=True)
    graph = split['context']
    uids = graph['user'].original_id.numpy()
    mids = graph['event'].original_id.numpy()
    coords = graph[FORWARD].edge_index.numpy()
    original = set(map(tuple, coords.T.tolist()))
    reserved = set(map(tuple, split['all_positive'].T.tolist())) - original
    u_neigh = defaultdict(set)
    m_neigh = defaultdict(set)
    for u,m in original:
        u_neigh[u].add(m)
        m_neigh[m].add(u)
    degrees = np.bincount(coords[0], minlength=len(uids))
    lo,hi = protocol['grau_carteira']
    eligible = np.flatnonzero((degrees>=lo)&(degrees<=hi))
    num_groups = protocol['grupos_por_realizacao']
    group_size = protocol['carteiras_por_grupo']
    targets = rng.choice(eligible, num_groups * group_size, replace=False).reshape(num_groups, group_size)
    protected = set(targets.flatten())
    controls = {}
    for u in targets.flatten():
        v = int(rng.choice([x for x in eligible if degrees[x]==degrees[u] and x not in protected]))
        controls[u]=v
        protected.add(v)
    levels = protocol['mercados_novos_comuns']
    doses = levels * (num_groups // len(levels)) + [levels[index]]
    rng.shuffle(doses)
    lo,hi=protocol['grau_mercado']
    pool = {m for m,ns in m_neigh.items() if lo<=len(ns)<=hi}
    rows = edges.copy()
    uid_index = dict(zip(uids,range(len(uids))))
    market_index = dict(zip(mids,range(len(mids))))
    blocks = defaultdict(list)
    for i,key in enumerate(zip(rows.wallet_id.map(uid_index), rows.condition_node_id.map(market_index))):
        blocks[key].append(i)
    addresses = users.set_index('wallet_id').wallet
    labels, changes, markets = [],[],[]
    for group,(members,k) in enumerate(zip(targets,doses)):
        options = sorted(m for m in pool if all((u,m) not in original and (u,m) not in reserved for u in members))
        selected = list(map(int,rng.choice(options,k,replace=False)))
        pool.difference_update(selected)
        markets.extend((group,k,int(mids[m])) for m in selected)
        for u in members:
            v=controls[u]
            labels.extend([(int(uids[u]),group,k,'alvo',int(degrees[u])),(int(uids[v]),group,k,'controle',int(degrees[v]))])
            for m in selected:
                donors = rng.permutation(sorted(m_neigh[m]-protected))
                outside = rng.permutation(sorted(u_neigh[u]-set(selected)))
                match=next(((int(d),int(a)) for d in donors for a in outside if a not in u_neigh[d] and (d,a) not in reserved),None)
                if match is None:
                    raise ValueError(f'Troca inviável, grupo {group}, mercado {m}')
                d,a=match
                left,right=blocks.pop((u,a)),blocks.pop((d,m))
                blocks[d,a],blocks[u,m]=left,right
                rows.loc[left,['wallet_id','wallet']]=[uids[d],addresses[uids[d]]]
                rows.loc[right,['wallet_id','wallet']]=[uids[u],addresses[uids[u]]]
                u_neigh[d].remove(m)
                u_neigh[d].add(a)
                u_neigh[u].remove(a)
                u_neigh[u].add(m)
                m_neigh[m].remove(d)
                m_neigh[m].add(u)
                m_neigh[a].remove(u)
                m_neigh[a].add(d)
                changes.append((group,k,int(uids[u]),int(uids[d]),int(mids[a]),int(mids[m])))
        assert set(selected).issubset(set.intersection(*(u_neigh[u] for u in members)))
    current={(u,m) for u,ns in u_neigh.items() for m in ns}
    assert len(current)==len(original) and not current.intersection(reserved)
    p=torch.tensor(sorted(current)).T.contiguous()
    for axis,n in enumerate((len(uids),len(mids))):
        assert torch.equal(torch.bincount(p[axis],minlength=n),torch.bincount(graph[FORWARD].edge_index[axis],minlength=n))
    assert all(u_neigh[v]=={m for m in u_neigh[v] if (v,m) in original} for v in controls.values())
    assert set(zip(rows.wallet_id.map(uid_index),rows.condition_node_id.map(market_index)))==current|reserved
    pd.testing.assert_frame_equal(rows.drop(columns=['wallet_id','wallet']),edges.drop(columns=['wallet_id','wallet']))
    assert rows.wallet.eq(rows.wallet_id.map(addresses)).all()
    altered=copy.deepcopy(split)
    altered['context'][FORWARD].edge_index=p
    altered['context'][REVERSE].edge_index=p.flip(0).clone()
    parts=['train_positive','val_positive','test_positive']
    altered['all_positive']=torch.unique(torch.cat([p]+[split[key] for key in parts],dim=1),dim=1)
    validar_divisao(altered)
    pd.DataFrame(labels,columns=['wallet_id','grupo','dose','papel','grau']).to_csv(folder/'rotulos.csv',index=False)
    pd.DataFrame(changes,columns=['grupo','dose','alvo','doadora','mercado_anterior','mercado_novo']).to_csv(folder/'trocas.csv',index=False)
    pd.DataFrame(markets,columns=['grupo','dose','mercado']).to_csv(folder/'mercados.csv',index=False)
    save_case(folder,altered,users,events,rows)

def preparar(source, out):
    split=separar(torch.load(source/'hetero.pt',weights_only=False),seed=protocol['divisao'])
    frames=[pd.read_csv(source/f'{key}.csv') for key in ['user','event','edge']]
    original=out/'original'
    original.mkdir(exist_ok=True)
    if not (original/'rede.pt').exists():
        save_case(original,split,*frames)
    for i,seed in enumerate(protocol['realizacoes']):
        if not (out/str(seed)/'rede.pt').exists():
            intervene(out,split,*frames,seed,i)
        print('Preparado',seed,flush=True)
