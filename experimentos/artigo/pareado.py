"""Sensibilidade à intensidade com os mesmos membros, mercados aninhados e trocas cumulativas."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import multiprocessing
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import pandas as pd
import torch

from experimentos.artigo.ablacao import classificar
from experimentos.artigo.executar import treinar_cenario
from experimentos.artigo.preparar import FORWARD, REVERSE, protocol, save_case, separar, validar_divisao


def preparar_doses(run, out, seed, split, users, events, edges):
    rng = np.random.default_rng(seed)
    labels = pd.read_csv(run / str(seed) / 'rotulos.csv').drop(columns='dose')
    assert labels.wallet_id.is_unique and labels.groupby('papel').size().eq(50).all()
    graph = split['context']
    ids = graph['user'].original_id.numpy()
    markets = graph['event'].original_id.numpy()
    user_index = dict(zip(ids, range(len(ids))))
    event_index = dict(zip(markets, range(len(markets))))
    original = set(map(tuple, graph[FORWARD].edge_index.T.tolist()))
    reserved = set(map(tuple, split['all_positive'].T.tolist())) - original
    neighbors = [set() for _ in ids]
    visitors = [set() for _ in markets]
    for u, m in original:
        neighbors[u].add(m)
        visitors[m].add(u)
    min_degree, max_degree = protocol['grau_mercado']
    pool = {m for m, ns in enumerate(visitors) if min_degree <= len(ns) <= max_degree}
    protected = set(labels.wallet_id.map(user_index))
    members = {}
    lists = {}
    for group, part in labels.query("papel=='alvo'").groupby('grupo'):
        members[group] = part.wallet_id.map(user_index).tolist()
        assert len(members[group]) == 5
        available = sorted(m for m in pool if all((u,m) not in original and (u,m) not in reserved for u in members[group]))
        lists[group] = rng.choice(available, max(protocol['mercados_novos_comuns']), replace=False).tolist()
        pool.difference_update(lists[group])
    addresses = users.set_index('wallet_id').wallet
    rows = edges.copy()
    blocks = {}
    for i, pair in enumerate(zip(rows.wallet_id.map(user_index),rows.condition_node_id.map(event_index))):
        blocks.setdefault(pair, []).append(i)
    out.mkdir()
    labels.to_csv(out/'rotulos.csv',index=False)
    log = []
    for level in range(1, max(protocol['mercados_novos_comuns'])+1):
        for group, targets in members.items():
            m = lists[group][level-1]
            for u in targets:
                donors = rng.permutation(sorted(visitors[m] - protected))
                outside = rng.permutation(sorted(neighbors[u] - set(lists[group])))
                match = next(((int(v),int(a)) for v in donors for a in outside
                              if a not in neighbors[v] and (v,a) not in reserved), None)
                if match is None:
                    raise ValueError(f'Troca inviável: {seed=}, {level=}, {group=}')
                v,a = match
                left,right = blocks.pop((u,a)),blocks.pop((v,m))
                blocks[v,a],blocks[u,m] = left,right
                rows.loc[left,['wallet_id','wallet']] = [ids[v],addresses[ids[v]]]
                rows.loc[right,['wallet_id','wallet']] = [ids[u],addresses[ids[u]]]
                neighbors[u].remove(a)
                neighbors[u].add(m)
                neighbors[v].remove(m)
                neighbors[v].add(a)
                visitors[a].remove(u)
                visitors[a].add(v)
                visitors[m].remove(v)
                visitors[m].add(u)
                log.append((level,group,ids[u],ids[v],markets[a],markets[m]))
        if level not in protocol['mercados_novos_comuns']:
            continue
        folder = out/str(level)
        folder.mkdir()
        current = {(u,m) for u, ns in enumerate(neighbors) for m in ns}
        assert current.isdisjoint(reserved) and len(current)==len(original)
        assert set(blocks)==current|reserved
        assert set(zip(rows.wallet_id.map(user_index),rows.condition_node_id.map(event_index)))==set(blocks)
        for group, targets in members.items():
            for u in targets:
                assert neighbors[u]-{m for m in neighbors[u] if (u,m) in original}==set(lists[group][:level])
        p = torch.tensor(sorted(current)).T.contiguous()
        for axis in (0,1):
            assert torch.equal(torch.bincount(p[axis]), torch.bincount(graph[FORWARD].edge_index[axis]))
        assert rows.wallet.eq(rows.wallet_id.map(addresses)).all()
        changed = copy.deepcopy(split)
        changed['context'][FORWARD].edge_index = p
        changed['context'][REVERSE].edge_index = p.flip(0).clone()
        changed['all_positive'] = torch.tensor(sorted(current|reserved)).T.contiguous()
        validar_divisao(changed)
        save_case(folder,changed,users,events,rows)
    pd.DataFrame(log,columns=['ordem','grupo','alvo','doadora','mercado_anterior','mercado_novo']).to_csv(out/'trocas.csv',index=False)


def comparar_posicoes(df):
    """Compara a posição do mesmo ID, sorteio e seed entre intensidades."""
    keys = ['realizacao', 'metodo', 'seed', 'papel']
    wide = df.pivot(index=keys + ['wallet_id'], columns='dose', values='posicao')
    assert wide.notna().all().all()
    comparisons = []
    for before, after in ((0, 4), (4, 10), (10, 20)):
        gain = (wide[before] - wide[after]).rename('ganho')
        counts = pd.DataFrame({'subiram': gain.gt(0), 'iguais': gain.eq(0), 'desceram': gain.lt(0)})
        result = counts.groupby(keys).sum()
        result['ganho_mediano_em_posicoes'] = gain.groupby(keys).median()
        assert (result[['subiram','iguais','desceram']].sum(axis=1)==50).all()
        comparisons.append(result.reset_index().assign(de=before, para=after))
    return pd.concat(comparisons, ignore_index=True)


def avaliar(run, out):
    original = classificar(run/'original')
    frames = []
    for seed in protocol['realizacoes']:
        folder = out/str(seed)
        labels = pd.read_csv(folder/'rotulos.csv').set_index('wallet_id')
        for dose in [0,*protocol['mercados_novos_comuns']]:
            ranks = classificar(folder/str(dose)) if dose else original
            for (method, model_seed), rank in ranks.items():
                frames.append(labels.assign(realizacao=seed, dose=dose, metodo=method, seed=model_seed,
                                            posicao=rank.loc[labels.index]).reset_index())
    df = pd.concat(frames,ignore_index=True)
    dest = out/'resultados'
    dest.mkdir(exist_ok=True)
    df.to_csv(dest/'trajetorias.csv',index=False)
    comparar_posicoes(df).to_csv(dest/'mudancas_pareadas.csv',index=False)
    keys=['realizacao','dose','metodo','seed','papel']
    counts=df.assign(top100=df.posicao.le(100),top541=df.posicao.le(541)).groupby(keys)[['top100','top541']].sum()
    counts.to_csv(dest/'por_sorteio_seed.csv')
    assert df.groupby(keys).size().eq(50).all()
    counts.groupby(['dose','metodo','papel']).mean().reset_index().to_csv(dest/'resumo.csv',index=False)


def executar(run, processos):
    out = Path(mkdtemp(prefix='pareado-',dir=run))
    print(out,flush=True)
    source = run/'dia'
    split = separar(torch.load(source/'hetero.pt',weights_only=False),seed=protocol['divisao'])
    frames = [pd.read_csv(source/f'{key}.csv') for key in ['user','event','edge']]
    futures=[]
    with ProcessPoolExecutor(max_workers=processos,mp_context=multiprocessing.get_context('spawn')) as pool:
        for seed in protocol['realizacoes']:
            folder=out/str(seed)
            preparar_doses(run,folder,seed,split,*frames)
            for dose in protocol['mercados_novos_comuns']:
                for model_seed in protocol['seeds_modelo']:
                    futures.append(pool.submit(treinar_cenario,(folder/str(dose),model_seed)))
        for f in futures:
            f.result()
    avaliar(run,out)
    print(out/'resultados',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rodada',type=Path)
    parser.add_argument('--processos',type=int,choices=(1,2,3),default=1)
    args=parser.parse_args()
    executar(args.rodada,args.processos)
