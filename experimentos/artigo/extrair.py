"""Extrai de uma execução os dados pequenos que as tabelas e os gráficos do artigo usam.

Grava em resultados/artigo/extra/: divisão das ligações, treino por seed, histórico por época,
scores das ligações reservadas, fichas dos casos do dia real e a conferência com os resultados publicados.
Depois disso, tabelas e gráficos não precisam mais da pasta pesada da execução.
"""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import poisson
import torch

from experimentos.artigo.preparar import FORWARD, protocol

ROOT = Path(__file__).resolve().parents[2]
CORTE = 100
SEED_FICHAS = 42

_PREFIXOS = re.compile(r'^(will|set \d+ winner|map \d+ winner|game \d+ winner)\s+', re.I)
_SUFIXOS = re.compile(r'\s+(end in a draw|to score first|\(bo\d\)).*$|\s+-\s+.*$|\s*\?$', re.I)
_PLACAR = re.compile(r'^exact score:\s*(.+?)\s+\d+\s*-\s*\d+\s+(.+?)\??$', re.I)


def _nome(texto):
    texto = _SUFIXOS.sub('', _PREFIXOS.sub('', texto.strip()))
    return re.sub(r'\s+', ' ', texto).strip(' .?').casefold()


def evento(pergunta):
    """Chave do evento quando a pergunta nomeia um confronto entre dois lados; None se indeterminado.

    Regra: "Exact Score: A 2 - 1 B?" ou um trecho "A vs. B" (o trecho entre dois-pontos que contém "vs").
    A chave é o par de nomes normalizado, sem ordem. Perguntas sobre um só lado, eleições, preços
    e afins não identificam o evento e contam como indeterminadas.
    """
    if not isinstance(pergunta, str):
        return None
    placar = _PLACAR.match(pergunta.strip())
    if placar:
        lados = placar.groups()
    else:
        trecho = next((t for t in pergunta.split(':') if re.search(r'\svs\.?\s', t, re.I)), None)
        if trecho is None:
            return None
        lados = re.split(r'\s+vs\.?\s+', trecho, maxsplit=1, flags=re.I)
    a, b = (_nome(lado) for lado in lados)
    if not a or not b or a == b:
        return None
    return ' | '.join(sorted((a, b)))


def contribuicoes(pairs, n_users, n_events, z, u):
    """Repete o score híbrido de uma carteira e devolve cada parceiro com sua contribuição."""
    matrix = sparse.csr_matrix((np.ones(pairs.shape[1], dtype=np.int64), pairs), shape=(n_users, n_events))
    du = np.asarray(matrix.sum(axis=1)).ravel()
    de = np.asarray(matrix.sum(axis=0)).ravel()
    edges = pairs.shape[1]
    factor = np.sum(de * (de - 1)) / (edges * (edges - 1))
    overlap = (matrix[u] @ matrix.T).tocsr()
    vs, cs = overlap.indices, overlap.data
    keep = (vs != u) & (cs >= 2)
    vs, cs = vs[keep], cs[keep]
    zn = z.astype(np.float64)
    zn = zn / np.maximum(np.linalg.norm(zn, axis=1, keepdims=True), 1e-12)
    cosine = np.clip(zn[vs] @ zn[u], -1, 1)
    lam = du[u] * du[vs] * factor
    surprise = np.minimum(-poisson.logsf(cs - 1, lam) / np.log(10), 300.)
    frame = pd.DataFrame({'parceiro_indice': vs, 'sobreposicao_observada': cs, 'sobreposicao_esperada': lam,
                          'surpresa_log10': surprise, 'cosseno': cosine,
                          'contribuicao': surprise * (1 + cosine) / 2})
    return frame, int(du[u])


def _controle(cobertura, caso, ja_usados):
    fora = cobertura[(cobertura[f'total_{SEED_FICHAS}'] > CORTE) & ~cobertura.wallet_id.isin(ja_usados)]
    distancia = (fora.mercados_total - caso.mercados_total).abs()
    return fora.assign(distancia=distancia).sort_values(['distancia', 'wallet_id']).iloc[0]


def fichas(run, cobertura):
    """T6: primeiro caso que permaneceu e primeiro que entrou no top 100, cada um com um controle."""
    split = torch.load(run / 'original' / 'rede.pt', weights_only=False)
    context = split['context']
    pairs = split['all_positive'].numpy()
    ids = context['user'].original_id.numpy()
    markets = context['event'].original_id.numpy()
    with np.load(run / 'resultados' / 'embeddings_total.npz') as emb:
        np.testing.assert_array_equal(emb['user_original_id'], ids)
        z = emb[f'z_{SEED_FICHAS}']
    nos = pd.read_csv(run / 'data/raw/betgraph_nos_evento_16set_final.csv').set_index('condition_node_id')
    chave = nos.question.map(evento)
    index = {w: i for i, w in enumerate(ids)}

    ordem = cobertura.sort_values(f'total_{SEED_FICHAS}')
    no_corte = ordem[f'total_{SEED_FICHAS}'] <= CORTE
    antes = ordem[f'contexto_{SEED_FICHAS}'] <= CORTE
    casos = [('permaneceu', ordem[no_corte & antes].iloc[0]), ('entrou', ordem[no_corte & ~antes].iloc[0])]
    usados = {int(c.wallet_id) for _, c in casos}
    linhas, partes = [], []
    for tipo, caso in casos:
        controle = _controle(cobertura, caso, usados)
        usados.add(int(controle.wallet_id))
        for papel, row in (('caso', caso), ('controle', controle)):
            u = index[int(row.wallet_id)]
            frame, grau = contribuicoes(pairs, len(ids), len(markets), z, u)
            top = frame.sort_values(['contribuicao', 'parceiro_indice'], ascending=[False, True]).head(3)
            score = top.contribuicao.sum() / 3 if len(top) else 0.
            assert np.isclose(score, row[f'score_total_{SEED_FICHAS}'], rtol=1e-6, atol=1e-9), (row.wallet_id, score)
            assert grau == row.mercados_total
            eventos = chave.reindex(markets[pairs[1][pairs[0] == u]])
            linhas.append(dict(tipo=tipo, papel=papel, wallet_id=int(row.wallet_id),
                               posicao_contexto=int(row[f'contexto_{SEED_FICHAS}']),
                               posicao_total=int(row[f'total_{SEED_FICHAS}']),
                               score=float(row[f'score_total_{SEED_FICHAS}']),
                               mercados_contexto=int(row.mercados_contexto), mercados_total=grau,
                               parceiros=len(frame), eventos_distintos=int(eventos.dropna().nunique()),
                               mercados_evento_indeterminado=int(eventos.isna().sum())))
            top = top.assign(tipo=tipo, papel=papel, wallet_id=int(row.wallet_id), ordem=range(1, len(top) + 1),
                             parceiro=ids[top.parceiro_indice.to_numpy()])
            partes.append(top.drop(columns='parceiro_indice'))
    colunas = ['tipo', 'papel', 'wallet_id', 'ordem', 'parceiro', 'contribuicao', 'surpresa_log10', 'cosseno',
               'sobreposicao_observada', 'sobreposicao_esperada']
    return pd.DataFrame(linhas), pd.concat(partes, ignore_index=True)[colunas]


def divisao(run):
    split = torch.load(run / 'original' / 'rede.pt', weights_only=False)
    context = split['context']
    contagem = {k: int(split[f'{k}_positive'].shape[1]) for k in ('train', 'val', 'test')}
    total = int(split['all_positive'].shape[1])
    resultado = dict(dia=protocol['dia'], carteiras=int(context['user'].num_nodes),
                     mercados=int(context['event'].num_nodes), ligacoes=total,
                     contexto=int(context[FORWARD].edge_index.shape[1]), treino=contagem['train'],
                     validacao=contagem['val'], teste=contagem['test'],
                     seed_divisao=protocol['divisao'], seeds_modelo=protocol['seeds_modelo'],
                     realizacoes=protocol['realizacoes'])
    assert resultado['contexto'] + sum(contagem.values()) == total
    users = context[FORWARD].edge_index[0].unique().numel()
    events = context[FORWARD].edge_index[1].unique().numel()
    resultado['carteiras_com_contexto'] = int(users)
    resultado['mercados_com_contexto'] = int(events)
    return resultado


def treinos(run):
    linhas, historicos = [], []
    for seed in protocol['seeds_modelo']:
        pasta = run / 'original' / f'gnn_{seed}'
        m = json.loads((pasta / 'manifest.json').read_text())
        linhas.append(dict(seed=seed, epocas_executadas=m['epochs_run'], melhor_epoca=m['best_epoch'],
                           segundos=m['seconds'], parametros=m['parameters'],
                           val_auc=m['validation']['auc'], val_ap=m['validation']['ap'],
                           val_loss=m['validation']['loss']))
        historicos.append(pd.read_csv(pasta / 'historico.csv').assign(seed=seed))
    return pd.DataFrame(linhas), pd.concat(historicos, ignore_index=True)


def conferir(run, publicados, casos):
    """Compara a cobertura reproduzida com a publicada na main, carteira a carteira.

    Diferenças de ponto flutuante entre máquinas mudam posições na cauda, entre scores quase empatados;
    o que importa ao artigo são os cortes, os líderes, as métricas e as posições dos casos.
    """
    nova = pd.read_csv(run / 'resultados' / 'cobertura_total.csv').set_index('wallet_id').sort_index()
    antiga = pd.read_csv(publicados / 'cobertura_total.csv').set_index('wallet_id').sort_index()
    assert nova.index.equals(antiga.index)
    aval_nova = json.loads((run / 'resultados/avaliacao_cobertura.json').read_text())
    aval_antiga = json.loads((publicados / 'avaliacao_cobertura.json').read_text())
    por_seed = {}
    for seed in map(str, protocol['seeds_modelo']):
        r = {}
        for k in ('contexto', 'total'):
            c = f'{k}_{seed}'
            for corte in (100, 541):
                r[f'{k}_top{corte}_identico'] = set(nova.index[nova[c] <= corte]) == set(antiga.index[antiga[c] <= corte])
            r[f'{k}_posicoes_diferentes'] = int((nova[c] != antiga[c]).sum())
        r['maior_diferenca_score'] = float((nova[f'score_total_{seed}'] - antiga[f'score_total_{seed}']).abs().max())
        for m in ('auc', 'ap'):
            r[f'diferenca_{m}'] = abs(aval_nova[seed][m] - aval_antiga[seed][m])
        r['lideres_e_comuns_identicos'] = all(aval_nova[seed][k] == aval_antiga[seed][k]
                                              for k in ('top100_comuns', 'primeiro_contexto', 'primeiro_total', 'n'))
        por_seed[seed] = r
    ids = casos.wallet_id.tolist()
    s = SEED_FICHAS
    casos_identicos = bool((nova.loc[ids, [f'contexto_{s}', f'total_{s}']]
                            == antiga.loc[ids, [f'contexto_{s}', f'total_{s}']]).all().all())
    return dict(por_seed=por_seed, posicoes_dos_casos_identicas=casos_identicos,
                cortes_identicos=all(v for r in por_seed.values() for k, v in r.items() if k.endswith('_identico')))


def extrair(run, destino):
    destino.mkdir(parents=True, exist_ok=True)
    publicados = ROOT / 'resultados' / 'artigo'
    (destino / 'divisao.json').write_text(json.dumps(divisao(run), indent=2) + '\n')
    treino, historico = treinos(run)
    treino.to_csv(destino / 'treino_por_seed.csv', index=False)
    historico.to_csv(destino / 'historico_treino.csv', index=False)
    scores = pd.read_csv(run / 'resultados' / 'scores_teste.csv.gz')
    scores[['seed', 'rotulo', 'score']].round({'score': 6}).to_csv(destino / 'scores_teste.csv.gz', index=False)
    casos, partes = fichas(run, pd.read_csv(run / 'resultados' / 'cobertura_total.csv'))
    casos.to_csv(destino / 'casos_dia_real.csv', index=False)
    partes.to_csv(destino / 'contribuicoes_casos.csv', index=False)
    verificacao = dict(execucao=run.name, **conferir(run, publicados, casos))
    (destino / 'verificacao.json').write_text(json.dumps(verificacao, indent=2) + '\n')
    print(json.dumps(verificacao, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rodada', type=Path)
    parser.add_argument('--destino', type=Path, default=ROOT / 'resultados' / 'artigo' / 'extra')
    args = parser.parse_args()
    extrair(args.rodada, args.destino)
