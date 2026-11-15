"""Monta e valida os recortes das quatro figuras (nós/arestas + atributos) e grava GEXF para o Gephi."""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd

from visualizacao_gephi.comum import (COR, LIDERES, RESULTADOS, gravar_recorte, graus, mercados, pares)

FA2_GLOBAL = dict(layout='fa2', iter=2500, iterAjuste=150, linlog='true', scaling=1.0, gravity=0.5,
                  dissuade='true', theta=1.2, threads=8, weightInfluence=0.0,
                  edgeOpacity=6, edgeThickness=0.1, labels='false', border=0.0, pdf='false')


def tamanho_log(grau, minimo=1.0, fator=3.0):
    return minimo + fator * np.log10(1 + np.asarray(grau, dtype=float))


# --------------------------------------------------------------------------- Fig. 1
def recorte_global():
    p = pares()
    gw, gm = graus()
    assert len(gw) == 27039 and len(gm) == 16693 and len(p) == 229072, (len(gw), len(gm), len(p))
    rng = np.random.default_rng(42)
    nw, nm = len(gw), len(gm)
    lider = set(LIDERES)
    nos = pd.concat([
        pd.DataFrame({'Id': 'w' + gw.index.astype(str), 'Label': [str(i) if i in lider else '' for i in gw.index],
                      'Tipo': 'carteira', 'IdOriginal': gw.index.astype('int64'), 'Grau': gw.values.astype('int64'),
                      'Lider': [i in lider for i in gw.index],
                      'color': [COR['foco'] if i in lider else COR['carteira'] for i in gw.index]}),
        pd.DataFrame({'Id': 'm' + gm.index.astype(str), 'Label': '', 'Tipo': 'mercado',
                      'IdOriginal': gm.index.astype('int64'), 'Grau': gm.values.astype('int64'), 'Lider': False,
                      'color': COR['mercado']}),
    ], ignore_index=True)
    nos['size'] = tamanho_log(nos.Grau, 1.0, 2.0)
    raio = 1000.0
    nos['x'] = rng.uniform(-raio, raio, len(nos))
    nos['y'] = rng.uniform(-raio, raio, len(nos))
    arestas = pd.DataFrame({'Source': 'w' + p.wallet_id.astype(str), 'Target': 'm' + p.condition_node_id.astype(str),
                            'Weight': 1.0, 'NTrades': p.n_trades.astype('int64'), 'color': COR['aresta'], 'alpha': 0.15})
    comp = componentes(p)
    manifesto = {
        'universo': {'carteiras': nw, 'mercados': nm, 'nos': nw + nm, 'arestas': len(p)},
        'cobertura': 'total do dia 16/09/2026 (todas as ligações observadas; sem divisão de treino)',
        'regra_aresta': 'um par carteira–mercado = uma aresta não direcionada, independentemente do nº de trades/outcomes; '
                        'sem duplicar as direções computacionais do PyG',
        'filtros': 'nenhum além da filtragem do estudo (CSV final); nenhum componente removido',
        'componentes': comp,
        'lideres': LIDERES,
        'layout': {'algoritmo': 'ForceAtlas2 (Gephi 0.10.1)', 'posicao_inicial': 'uniforme em [-1000,1000]², seed 42', **FA2_GLOBAL},
        'codificacao': {'cor': 'azul=carteira, laranja=mercado, vermelho=cinco líderes',
                        'forma': 'círculo=carteira, quadrado=mercado (composição final)',
                        'tamanho': 'proporcional a log10(1+grau)', 'arestas': 'cinza, finas, transparentes; sem peso'},
    }
    return gravar_recorte('gephi_global', nos, arestas, manifesto)


def componentes(p: pd.DataFrame) -> dict:
    import networkx as nx
    g = nx.Graph()
    g.add_edges_from(zip('w' + p.wallet_id.astype(str), 'm' + p.condition_node_id.astype(str)))
    n = g.number_of_nodes()
    tam = np.array(sorted((len(c) for c in nx.connected_components(g)), reverse=True))
    k = len(tam)
    return {'n_componentes': int(k), 'maior_componente_nos': int(tam[0]),
            'fracao_maior': round(float(tam[0] / n), 5), 'nos_fora_da_maior': int(n - tam[0]),
            'tamanhos_top10': tam[:10].tolist()}


# --------------------------------------------------------------------------- Fig. 2
def sobreposicao_dia(ids):
    p = pares()
    s = {w: set(p.loc[p.wallet_id == w, 'condition_node_id']) for w in ids}
    return s, {(a, b): len(s[a] & s[b]) for a, b in itertools.combinations(ids, 2)}


def dados_nucleo():
    s, pesos = sobreposicao_dia(LIDERES)
    inter = sorted(set.intersection(*s.values()))
    assert len(inter) == 144, len(inter)
    assert min(pesos.values()) == 153 and max(pesos.values()) == 160, pesos
    assert all(v != 144 for v in pesos.values())
    m = mercados().loc[inter, ['question', 'tags', 'categoria_principal', 'modalidade', 'evento', 'regra_evento']]
    gw, gm = graus()
    m['grau_global'] = gm.loc[inter].values
    return s, pesos, m, gw


# --------------------------------------------------------------------------- Fig. 3
CASOS = [('A', 'caso estável', 17762), ('B', 'controle do estável', 1507),
         ('C', 'caso entrante', 16179), ('D', 'controle do entrante', 53)]
ESPERADO_CASOS = {17762: (171, (4, 1), [17487, 18767, 20748]), 1507: (171, (3599, 698), [8299, 1844, 7825]),
                  16179: (41, (123, 85), [16469, 17181, 16795]), 53: (41, (6864, 6063), [4987, 409, 3924])}


def dados_casos():
    contrib = pd.read_csv(RESULTADOS / 'extra' / 'contribuicoes_casos.csv')
    casos = pd.read_csv(RESULTADOS / 'extra' / 'casos_dia_real.csv')
    p = pares()
    gw, gm = graus()
    saida = []
    for painel, titulo, foco in CASOS:
        grau, pos, parceiros = ESPERADO_CASOS[foco]
        cc = contrib[contrib.wallet_id == foco].sort_values('ordem')
        assert cc.parceiro.tolist() == parceiros, (foco, cc.parceiro.tolist())
        linha = casos[casos.wallet_id == foco].iloc[0]
        assert (int(linha.posicao_contexto), int(linha.posicao_total)) == pos
        assert int(linha.mercados_total) == grau == int(gw.loc[foco])
        mf = sorted(p.loc[p.wallet_id == foco, 'condition_node_id'])
        sub = p[p.wallet_id.isin([foco, *parceiros]) & p.condition_node_id.isin(mf)]
        saida.append(dict(painel=painel, titulo=titulo, foco=foco, parceiros=parceiros, mercados=mf,
                          arestas=sub[['wallet_id', 'condition_node_id']].copy(), contrib=cc, linha=linha,
                          grau_global={w: int(gw.loc[w]) for w in [foco, *parceiros]}))
    return saida
