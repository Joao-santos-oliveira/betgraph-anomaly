"""Layouts determinísticos (anéis ordenados por modalidade/evento) usados nas Figs. 2–4."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from matplotlib.patches import Wedge

from visualizacao_gephi.comum import COR_MODALIDADE, ORDEM_MODALIDADE


def ordenar_mercados(meta: pd.DataFrame, chave_extra: pd.Series | None = None) -> list:
    """Ordena por modalidade, depois evento (eventos maiores primeiro, 'indeterminado' por último)."""
    m = meta.copy()
    m['_om'] = m.modalidade.map(ORDEM_MODALIDADE)
    tam = m.groupby(['modalidade', 'evento']).evento.transform('size')
    m['_indet'] = (m.evento == 'indeterminado').astype(int)
    m['_tam'] = -tam
    m['_extra'] = 0 if chave_extra is None else -chave_extra.reindex(m.index).fillna(0)
    m['_id'] = m.index
    m = m.sort_values(['_om', '_indet', '_tam', 'evento', '_extra', '_id'])
    return m.index.tolist()


def anel(ordem: list, grupos: pd.Series, raio=1.0, inicio=90.0, lacuna_grupo=2.5, sentido=-1,
         sub: pd.Series | None = None, lacuna_sub=0.0):
    """Posições num círculo com lacuna angular entre grupos (e, opcionalmente, entre subgrupos/eventos).

    Retorna pos {k: (x, y, ângulo)} e arcos [(grupo, ângulo_inicial, ângulo_final)].
    """
    n = len(ordem)
    trocas_g = [i for i in range(1, n) if grupos[ordem[i]] != grupos[ordem[i - 1]]]
    trocas_s = [i for i in range(1, n) if sub is not None and i not in trocas_g
                and sub[ordem[i]] != sub[ordem[i - 1]]]
    livre = 360.0 - lacuna_grupo * (len(trocas_g) + 1) - lacuna_sub * len(trocas_s)
    passo = livre / max(n, 1)
    tg, ts = set(trocas_g), set(trocas_s)
    pos, arcos = {}, []
    ang = inicio + sentido * lacuna_grupo / 2
    ini = ang
    for i, k in enumerate(ordem):
        if i in tg:
            arcos.append((grupos[ordem[i - 1]], ini, ang))
            ang += sentido * lacuna_grupo
            ini = ang
        elif i in ts:
            ang += sentido * lacuna_sub
        a = ang + sentido * passo / 2
        pos[k] = (raio * math.cos(math.radians(a)), raio * math.sin(math.radians(a)), a)
        ang += sentido * passo
    if n:
        arcos.append((grupos[ordem[-1]], ini, ang))
    return pos, arcos


def desenhar_faixas(ax, arcos, raio, largura, cores=COR_MODALIDADE, rotulos=None, z=1, fonte=8, min_graus=18):
    """Faixa colorida por modalidade fora do anel; rótulo curto quando o arco é grande o bastante."""
    for g, a0, a1 in arcos:
        t1, t2 = sorted((a0, a1))
        cor = cores.get(g, '#EEEEEE')
        w = Wedge((0, 0), raio + largura, t1, t2, width=largura, facecolor=cor, edgecolor='white', lw=0.6, zorder=z)
        if g == 'indeterminado':
            w.set_hatch('//////')
            w.set_edgecolor('#BDBDBD')
            w.set_linewidth(0.3)
        ax.add_patch(w)
        if rotulos is not None and (t2 - t1) >= min_graus and g in rotulos:
            meio = math.radians((t1 + t2) / 2)
            r = raio + largura + 0.08
            x, y = r * math.cos(meio), r * math.sin(meio)
            rot = math.degrees(meio) - 90
            if math.sin(meio) < 0:
                rot += 180
            ax.text(x, y, rotulos[g], rotation=rot, rotation_mode='anchor', ha='center', va='center',
                    fontsize=fonte, color='#333333', zorder=z + 1)


def poligono(n, raio, inicio=90.0):
    return [(raio * math.cos(math.radians(inicio + 360 * i / n)), raio * math.sin(math.radians(inicio + 360 * i / n)))
            for i in range(n)]
