"""Composição final das quatro figuras (PDF vetorial, SVG editável, PNG 600 dpi) + projetos Gephi por painel.

Uso, a partir da raiz:  python -m visualizacao_gephi.figuras [casos nucleo intervencao global]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle

from visualizacao_gephi import gephi
from visualizacao_gephi import recortes as R
from visualizacao_gephi.comum import (COR, COR_MODALIDADE, FIGURAS, LIDERES, ORDEM_MODALIDADE, RECORTES,
                                      escrever_gexf, graus, mercados, pares)
from visualizacao_gephi.estilo import CM, LARGURA, inteiro, legenda_marcadores, letra_painel, num, salvar
from visualizacao_gephi.layout import anel, desenhar_faixas, ordenar_mercados, poligono

HALO = [pe.withStroke(linewidth=2.2, foreground='white')]
GEPHI_DIR = FIGURAS / 'gephi'


def tam_carteira(grau):
    """Área do marcador (pt²) ∝ log10(1+grau global); mesma escala em todas as figuras."""
    return 6 + 14 * np.log10(1 + np.asarray(grau, float))


def exportar_gephi(nome: str, nos: list[dict], arestas: list[dict]):
    """Grava GEXF do painel com as mesmas coordenadas e gera .gephi + prévia nativa do Gephi."""
    pasta = RECORTES / nome.split('__')[0]
    pasta.mkdir(parents=True, exist_ok=True)
    dn, de = pd.DataFrame(nos), pd.DataFrame(arestas)
    dn.to_csv(pasta / f'{nome}_nos.csv', index=False)
    de.to_csv(pasta / f'{nome}_arestas.csv', index=False)
    gexf = pasta / f'{nome}.gexf'
    escrever_gexf(gexf, dn.assign(x=dn.x * 400, y=dn.y * 400), de)
    gephi.renderizar(gexf, GEPHI_DIR / nome, memoria='2g', layout='fixo', edgeOpacity=60, edgeThickness=0.6,
                     labels='true', labelSize=12, border=0.4)


# =============================================================================================== Fig. 3
def fig_casos(com_gephi=True):
    dados = R.dados_casos()
    meta_all = mercados()
    gw, gm = graus()
    fig = plt.figure(figsize=(LARGURA, 10.0 * CM))
    # grade 2×2; cada painel: grafo (esq.) + quadro (dir.)
    H = 0.385
    origens = [(0.0, 0.595), (0.5, 0.595), (0.0, 0.185), (0.5, 0.185)]
    R_ANEL, R_PARC = 1.0, 0.50
    ang_parc = [90, 210, 330]
    modal_presentes = set()
    resumo = []
    for (x0, y0), d in zip(origens, dados):
        ax = fig.add_axes([x0 + 0.004, y0, 0.229, H])
        axq = fig.add_axes([x0 + 0.240, y0, 0.245, H + 0.02])
        axq.axis('off')
        ax.set_aspect('equal')
        ax.axis('off')
        foco, parceiros, mf = d['foco'], d['parceiros'], d['mercados']
        meta = meta_all.loc[mf]
        ar = d['arestas']
        por_mercado = ar[ar.wallet_id.isin(parceiros)].groupby('condition_node_id').wallet_id.nunique()
        ordem = ordenar_mercados(meta, por_mercado)
        pos, arcos = anel(ordem, meta.modalidade, raio=R_ANEL, lacuna_grupo=3.0)
        modal_presentes |= set(meta.modalidade)
        pw = {foco: (0.0, 0.0)}
        for w, a in zip(parceiros, ang_parc):
            pw[w] = (R_PARC * math.cos(math.radians(a)), R_PARC * math.sin(math.radians(a)))
        # arestas: foco (raios) e parceiros
        seg_f = [[pw[foco], pos[m][:2]] for m in mf]
        ax.add_collection(LineCollection(seg_f, colors=COR['foco'], linewidths=0.3, alpha=0.30, zorder=1))
        seg_p = [[pw[w], pos[m][:2]] for w, m in zip(ar.wallet_id, ar.condition_node_id) if w != foco]
        ax.add_collection(LineCollection(seg_p, colors=COR['parceiro'], linewidths=0.25, alpha=0.32, zorder=2))
        desenhar_faixas(ax, arcos, R_ANEL + 0.07, 0.09, z=0)
        comp = [m for m in mf if por_mercado.get(m, 0) > 0]
        excl = [m for m in mf if por_mercado.get(m, 0) == 0]
        for grupo, cor, ec in ((excl, '#FFFFFF', '#8C8C8C'), (comp, COR['compartilhado'], '#00563F')):
            if grupo:
                xy = np.array([pos[m][:2] for m in grupo])
                ax.scatter(xy[:, 0], xy[:, 1], s=7, marker='s', c=cor, edgecolors=ec, linewidths=0.35, zorder=4)
        ax.scatter([0], [0], s=tam_carteira(gw[foco]), c=COR['foco'], edgecolors='white', linewidths=0.6, zorder=6)
        for w in parceiros:
            ax.scatter(*pw[w], s=tam_carteira(gw[w]), c=COR['parceiro'], edgecolors='white', linewidths=0.6, zorder=6)
            dx, dy = pw[w]
            ax.text(dx * 1.42, dy * 1.42 + (0.02 if dy >= 0 else -0.02), str(w), fontsize=8, ha='center',
                    va='center', color=COR['parceiro'], path_effects=HALO, zorder=7)
        ax.text(0, -0.15, str(foco), fontsize=8, fontweight='bold', ha='center', va='top', color=COR['foco'],
                path_effects=HALO, zorder=7)
        lim = R_ANEL + 0.2
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        # quadro
        ln = d['linha']
        c = d['contrib']
        n3 = int((por_mercado == 3).sum())
        axq.text(0.0, 1.0, d['painel'], fontsize=10, fontweight='bold', va='top')
        axq.text(0.085, 0.992, d['titulo'], fontsize=8.5, va='top')
        passo = 0.088
        linhas = [
            ('mercados do foco', f'{len(mf)}'),
            ('score híbrido (seed 42)', num(ln.score, 2)),
            ('posição ctx→total', f'{inteiro(ln.posicao_contexto)} → {inteiro(ln.posicao_total)}'),
            ('compartilhados ≥1 | 3', f'{len(comp)} | {n3}'),
        ]
        y = 0.87
        for k, v in linhas:
            axq.text(0.0, y, k, fontsize=8, va='top', color='#444444')
            axq.text(1.0, y, v, fontsize=8, va='top', ha='right', color=COR['texto'])
            y -= passo
        y -= 0.025
        axq.plot([0, 1], [y + 0.012, y + 0.012], color='#9E9E9E', lw=0.5)
        axq.text(0.0, y, 'parceiro', fontsize=8, va='top', color='#444444')
        axq.text(0.70, y, 'obs. / esp.', fontsize=8, va='top', ha='right', color='#444444')
        axq.text(1.0, y, 'grau', fontsize=8, va='top', ha='right', color='#444444')
        y -= passo
        for r in c.itertuples():
            axq.text(0.0, y, str(r.parceiro), fontsize=8, va='top', color=COR['parceiro'])
            axq.text(0.70, y, f'{int(r.sobreposicao_observada)} / {num(r.sobreposicao_esperada, 1)}', fontsize=8,
                     va='top', ha='right')
            axq.text(1.0, y, str(gw[r.parceiro]), fontsize=8, va='top', ha='right', color='#444444')
            y -= passo
        axq.set_xlim(0, 1)
        axq.set_ylim(0, 1)
        resumo.append(dict(painel=d['painel'], foco=foco, mercados=len(mf), arestas_desenhadas=len(ar),
                           compartilhados_ge1=len(comp), compartilhados_3=n3,
                           score=float(ln.score), parceiros=parceiros))
        # recorte + Gephi
        nos = [dict(Id=f'w{foco}', Label=str(foco), Tipo='carteira', Papel='foco', GrauGlobal=int(gw[foco]),
                    Modalidade='', Evento='', x=0.0, y=0.0, size=12.0, color=COR['foco'])]
        nos += [dict(Id=f'w{w}', Label=str(w), Tipo='carteira', Papel='parceiro', GrauGlobal=int(gw[w]),
                     Modalidade='', Evento='', x=pw[w][0], y=pw[w][1], size=9.0, color=COR['parceiro'])
                for w in parceiros]
        nos += [dict(Id=f'm{m}', Label='', Tipo='mercado',
                     Papel='compartilhado' if m in comp else 'exclusivo', GrauGlobal=int(gm[m]),
                     Modalidade=meta.loc[m, 'modalidade'], Evento=meta.loc[m, 'evento'],
                     x=pos[m][0], y=pos[m][1], size=3.0,
                     color=COR['compartilhado'] if m in comp else '#BDBDBD') for m in ordem]
        arestas = [dict(Source=f'w{w}', Target=f'm{m}', Weight=1.0,
                        color=COR['foco'] if w == foco else COR['parceiro'], alpha=0.5)
                   for w, m in zip(ar.wallet_id, ar.condition_node_id)]
        if com_gephi:
            exportar_gephi(f'gephi_casos__{d["painel"]}_{foco}', nos, arestas)
    # legenda comum
    axl = fig.add_axes([0.0, 0.0, 1.0, 0.17])
    axl.axis('off')
    leg = legenda_marcadores(axl, [
        ('foco', 'o', COR['foco'], 'white', 6.5), ('parceiros (3 maiores contribuições)', 'o', COR['parceiro'], 'white', 5.5),
        ('mercado compartilhado com ≥1 parceiro', 's', COR['compartilhado'], '#00563F', 4.5),
        ('mercado só do foco', 's', 'white', '#8C8C8C', 4.5)],
        loc='upper left', bbox_to_anchor=(0.0, 1.0), ncol=4, columnspacing=1.4)
    axl.add_artist(leg)
    mods = sorted(modal_presentes, key=ORDEM_MODALIDADE.get)
    from matplotlib.patches import Patch
    h = [Patch(facecolor=COR_MODALIDADE[m], edgecolor='#BDBDBD' if m == 'indeterminado' else 'none',
               hatch='//////' if m == 'indeterminado' else None, label=m, lw=0.3) for m in mods]
    axl.text(0.0, 0.38, 'faixa externa =\nmodalidade', fontsize=8, va='center', color='#444444')
    axl.legend(handles=h, loc='center left', bbox_to_anchor=(0.105, 0.36), ncol=math.ceil(len(mods) / 3),
               frameon=False, handlelength=1.2, handletextpad=0.35, columnspacing=0.9, labelspacing=0.15)
    meta_fig = {
        'titulo': 'Casos e controles — quatro painéis', 'figura': 'gephi_casos',
        'universo': 'cobertura total do dia (229.072 pares distintos); por painel: todos os mercados do foco e as '
                    'arestas do foco e dos três parceiros para esses mercados',
        'seed': 'score híbrido da seed 42 (resultados/artigo/extra/contribuicoes_casos.csv, casos_dia_real.csv)',
        'selecao': 'focos e parceiros conforme o roteiro; parceiros = três maiores contribuições ao score',
        'filtros': 'arestas dos parceiros para mercados fora do foco omitidas (grau desenhado ≠ grau global)',
        'layout': 'anel determinístico: mercados ordenados por modalidade → evento → nº de parceiros; foco no centro, '
                  'parceiros em 90°/210°/330° (r=0,5); mesmas coordenadas no projeto Gephi 0.10.1',
        'codificacao': {'cor': 'vermelho=foco, roxo=parceiros, verde=mercado compartilhado, branco=só do foco; '
                               'faixa externa = modalidade (pastel; hachura = indeterminado)',
                        'tamanho': 'carteiras ∝ log10(1+grau global), mesma escala nos quatro painéis; mercados fixos',
                        'arestas': 'coparticipação (sem direção); vermelho=foco, roxo=parceiros'},
        'paineis': resumo,
    }
    return salvar(fig, 'gephi_casos', meta_fig)


# =============================================================================================== Fig. 2
def fig_nucleo(com_gephi=True):
    s_dia, pesos, meta, gw = R.dados_nucleo()
    inter = meta.index.tolist()
    fig = plt.figure(figsize=(LARGURA, 10.0 * CM))
    axA = fig.add_axes([0.0, 0.25, 0.55, 0.66])
    axB = fig.add_axes([0.585, 0.28, 0.415, 0.58])
    for ax in (axA, axB):
        ax.set_aspect('equal')
        ax.axis('off')
    cor_w = {w: (COR['foco'] if w == 17762 else COR['carteira']) for w in LIDERES}
    # ---- Painel A: bipartido 5 × 144
    ordem = ordenar_mercados(meta)
    pos, arcos = anel(ordem, meta.modalidade, raio=1.0, lacuna_grupo=4.0, sub=meta.evento, lacuna_sub=0.9)
    pw = dict(zip(LIDERES, poligono(5, 0.30, 90)))
    seg = [[pw[w], pos[m][:2]] for w in LIDERES for m in inter]
    assert len(seg) == 720
    axA.add_collection(LineCollection(seg, colors=[cor_w[w] for w in LIDERES for _ in inter],
                                      linewidths=0.25, alpha=0.22, zorder=1))
    n_ev = meta[meta.evento != 'indeterminado'].groupby('modalidade').evento.nunique()
    n_m = meta.groupby('modalidade').size()
    n_ind = meta[meta.evento == 'indeterminado'].groupby('modalidade').size()
    rot = {mod: f'{mod} ({n_m[mod]})' for mod in n_m.index}
    desenhar_faixas(axA, arcos, 1.07, 0.10, rotulos=rot, min_graus=14, z=0)
    xy = np.array([pos[m][:2] for m in ordem])
    ind = np.array([meta.loc[m, 'evento'] == 'indeterminado' for m in ordem])
    axA.scatter(xy[~ind, 0], xy[~ind, 1], s=9, marker='s', c=COR['mercado'], edgecolors='#8A5A00', linewidths=0.35, zorder=4)
    axA.scatter(xy[ind, 0], xy[ind, 1], s=9, marker='s', c='white', edgecolors='#8A5A00', linewidths=0.5, zorder=4)
    for w in LIDERES:
        axA.scatter(*pw[w], s=tam_carteira(gw[w]), c=cor_w[w], edgecolors='white', linewidths=0.7, zorder=6)
        x, y = pw[w]
        axA.text(x * 1.75, y * 1.75, str(w), fontsize=8, ha='center', va='center', color=cor_w[w],
                 fontweight='bold' if w == 17762 else 'normal', path_effects=HALO, zorder=7)
    axA.set_xlim(-1.42, 1.42)
    axA.set_ylim(-1.30, 1.30)
    fig.text(0.005, 0.985, 'A', fontsize=10, fontweight='bold', va='top')
    fig.text(0.035, 0.982, 'bipartido: 5 carteiras × 144 mercados comuns\nàs cinco (149 nós, 720 arestas)',
             fontsize=8.5, va='top', linespacing=1.05)
    # ---- Painel B: projeção com pesos do dia inteiro
    pb = dict(zip(LIDERES, poligono(5, 1.0, 90)))
    wmin, wmax = 153, 160

    def larg(v):
        return 0.8 + 2.6 * (v - wmin) / (wmax - wmin)

    for (a, b), v in sorted(pesos.items(), key=lambda kv: kv[1]):
        (x1, y1), (x2, y2) = pb[a], pb[b]
        axB.plot([x1, x2], [y1, y2], color='#4D5D6E', lw=larg(v), alpha=0.85, solid_capstyle='round', zorder=1)
        # diagonais cruzam no centro: rótulo deslocado para não sobrepor
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2  # pontos médios das diagonais são distintos (r≈0,31)
        axB.text(mx, my, str(v), fontsize=8, ha='center', va='center', zorder=3,
                 bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none'))
    for w in LIDERES:
        x, y = pb[w]
        axB.scatter(x, y, s=tam_carteira(gw[w]) * 1.6, c=cor_w[w], edgecolors='white', linewidths=0.8, zorder=4)
        lado = 'center' if abs(x) < 0.3 else ('left' if x > 0 else 'right')
        dx = 0.0 if lado == 'center' else (0.12 if x > 0 else -0.12)
        dy = 0.24 if y > 0.5 else (-0.24 if y < -0.5 else 0.0)
        axB.text(x + dx, y + dy, f'{w}\n({gw[w]} merc.)', fontsize=8,
                 ha=lado, va='center', color=cor_w[w], fontweight='bold' if w == 17762 else 'normal',
                 path_effects=HALO, linespacing=1.0, zorder=5)
    axB.set_xlim(-1.95, 1.95)
    axB.set_ylim(-1.15, 1.35)
    fig.text(0.585, 0.985, 'B', fontsize=10, fontweight='bold', va='top')
    fig.text(0.615, 0.982, 'projeção nas 5 carteiras: peso = mercados em\ncomum entre todas as participações do dia',
             fontsize=8.5, va='top', linespacing=1.05)
    # ---- legenda e nota de universos
    axl = fig.add_axes([0.0, 0.0, 1.0, 0.235])
    axl.axis('off')
    leg = legenda_marcadores(axl, [
        ('carteira 17762 (líder do alerta)', 'o', COR['foco'], 'white', 6.5),
        ('demais líderes', 'o', COR['carteira'], 'white', 6.5),
        ('mercado (evento identificado)', 's', COR['mercado'], '#8A5A00', 4.5),
        ('mercado (evento indeterminado)', 's', 'white', '#8A5A00', 4.5)],
        loc='upper left', bbox_to_anchor=(0.0, 1.0), ncol=4, columnspacing=1.2)
    axl.add_artist(leg)
    from matplotlib.patches import Patch
    mods = sorted(n_m.index, key=ORDEM_MODALIDADE.get)
    axl.legend(handles=[Patch(facecolor=COR_MODALIDADE[m], label=f'{m} ({n_m[m]})') for m in mods],
               loc='center left', bbox_to_anchor=(0.0, 0.50), ncol=5, frameon=False, handlelength=1.1,
               handletextpad=0.35, columnspacing=0.9)
    axl.text(0.0, 0.13, 'Universos diferentes. A: interseção de 144 mercados (nela, todo par teria peso 144).\n'
             'B: todas as participações do dia (pesos 153–160, espessura linear). Ligações = coparticipação, sem direção.', fontsize=8, va='center', color='#333333', linespacing=1.15)
    pasta = RECORTES / 'gephi_nucleo'
    pasta.mkdir(parents=True, exist_ok=True)
    corresp = meta.reset_index()[['condition_node_id', 'question', 'tags', 'modalidade', 'evento', 'regra_evento', 'grau_global']]
    corresp.to_csv(pasta / 'correspondencia_mercado_evento.csv', index=False, encoding='utf-8')
    pd.DataFrame([dict(a=a, b=b, peso=v) for (a, b), v in pesos.items()]).to_csv(pasta / 'projecao_pesos.csv', index=False)
    eventos_mod = {m: dict(mercados=int(n_m[m]), eventos=int(n_ev.get(m, 0)), indeterminados=int(n_ind.get(m, 0)))
                   for m in n_m.index}
    if com_gephi:
        nos = [dict(Id=f'w{w}', Label=str(w), Tipo='carteira', GrauGlobal=int(gw[w]), Modalidade='', Evento='',
                    x=pw[w][0], y=pw[w][1], size=10.0, color=cor_w[w]) for w in LIDERES]
        nos += [dict(Id=f'm{m}', Label='', Tipo='mercado', GrauGlobal=int(meta.loc[m, 'grau_global']),
                     Modalidade=meta.loc[m, 'modalidade'], Evento=meta.loc[m, 'evento'], x=pos[m][0], y=pos[m][1],
                     size=3.0, color=COR['mercado']) for m in ordem]
        ar = [dict(Source=f'w{w}', Target=f'm{m}', Weight=1.0, color=cor_w[w], alpha=0.3) for w in LIDERES for m in inter]
        exportar_gephi('gephi_nucleo__A_bipartido', nos, ar)
        nos = [dict(Id=f'w{w}', Label=str(w), Tipo='carteira', GrauGlobal=int(gw[w]), x=pb[w][0], y=pb[w][1],
                    size=12.0, color=cor_w[w]) for w in LIDERES]
        ar = [dict(Source=f'w{a}', Target=f'w{b}', Weight=float(v), Peso=int(v), color='#4D5D6E', alpha=0.9)
              for (a, b), v in pesos.items()]
        exportar_gephi('gephi_nucleo__B_projecao', nos, ar)
    meta_fig = {
        'titulo': 'Núcleo das cinco líderes — dois painéis', 'figura': 'gephi_nucleo',
        'universo': {'A': '5 carteiras + 144 mercados comuns às cinco = 149 nós, 720 arestas',
                     'B': '5 carteiras, 10 pares; peso = |mercados(a) ∩ mercados(b)| com todas as participações do dia'},
        'seed': 'cobertura total; líderes da seed 42', 'ids': LIDERES,
        'pesos': {f'{a}-{b}': v for (a, b), v in pesos.items()},
        'modalidade_evento': eventos_mod,
        'regra_evento': 'pergunta "A vs B" → evento; pergunta que cita uma equipe de um único evento com as mesmas tags '
                        '→ esse evento; senão indeterminado (não significa ausência de evento). '
                        'Correspondência: recortes/gephi_nucleo/correspondencia_mercado_evento.csv',
        'layout': 'A: anel determinístico (modalidade → evento; lacuna de 4° entre modalidades e 0,9° entre eventos), '
                  'carteiras em pentágono r=0,3; B: pentágono regular. Mesmas coordenadas no projeto Gephi 0.10.1',
        'codificacao': {'cor': 'vermelho=17762; azul=demais líderes; laranja=mercado; vazado=evento indeterminado; '
                               'faixa=modalidade', 'tamanho': 'carteiras ∝ log10(1+grau global)',
                        'espessura': 'B: linear em 153–160', 'rotulos': 'IDs internos e pesos; sem setas'},
    }
    return salvar(fig, 'gephi_nucleo', meta_fig)


# =============================================================================================== Fig. 4
ALVOS_INT = [7896, 36905, 10780, 7219, 9621]
DOSES = [0, 4, 10, 20]


def fig_intervencao(com_gephi=True):
    fonte = RECORTES / 'gephi_intervencao' / 'fonte'
    ar = pd.read_csv(fonte / 'arestas_grupo0_por_dose.csv')
    man = json.loads((fonte / 'manifesto_intervencao.json').read_text(encoding='utf-8'))
    novos = man['mercados_novos_ordem']
    graus_ctx = {int(k): v for k, v in man['graus_contexto'].items()}
    por_dose = {d: set(map(tuple, ar[ar.dose == d][['wallet_id', 'condition_node_id']].to_numpy())) for d in DOSES}
    for d in DOSES:
        g = pd.Series([w for w, _ in por_dose[d]]).value_counts().to_dict()
        assert g == graus_ctx, (d, g)
    universo = sorted(ar.condition_node_id.unique())
    orig = por_dose[0]
    dono = {}
    for m in universo:
        ws = sorted({w for w, mm in orig if mm == m})
        dono[m] = 'novo' if m in novos else (ws[0] if len(ws) == 1 else 'vários')
    assert all(m not in {mm for _, mm in orig} for m in novos)  # mercados novos não eram vizinhos antes
    # dose em que o mercado original perde a última aresta do grupo (0 = nunca)
    def removido_em(m):
        for d in DOSES[1:]:
            if not any(mm == m for _, mm in por_dose[d]):
                return d
        return 0
    rem = {m: removido_em(m) for m in universo if dono[m] != 'novo'}
    ang_w = dict(zip(ALVOS_INT, [126, 198, 270, 342, 54]))
    ordem_grupos = ['novo', 9621, 7219, 10780, 36905, 7896, 'vários']
    ordem = sorted(novos, key=novos.index)
    for gk in ordem_grupos[1:]:
        ms = [m for m in universo if dono[m] == gk]
        ordem += sorted(ms, key=lambda m: (rem[m] == 0, -rem[m] if rem[m] else 0, m))
    grupos = pd.Series({m: str(dono[m]) for m in ordem})
    sub = pd.Series({m: (f'n{4 if novos.index(m) < 4 else 10 if novos.index(m) < 10 else 20}' if dono[m] == 'novo'
                         else f'r{rem[m]}') for m in ordem})
    vao = 360 * 20 / len(ordem)
    pos, arcos = anel(ordem, grupos, raio=1.0, inicio=90 + vao / 2 + 2, lacuna_grupo=4.0, sub=sub, lacuna_sub=1.6)
    pw = {w: (0.36 * math.cos(math.radians(a)), 0.36 * math.sin(math.radians(a))) for w, a in ang_w.items()}

    fig = plt.figure(figsize=(LARGURA, 10.0 * CM))
    H = 0.385
    origens = [(0.0, 0.595), (0.5, 0.595), (0.0, 0.185), (0.5, 0.185)]
    titulos = {0: 'original (antes da intervenção)', 4: '4 novos mercados comuns', 10: '10 novos mercados comuns',
               20: '20 novos mercados comuns'}
    resumo = []
    for (x0, y0), d, letra in zip(origens, DOSES, 'ABCD'):
        ax = fig.add_axes([x0 + 0.004, y0, 0.229, H])
        axq = fig.add_axes([x0 + 0.240, y0, 0.245, H + 0.02])
        ax.set_aspect('equal')
        ax.axis('off')
        axq.axis('off')
        E = por_dose[d]
        ativos_novos = set(novos[:d])
        # arco de destaque dos mercados da intervenção (todas as doses, para referência fixa)
        from matplotlib.patches import Wedge
        a_n = [pos[m][2] for m in novos]
        ax.add_patch(Wedge((0, 0), 1.16, min(a_n) - 2, max(a_n) + 2, width=0.07, facecolor=COR['intervencao'],
                           alpha=0.18, edgecolor='none', zorder=0))
        seg_o = [[pw[w], pos[m][:2]] for w, m in E if m not in novos]
        seg_n = [[pw[w], pos[m][:2]] for w, m in E if m in novos]
        ax.add_collection(LineCollection(seg_o, colors='#5F6B7A', linewidths=0.35, alpha=0.45, zorder=1))
        if seg_n:
            ax.add_collection(LineCollection(seg_n, colors=COR['intervencao'], linewidths=0.55, alpha=0.85, zorder=2))
        lig = {m for _, m in E}
        for m in ordem:
            x, y, _ = pos[m]
            if m in novos:
                on = m in ativos_novos
                ax.scatter(x, y, s=9, marker='s', c=COR['intervencao'] if on else 'white',
                           edgecolors='#8E3B6E' if on else '#D7A6C4', linewidths=0.5, zorder=4)
            else:
                on = m in lig
                ax.scatter(x, y, s=9, marker='s', c=COR['mercado'] if on else 'white',
                           edgecolors='#8A5A00' if on else '#BDBDBD', linewidths=0.4, zorder=4)
        for w in ALVOS_INT:
            ax.scatter(*pw[w], s=tam_carteira(graus_ctx[w]), c=COR['carteira'], edgecolors='white', linewidths=0.6, zorder=6)
            x, y = pw[w]
            ax.text(x * 1.75, y * 1.75, str(w), fontsize=8, ha='center', va='center', color=COR['carteira'],
                    path_effects=HALO, zorder=7)
        ax.set_xlim(-1.22, 1.22)
        ax.set_ylim(-1.22, 1.22)
        # quadro
        comuns5 = sum(1 for m in universo if all((w, m) in E for w in ALVOS_INT))
        n_novos = len(ativos_novos)
        n_ar_novos = len(seg_n)
        trocas = 5 * n_novos
        axq.text(0.0, 1.0, letra, fontsize=10, fontweight='bold', va='top')
        axq.text(0.085, 0.992, titulos[d], fontsize=8.5, va='top')
        linhas = [('mercados novos comuns', f'{n_novos} / 20'),
                  ('arestas p/ mercados novos', f'{n_ar_novos}'),
                  ('trocas', f'{trocas}'),
                  ('mercados comuns aos 5', f'{comuns5}'),
                  ('arestas no recorte', f'{len(E)}')]
        y = 0.87
        for k, v in linhas:
            axq.text(0.0, y, k, fontsize=8, va='top', color='#444444')
            axq.text(1.0, y, v, fontsize=8, va='top', ha='right')
            y -= 0.088
        y -= 0.025
        axq.plot([0, 1], [y + 0.012, y + 0.012], color='#9E9E9E', lw=0.5)
        axq.text(0.0, y, 'carteira', fontsize=8, va='top', color='#444444')
        axq.text(1.0, y, 'grau global (contexto)', fontsize=8, va='top', ha='right', color='#444444')
        y -= 0.088
        for i in range(0, 5, 3):
            parte = ALVOS_INT[i:i + 3]
            axq.text(0.0, y, '   '.join(f'{w}: {graus_ctx[w]}' for w in parte), fontsize=8, va='top', color=COR['carteira'])
            y -= 0.088
        axq.set_xlim(0, 1)
        axq.set_ylim(0, 1)
        resumo.append(dict(painel=letra, dose=d, mercados_novos=n_novos, arestas_novos=n_ar_novos, trocas=trocas,
                           comuns_aos_5=comuns5, arestas_recorte=len(E)))
        if com_gephi:
            nos = [dict(Id=f'w{w}', Label=f'{w} (grau {graus_ctx[w]})', Tipo='carteira', GrauGlobal=int(graus_ctx[w]),
                        Papel='alvo', x=pw[w][0], y=pw[w][1], size=9.0, color=COR['carteira']) for w in ALVOS_INT]
            nos += [dict(Id=f'm{m}', Label='', Tipo='mercado', GrauGlobal=-1,
                         Papel='intervencao' if m in novos else 'original',
                         x=pos[m][0], y=pos[m][1], size=3.0,
                         color=(COR['intervencao'] if m in ativos_novos else '#F3D9E8') if m in novos
                         else (COR['mercado'] if m in lig else '#E0E0E0')) for m in ordem]
            arr = [dict(Source=f'w{w}', Target=f'm{m}', Weight=1.0,
                        color=COR['intervencao'] if m in novos else '#5F6B7A', alpha=0.6) for w, m in sorted(E)]
            exportar_gephi(f'gephi_intervencao__{letra}_dose{d}', nos, arr)
    axl = fig.add_axes([0.0, 0.0, 1.0, 0.17])
    axl.axis('off')
    leg = legenda_marcadores(axl, [
        ('carteira-alvo (grupo 0)', 'o', COR['carteira'], 'white', 6),
        ('mercado novo (cheio = ligado)', 's', COR['intervencao'], '#8E3B6E', 4.5),
        ('mercado original ligado', 's', COR['mercado'], '#8A5A00', 4.5),
        ('mercado original já trocado', 's', 'white', '#BDBDBD', 4.5)],
        loc='upper left', bbox_to_anchor=(0.0, 1.0), ncol=4, columnspacing=1.1)
    axl.add_artist(leg)
    axl.text(0.0, 0.33, 'Sorteio 51011, grupo 0 do teste pareado: cópias de contexto (divisão seed 42) efetivamente usadas no experimento.\n'
             'Coordenadas fixas; universo = união dos vizinhos dos 5 alvos nas 4 condições. Cada troca leva uma aresta a um\n'
             'mercado novo e devolve outra a uma doadora fora do grupo (não desenhada; não é controle): '
             'graus 34, 27, 24, 31 e 42 inalterados.',
             fontsize=8, va='center', color='#333333', linespacing=1.15)
    pasta = RECORTES / 'gephi_intervencao'
    pd.DataFrame([dict(condition_node_id=m, grupo_anel=dono[m], ordem_intervencao=(novos.index(m) + 1) if m in novos else '',
                       perde_ultima_aresta_na_dose=rem.get(m, ''), x=pos[m][0], y=pos[m][1]) for m in ordem]
                 ).to_csv(pasta / 'mercados_universo_visual.csv', index=False)
    meta_fig = {
        'titulo': 'Intervenção artificial — quatro condições do mesmo grupo', 'figura': 'gephi_intervencao',
        'selecao': 'regra fixa: sorteio 51011, grupo de ID 0 do teste PAREADO; alvos 7896, 36905, 10780, 7219, 9621',
        'fonte': 'cópias regeneradas com experimentos.artigo.pareado.preparar_doses (mesma seed, divisão 42 e rótulos); '
                 'ver recortes/gephi_intervencao/fonte/manifesto_intervencao.json',
        'universo': f'{len(universo)} mercados (união das vizinhanças nas 4 condições) + 5 carteiras; '
                    f'{len(orig)} arestas em cada condição',
        'graus': graus_ctx, 'mercados_novos_ordem': novos, 'paineis': resumo,
        'layout': 'anel fixo: arco da intervenção no topo (ordem 1–20, lacunas em 4|10|20); demais mercados agrupados '
                  'pela carteira-alvo dona e pela dose em que são trocados; carteiras em pentágono r=0,36. '
                  'Mesmas coordenadas nos 4 painéis e nos projetos Gephi 0.10.1',
        'omissoes': 'carteiras doadoras e demais vizinhos dos mercados não são desenhados',
        'codificacao': {'cor': 'azul=alvos; magenta=mercados da intervenção (cheio=ligado nesta condição); '
                               'laranja=mercado original ligado; vazado cinza=mercado original já trocado',
                        'tamanho': 'carteiras ∝ log10(1+grau)', 'arestas': 'magenta=para mercado novo; cinza=demais'},
    }
    return salvar(fig, 'gephi_intervencao', meta_fig)


# =============================================================================================== Fig. 1
LIMITE_PDF_MB = 25


def fig_global(com_gephi=True, rasterizar_arestas=None):
    pasta = RECORTES / 'gephi_global'
    nos = pd.read_csv(pasta / 'nos.csv')
    ar = pd.read_csv(pasta / 'arestas.csv', usecols=['Source', 'Target'])
    posf = GEPHI_DIR / 'gephi_global_posicoes.csv'
    if not posf.exists():
        if not com_gephi:
            raise FileNotFoundError('rode antes o ForceAtlas2 no Gephi (fig_global com Gephi)')
        gephi.renderizar(pasta / 'gephi_global.gexf', GEPHI_DIR / 'gephi_global', memoria='10g', **R.FA2_GLOBAL)
    P = pd.read_csv(posf).set_index('Id')
    if com_gephi and not (GEPHI_DIR / 'gephi_global.gephi').exists():
        # projeto nativo com as coordenadas finais do ForceAtlas2 (layout fixo; sem PDF nativo, ver RenderGephi)
        base = nos.set_index('Id')
        g = base.assign(x=P.x.reindex(base.index).values, y=P.y.reindex(base.index).values).reset_index()
        arg = pd.read_csv(pasta / 'arestas.csv')
        escrever_gexf(pasta / 'gephi_global_fa2.gexf', g, arg)
        gephi.renderizar(pasta / 'gephi_global_fa2.gexf', GEPHI_DIR / 'gephi_global', memoria='10g', layout='fixo',
                         pdf='false', edgeOpacity=6, edgeThickness=0.1, labels='true', labelSize=14, border=0.0)
        P = pd.read_csv(posf).set_index('Id')
    assert len(P) == 43732 and len(ar) == 229072
    nos = nos.join(P[['x', 'y']].rename(columns={'x': 'X', 'y': 'Y'}), on='Id')
    assert nos.X.notna().all()
    # normaliza para [-1, 1] mantendo proporção
    cx, cy = nos.X.median(), nos.Y.median()
    esc = np.nanpercentile(np.hypot(nos.X - cx, nos.Y - cy), 99.9)
    nos['u'] = (nos.X - cx) / esc
    nos['v'] = (nos.Y - cy) / esc
    xy = nos.set_index('Id')[['u', 'v']]
    a = xy.loc[ar.Source].to_numpy()
    b = xy.loc[ar.Target].to_numpy()
    segs = np.stack([a, b], axis=1)

    def desenhar(rasterizar):
        fig = plt.figure(figsize=(LARGURA, 13.0 * CM))
        ax = fig.add_axes([0.0, 0.12, 0.70, 0.87])
        ax.set_aspect('equal')
        ax.axis('off')
        lc = LineCollection(segs, colors='#5F6B7A', linewidths=0.08, alpha=0.035, zorder=1)
        lc.set_rasterized(rasterizar)
        ax.add_collection(lc)
        w = nos[nos.Tipo == 'carteira']
        m = nos[nos.Tipo == 'mercado']
        sz = lambda g: 0.05 + 0.9 * np.log10(1 + g) ** 2
        sc2 = ax.scatter(m.u, m.v, s=sz(m.Grau) * 0.6, marker='s', c=COR['mercado'], linewidths=0, alpha=0.45, zorder=2)
        sc1 = ax.scatter(w.u, w.v, s=sz(w.Grau), marker='o', c=COR['carteira'], linewidths=0, alpha=0.6, zorder=3)
        for sc in (sc1, sc2):
            sc.set_rasterized(rasterizar)
        L = nos[nos.Lider].set_index('IdOriginal').loc[LIDERES]
        ax.scatter(L.u, L.v, s=60, facecolors='none', edgecolors=COR['foco'], linewidths=1.0, zorder=5)
        ax.scatter(L.u, L.v, s=6, c=COR['foco'], linewidths=0, zorder=6)
        lim = 1.08
        # rótulos das líderes numa coluna na margem do lado em que estão, com linhas-guia
        cxl, cyl = L.u.mean(), L.v.mean()
        lado = -1 if cxl < 0 else 1
        margem = 1.42
        ax.set_xlim(-margem if lado < 0 else -lim, margem if lado > 0 else lim)
        ax.set_ylim(-lim, lim)
        x_txt = lado * (margem - 0.02)
        ys = np.linspace(cyl + 0.36, cyl - 0.36, 5)
        for (wid, r), yt in zip(L.sort_values('v', ascending=False).iterrows(), ys):
            ax.annotate(f'{wid}' + (' (alerta)' if wid == 17762 else ''), xy=(r.u, r.v), xytext=(x_txt, yt),
                        fontsize=8, color=COR['foco'], ha='left' if lado < 0 else 'right', va='center',
                        fontweight='bold' if wid == 17762 else 'normal',
                        arrowprops=dict(arrowstyle='-', color=COR['foco'], lw=0.5, shrinkA=0, shrinkB=3),
                        path_effects=HALO, zorder=7)
        # inset: zoom na vizinhança das líderes (mesmas coordenadas, sem amostragem)
        axi = fig.add_axes([0.715, 0.50, 0.28, 0.46])
        axi.set_aspect('equal')
        # janela quadrada no aglomerado da líder do alerta (líderes a < 0,15 da 17762); as demais ficam citadas
        ref = L.loc[17762]
        perto = L[np.hypot(L.u - ref.u, L.v - ref.v) < 0.15]
        fora = [int(i) for i in L.index if i not in perto.index]
        cxl = (perto.u.max() + perto.u.min()) / 2
        cyl = (perto.v.max() + perto.v.min()) / 2
        r0 = max(perto.u.max() - perto.u.min(), perto.v.max() - perto.v.min()) / 2 + 0.035
        sel_m = (np.abs(segs[:, :, 0] - cxl) < r0 * 1.6).any(1) & (np.abs(segs[:, :, 1] - cyl) < r0 * 1.6).any(1)
        lci = LineCollection(segs[sel_m], colors='#5F6B7A', linewidths=0.12, alpha=0.045, zorder=1)
        lci.set_rasterized(rasterizar)
        axi.add_collection(lci)
        win = lambda d: d[(np.abs(d.u - cxl) < r0 * 1.3) & (np.abs(d.v - cyl) < r0 * 1.3)]
        wi, mi = win(w), win(m)
        k = min(4.0, 0.12 / r0 * 4)
        s2 = axi.scatter(mi.u, mi.v, s=sz(mi.Grau) * 0.6 * k, marker='s', c=COR['mercado'], linewidths=0, alpha=0.5, zorder=2)
        s1 = axi.scatter(wi.u, wi.v, s=sz(wi.Grau) * k, marker='o', c=COR['carteira'], linewidths=0, alpha=0.7, zorder=3)
        ordem_l = perto.sort_values('v', ascending=False)
        for i, (wid, r) in enumerate(ordem_l.iterrows()):
            axi.annotate(str(wid), (r.u, r.v), xytext=(0.97, 0.88 - i * 0.12), textcoords='axes fraction',
                         fontsize=8, ha='right', va='center', color=COR['foco'],
                         fontweight='bold' if wid == 17762 else 'normal', path_effects=HALO, zorder=7,
                         arrowprops=dict(arrowstyle='-', color=COR['foco'], lw=0.5, shrinkA=0, shrinkB=3))
        if fora:
            axi.text(0.03, 0.04, f"fora da janela: {', '.join(map(str, fora))}", transform=axi.transAxes,
                     fontsize=8, color=COR['foco'], path_effects=HALO, zorder=7)
        for sc in (s1, s2):
            sc.set_rasterized(rasterizar)
        axi.scatter(perto.u, perto.v, s=40, c=COR['foco'], edgecolors='white', linewidths=0.6, zorder=5)
        axi.set_xlim(cxl - r0, cxl + r0)
        axi.set_ylim(cyl - r0, cyl + r0)
        axi.set_xticks([])
        axi.set_yticks([])
        for sp in axi.spines.values():
            sp.set_color('#9E9E9E')
            sp.set_linewidth(0.6)
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle((cxl - r0, cyl - r0), 2 * r0, 2 * r0, fill=False, ec='#9E9E9E', lw=0.6, zorder=6))
        axi.set_title('ampliação (mesmo desenho, sem amostragem)', fontsize=8, pad=3, loc='right')
        # quadro de metadados
        man = json.loads((pasta / 'manifesto.json').read_text(encoding='utf-8'))
        c = man['componentes']
        txt = (f"27.039 carteiras + 16.693 mercados\n= 43.732 vértices; 229.072 arestas\n"
               f"(1 aresta por par carteira–mercado)\n\n"
               f"{c['n_componentes']} componentes; o maior reúne\n{inteiro(c['maior_componente_nos'])} nós "
               f"({num(100 * c['fracao_maior'], 2)}%).\nNenhum componente removido.\n\n"
               f"ForceAtlas2 (Gephi 0.10.1), LinLog,\nscaling 1, gravidade 0,5,\n"
               f"{inteiro(R.FA2_GLOBAL['iter'])} iterações; início seed 42.\n\n"
               f"Área do nó proporcional a\n(log10(1 + grau))².")
        fig.text(0.72, 0.465, txt, fontsize=8, va='top', ha='left', color='#333333', linespacing=1.15)
        axl = fig.add_axes([0.0, 0.0, 1.0, 0.115])
        axl.axis('off')
        legenda_marcadores(axl, [
            ('carteira', 'o', COR['carteira'], COR['carteira'], 5), ('mercado', 's', COR['mercado'], COR['mercado'], 4.5),
            ('cinco líderes', 'o', 'white', COR['foco'], 6.5),
            ('aresta (participação; 3,5% de opacidade)', 'linha', '#5F6B7A', 0.8, 0.6)],
            loc='upper left', bbox_to_anchor=(0.0, 1.0), ncol=4, columnspacing=1.3)
        axl.text(0.0, 0.2, 'Vista de contexto da rede completa (cobertura total). Com 229.072 arestas o núcleo satura visualmente;\n'
                 'a exportação completa (.gephi, GEXF, CSV) acompanha a figura. A proximidade no desenho não indica relação '
                 'entre carteiras.', fontsize=8, va='center', color='#333333', linespacing=1.15)
        return fig

    meta = {
        'titulo': 'Rede bipartida completa do dia', 'figura': 'gephi_global',
        'universo': {'carteiras': 27039, 'mercados': 16693, 'nos': 43732, 'arestas': 229072},
        'cobertura': 'total do dia (sem divisão de treino)', 'filtros': 'nenhum; nenhum componente removido',
        'layout': {'algoritmo': 'ForceAtlas2', 'software': 'Gephi 0.10.1 (bibliotecas do aplicativo instalado)',
                   **{k: v for k, v in R.FA2_GLOBAL.items() if k not in ('edgeOpacity', 'edgeThickness', 'labels', 'border')}},
        'codificacao': {'cor': 'azul=carteira, laranja=mercado, anel vermelho=cinco líderes',
                        'forma': 'círculo=carteira, quadrado=mercado',
                        'tamanho': 'área ∝ (log10(1+grau))²', 'arestas': 'cinza, 0,08 pt, opacidade 3,5%'},
        'arquivo_completo': 'figuras/gephi/gephi_global.gephi e recortes/gephi_global/{gephi_global.gexf,nos.csv,arestas.csv}',
    }
    if rasterizar_arestas is None:
        fig = desenhar(False)
        saidas = salvar(fig, 'gephi_global', {**meta, 'renderizacao': 'totalmente vetorial'})
        if saidas['pdf'].stat().st_size > LIMITE_PDF_MB * 1e6:
            fig = desenhar(True)
            saidas = salvar(fig, 'gephi_global', {**meta, 'renderizacao':
                            f'arestas e nós em camada raster de 600 dpi (PDF vetorial excedia {LIMITE_PDF_MB} MB); '
                            'rótulos, contornos das líderes e legenda vetoriais'})
        return saidas
    fig = desenhar(rasterizar_arestas)
    return salvar(fig, 'gephi_global', {**meta, 'renderizacao': 'raster 600 dpi' if rasterizar_arestas else 'vetorial'})


if __name__ == '__main__':
    alvo = sys.argv[1:] or ['casos', 'nucleo', 'intervencao', 'global']
    sem_gephi = '--sem-gephi' in alvo
    for nome in [a for a in alvo if not a.startswith('--')]:
        print(globals()[f'fig_{nome}'](com_gephi=not sem_gephi))
