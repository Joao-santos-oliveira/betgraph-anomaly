"""Versão de publicação dos dez gráficos do artigo (entrega/graficos_publicacao/).

Mesmos dados de graficos.py, sem recalcular métricas oficiais. Muda a apresentação:
tamanho real de impressão (7,0 pol. de largura de página; 3,5 pol. de coluna), fontes de 7,5 a 8,5 pt,
paleta validada para daltonismo, painéis rotulados (a, b, c) e exportação em PNG 600 dpi, PDF, SVG e HTML.
Também grava validacao.json (conferência de cada figura contra as fontes) e comparacao.html
(antes e depois, com alternância para tons de cinza).

Uso: python -m experimentos.conteudo.graficos_publicacao [--sem-imagens] [--apenas G3 S2]
"""
import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experimentos.conteudo.estilo import ENTREGA, EXTRA, RESULTADOS, ROOT, exportar_imagens, numero, tex
from experimentos.conteudo.graficos import recuperacao_por_ranking

DESTINO = ENTREGA / 'graficos_publicacao'
ORIGINAIS = ENTREGA / 'graficos'

# Tamanho real de impressão: 96 px por polegada (CSS). PNG a 600 dpi = escala 6,25.
PX_POL = 96
PAGINA = int(7.0 * PX_POL)   # 672 px: largura de texto em duas colunas
COLUNA = int(3.5 * PX_POL)   # 336 px
ESCALA_PNG = 600 / PX_POL

FONTE = 'Arial, "Liberation Sans", Helvetica, sans-serif'
TAM = dict(base=10.5, eixo=9.5, titulo=10.5, legenda=9.5, nota=9)  # px: 10,5 px = 7,9 pt impresso
TINTA = '#111827'
TINTA_2 = '#4B5563'
GRADE = '#E5E7EB'
EIXO = '#9CA3AF'

# Paleta validada (validate_palette.js, todos os pares, modo claro): ver AUDITORIA.md.
AZUL = '#2563EB'      # resultado principal (GNN híbrida, alvos, treino)
AMBAR = '#D97706'     # comparador / destaque secundário (relacional, validação, saídas)
ARDOSIA = '#64748B'   # neutro: ablação, controles, execuções individuais
VERDE = '#16A34A'     # melhor execução
VERMELHO = '#B91C1C'  # pior execução (vermelho escuro: separa do verde para deuteranopia, ΔE 11)
PRETO = '#111827'

METODOS = {
    'GNN híbrida': dict(rotulo='GNN híbrida', cor=AZUL, marcador='circle', traco='solid'),
    'Relacional': dict(rotulo='Relacional direto', cor=AMBAR, marcador='square', traco='dash'),
    'Similaridade da GNN': dict(rotulo='Similaridade isolada', cor=ARDOSIA, marcador='diamond-open', traco='dot'),
}
ORDEM = list(METODOS)
SEEDS = (42, 73, 2026)
CORTES = (100, 541)
N_CARTEIRAS = 27039
DOSES = {0: 'Original', 4: '4', 10: '10', 20: '20'}


@dataclass
class Figura:
    nome: str
    legenda: str
    figura: object
    dados: pd.DataFrame
    larga: bool = True
    validacao: dict = field(default_factory=dict)
    html: bool = True


def rgba(hexa, alfa):
    h = hexa.lstrip('#')
    return f'rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alfa})'


def _go():
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    return go, make_subplots


def tema(fig, largura, altura, margem=None, legenda=None):
    """Tema único: fundo branco, grade discreta, eixos finos, tipografia de impressão."""
    fig.update_layout(
        width=largura, height=altura, paper_bgcolor='white', plot_bgcolor='white', separators=',.',
        font=dict(family=FONTE, size=TAM['base'], color=TINTA),
        margin=margem or dict(l=48, r=12, t=28, b=40),
        legend=legenda or dict(orientation='h', x=0, y=1.02, xanchor='left', yanchor='bottom',
                               font=dict(size=TAM['legenda']), bgcolor='rgba(255,255,255,0)', itemwidth=30,
                               tracegroupgap=2),
        hoverlabel=dict(font=dict(family=FONTE, size=11)))
    comum = dict(showline=True, linecolor=EIXO, linewidth=0.8, mirror=False, gridcolor=GRADE, gridwidth=0.6,
                 zeroline=False, ticks='outside', ticklen=3, tickwidth=0.8, tickcolor=EIXO,
                 tickfont=dict(size=TAM['eixo'], color=TINTA_2), title_font=dict(size=TAM['base'], color=TINTA),
                 title_standoff=4)
    fig.update_xaxes(**comum)
    fig.update_yaxes(**comum)
    return fig


def paineis(fig, textos):
    """Títulos de painel alinhados à esquerda, com letra em negrito: 'a  Teste misto'."""
    for anotacao, (letra, texto) in zip(fig.layout.annotations, textos):
        anotacao.update(text=f'<b>{letra}</b>  {texto}', xanchor='left', font=dict(size=TAM['titulo'], color=TINTA))
    # make_subplots centraliza; reposiciona no início do domínio de cada coluna
    dominios = sorted({tuple(fig.layout[k].domain) for k in fig.layout if k.startswith('xaxis')})
    for anotacao in fig.layout.annotations[:len(textos)]:
        for d in dominios:
            if d[0] - 1e-6 <= anotacao.x <= d[1] + 1e-6:
                anotacao.x = d[0]
                break


def titulo_unico(fig, letra, texto, x=0):
    fig.add_annotation(text=f'<b>{letra}</b>  {texto}' if letra else texto, x=x, y=1, xref='paper', yref='paper',
                       xanchor='left', yanchor='bottom', yshift=4, showarrow=False,
                       font=dict(size=TAM['titulo'], color=TINTA))


LOG_TICKS = dict(tickvals=[1, 10, 100, 1000, 10000], ticktext=['1', '10', '100', '1.000', '10.000'])


# G1 ---------------------------------------------------------------------------------------------

def g1():
    go, make_subplots = _go()
    dados = recuperacao_por_ranking()
    t3 = pd.concat([pd.read_csv(RESULTADOS / 'resumo.csv').query("fase=='depois' and papel=='alvo'").assign(desenho='Misto'),
                    pd.read_csv(RESULTADOS / 'pareado/resumo.csv').query("papel=='alvo' and dose>0")
                    .rename(columns={'dose': 'mercados'}).assign(desenho='Pareado')])
    fig = make_subplots(rows=2, cols=2, shared_xaxes=True, shared_yaxes=True, horizontal_spacing=0.04,
                        vertical_spacing=0.13, subplot_titles=[' '] * 4)
    paineis(fig, [('a', 'Teste misto · top 100'), ('b', 'Teste pareado · top 100'),
                  ('c', 'Teste misto · top 541'), ('d', 'Teste pareado · top 541')])
    desloc = dict(zip(ORDEM, (-0.2, 0, 0.2)))
    pos = {4: 0, 10: 1, 20: 2}
    resumo = []
    for linha, corte in enumerate(CORTES, start=1):
        for coluna, desenho in enumerate(('Misto', 'Pareado'), start=1):
            parte = dados[(dados.desenho == desenho) & (dados.corte == corte)]
            for metodo in ORDEM:
                m = METODOS[metodo]
                d = parte[parte.metodo == metodo]
                x = d.mercados.map(pos) + desloc[metodo]
                fig.add_trace(go.Scatter(
                    x=x, y=d.percentual, mode='markers', showlegend=False,
                    marker=dict(color=rgba(m['cor'], 0.45), size=4, symbol='circle',
                                line=dict(width=0)),
                    customdata=d[['unidade', 'alvos']],
                    hovertemplate=f'{m["rotulo"]}<br>%{{customdata[0]}}: %{{customdata[1]}}/50<extra></extra>'),
                    row=linha, col=coluna)
                g = d.groupby('mercados').percentual
                media, mn, mx, n = g.mean(), g.min(), g.max(), g.size()
                fig.add_trace(go.Scatter(
                    x=[pos[k] + desloc[metodo] for k in media.index], y=media.values, mode='lines+markers',
                    name=m['rotulo'], legendgroup=metodo, showlegend=(linha == 1 and coluna == 1),
                    line=dict(color=m['cor'], width=1.4, dash=m['traco']),
                    marker=dict(color='white' if 'open' in m['marcador'] else m['cor'], symbol=m['marcador'].replace('-open', ''),
                                size=7, line=dict(color=m['cor'], width=1.4 if 'open' in m['marcador'] else 0.8)),
                    error_y=dict(type='data', symmetric=False, array=(mx - media).values, arrayminus=(media - mn).values,
                                 color=m['cor'], thickness=0.9, width=2),
                    hovertemplate=f'{m["rotulo"]}: média %{{y:.1f}}%<extra></extra>'), row=linha, col=coluna)
                for k in media.index:
                    resumo.append(dict(desenho=desenho, corte=corte, metodo=metodo, mercados=k, rankings=int(n[k]),
                                       media=media[k], minimo=mn[k], maximo=mx[k]))
    fig.update_xaxes(tickvals=[0, 1, 2], ticktext=['4', '10', '20'], range=[-0.5, 2.5], showgrid=False)
    fig.update_xaxes(title_text='Novos mercados em comum por grupo', row=2)
    fig.update_yaxes(range=[-6, 106], tickvals=[0, 25, 50, 75, 100], ticktext=['0%', '25%', '50%', '75%', '100%'])
    fig.update_yaxes(title_text='Alvos no corte (% de 50)', col=1)
    tema(fig, PAGINA, 430, margem=dict(l=54, r=10, t=44, b=40))
    fig.update_layout(legend=dict(y=1.075))
    resumo = pd.DataFrame(resumo)
    oficial = t3.set_index(['desenho', 'mercados', 'metodo'])
    diff = max(abs(r.media - 100 * oficial.loc[(r.desenho, r.mercados, r.metodo), f'top{r.corte}'] / 50)
               for r in resumo.itertuples())
    validacao = dict(pontos=len(dados), rankings_por_celula=sorted(resumo.rankings.unique().tolist()),
                     maior_diferenca_media_vs_T3_pp=diff, ok=diff < 1e-9)
    return Figura('G1_recuperacao_intensidade',
                  'Recuperação dos alvos artificiais por intensidade. Percentual dos 50 alvos no top 100 (a, b) e no '
                  'top 541 (c, d). Pontos claros: cada ranking (misto: 3 seeds; pareado: 3 sorteios × 3 seeds; o '
                  'relacional é determinístico e tem 1 ranking por sorteio). Marcador cheio: média entre rankings; '
                  'barra: mínimo e máximo observados entre rankings (dispersão empírica, não intervalo de confiança). '
                  'Os métodos são deslocados horizontalmente para não se sobreporem.',
                  fig, pd.concat([dados.assign(tipo='ranking'), resumo.assign(tipo='resumo')], ignore_index=True),
                  validacao=validacao)


# G2 ---------------------------------------------------------------------------------------------

def g2():
    go, _ = _go()
    p = pd.read_csv(RESULTADOS / 'pareado/por_sorteio_seed.csv').query("metodo == 'GNN híbrida'")
    sorteio = p.groupby(['realizacao', 'dose', 'papel']).top100.mean().reset_index()
    resumo = sorteio.groupby(['dose', 'papel']).top100.agg(['mean', 'min', 'max', 'size']).reset_index()
    fig = go.Figure()
    estilos = {'alvo': ('Alvos', AZUL, 'circle', -0.09), 'controle': ('Controles', ARDOSIA, 'square', 0.09)}
    for papel, (nome, cor, simbolo, dx) in estilos.items():
        r = resumo[resumo.papel == papel].sort_values('dose')
        s = sorteio[sorteio.papel == papel]
        x0 = [list(DOSES).index(d) for d in r.dose]
        fig.add_trace(go.Scatter(x=[list(DOSES).index(d) + dx for d in s.dose], y=s.top100, mode='markers',
                                 showlegend=False, marker=dict(color='white', size=5, symbol=simbolo,
                                                               line=dict(color=cor, width=1)),
                                 customdata=s.realizacao,
                                 hovertemplate=f'{nome}, sorteio %{{customdata}}: %{{y:.0f}}<extra></extra>'))
        fig.add_trace(go.Scatter(x=[x + dx for x in x0], y=r['mean'], mode='lines+markers', name=nome,
                                 line=dict(color=cor, width=1.8, dash='solid' if papel == 'alvo' else 'dash'),
                                 marker=dict(color=cor, size=7, symbol=simbolo, line=dict(color='white', width=0.8)),
                                 hovertemplate=f'{nome}: média %{{y:.2f}} de 50<extra></extra>'))
        ultimo = r.iloc[-1]
        fig.add_annotation(x=3 + dx, y=ultimo['mean'], text=f'{nome}<br>{numero(ultimo["mean"], 2)}', showarrow=False,
                           xanchor='left', yanchor='bottom' if papel == 'controle' else 'middle', xshift=8,
                           yshift=4 if papel == 'controle' else 0, align='left',
                           font=dict(size=TAM['nota'], color=TINTA))
    fig.update_xaxes(tickvals=[0, 1, 2, 3], ticktext=list(DOSES.values()), range=[-0.4, 3.75], showgrid=False,
                     title_text='Novos mercados em comum por grupo')
    fig.update_yaxes(range=[-1.5, 51.5], tickvals=[0, 10, 20, 30, 40, 50], title_text='Carteiras no top 100 (de 50)')
    tema(fig, COLUNA, 270, margem=dict(l=44, r=8, t=28, b=38))
    fig.update_layout(showlegend=False)
    oficial = pd.read_csv(RESULTADOS / 'pareado/resumo.csv').query("metodo == 'GNN híbrida'").set_index(['dose', 'papel'])
    diff = float((resumo.set_index(['dose', 'papel'])['mean'] - oficial.top100).abs().max())
    validacao = dict(sorteios_por_ponto=sorted(resumo['size'].unique().tolist()), maior_diferenca_vs_resumo=diff,
                     ok=diff < 1e-9)
    dados = pd.concat([sorteio.assign(tipo='sorteio'), resumo.rename(columns={'mean': 'top100', 'size': 'sorteios'})
                       .assign(tipo='resumo')], ignore_index=True)
    return Figura('G2_alvos_vs_controles',
                  'Alvos e controles no teste pareado (GNN híbrida): carteiras no top 100, de 50 em cada categoria. '
                  'Marcadores vazados: cada um dos 3 sorteios (média das 3 seeds, que coincidem); linha: média dos '
                  'sorteios. Controles têm o mesmo grau inicial do alvo e ligações intactas.',
                  fig, dados, larga=False, validacao=validacao)


# Curvas das ligações reservadas -----------------------------------------------------------------

def _curvas_completas():
    """ROC e precisão–revocação de cada seed, com todos os limiares distintos (sem amostragem)."""
    scores = pd.read_csv(EXTRA / 'scores_teste.csv.gz')
    saida = {}
    for seed, d in scores.groupby('seed'):
        ordem = np.argsort(-d.score.to_numpy(), kind='mergesort')
        y = d.rotulo.to_numpy()[ordem]
        s = d.score.to_numpy()[ordem]
        corte = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
        tp, fp = np.cumsum(y)[corte], np.cumsum(1 - y)[corte]
        P, N = y.sum(), (1 - y).sum()
        tpr, fpr = np.r_[0, tp / P], np.r_[0, fp / N]
        prec = np.r_[1, tp / (tp + fp)]
        saida[int(seed)] = dict(fpr=fpr, tpr=tpr, precisao=prec, positivos=int(P), negativos=int(N),
                                auroc=float(np.trapezoid(tpr, fpr)), ap=float(np.sum(np.diff(tpr) * prec[1:])))
    return saida


def fundo_inset(fig, zoom, kw):
    """Retângulo branco no eixo do quadro ampliado: o subgráfico posterior cobre o principal só ali."""
    import plotly.graph_objects as go
    fig.add_trace(go.Scatter(x=[zoom['x0'], zoom['x1'], zoom['x1'], zoom['x0'], zoom['x0']],
                             y=[zoom['y0'], zoom['y0'], zoom['y1'], zoom['y1'], zoom['y0']], fill='toself',
                             fillcolor='white', line=dict(width=0), mode='lines', hoverinfo='skip', showlegend=False, **kw))


def _afinar(x, y, tolerancia=0.0004):
    """Omite vértices a menos de `tolerancia` (em unidades de taxa) do último mantido: só para desenhar."""
    manter = [0]
    for i in range(1, len(x)):
        if max(abs(x[i] - x[manter[-1]]), abs(y[i] - y[manter[-1]])) >= tolerancia:
            manter.append(i)
    if manter[-1] != len(x) - 1:
        manter.append(len(x) - 1)
    return x[manter], y[manter]


# G3 ---------------------------------------------------------------------------------------------

def _dados_roc():
    """Curvas por seed, média interpolada e faixa ±1 DP, com os AUROC oficiais (avaliacao_cobertura.json)."""
    curvas = _curvas_completas()
    oficial = json.loads((RESULTADOS / 'avaliacao_cobertura.json').read_text())
    auroc = {s: oficial[str(s)]['auc'] for s in SEEDS}
    grade = np.linspace(0, 1, 1001)
    interp = []
    for s in SEEDS:
        c = pd.DataFrame({'fpr': curvas[s]['fpr'], 'tpr': curvas[s]['tpr']}).groupby('fpr').tpr.max()
        interp.append(np.interp(grade, c.index.to_numpy(), c.to_numpy()))
    interp = np.array(interp)
    media, dp = interp.mean(axis=0), interp.std(axis=0, ddof=1)
    media[0], media[-1] = 0.0, 1.0
    valores = np.array(list(auroc.values()))
    return dict(curvas=curvas, auroc=auroc, melhor=max(auroc, key=auroc.get), pior=min(auroc, key=auroc.get),
                grade=grade, media=media, dp=dp, baixo=np.clip(media - dp, 0, 1), cima=np.clip(media + dp, 0, 1),
                auc_media=valores.mean(), auc_dp=valores.std(ddof=1),
                auc_curva_media=float(np.trapezoid(media, grade)))


# Linguagem visual da referência: média azul grossa, execuções em cinza translúcido, faixa lavanda,
# melhor em verde, pior em vermelho, diagonal preta tracejada, moldura fechada e legenda no canto inferior direito.
FAIXA_ROC = 'rgba(37,99,235,0.16)'
MOLDURA = '#111827'


def _figura_roc(d, regiao=None):
    go, _ = _go()
    fig = go.Figure()
    g, curvas, auroc = d['grade'], d['curvas'], d['auroc']
    x0, x1, y0, y1 = regiao or (0, 1, 0, 1)
    fig.add_trace(go.Scatter(x=np.r_[g, g[::-1]], y=np.r_[d['cima'], d['baixo'][::-1]], fill='toself', mode='lines',
                             fillcolor=FAIXA_ROC, line=dict(color='rgba(37,99,235,0.35)', width=0.6),
                             hoverinfo='skip', name='± 1 desvio padrão', legendrank=5))
    for s in SEEDS:
        x, y = _afinar(curvas[s]['fpr'], curvas[s]['tpr'], 0.0004 if regiao is None else 0.00005)
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line_shape='hv', line=dict(color='rgba(100,116,139,0.45)', width=0.9),
                                 hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(color=PRETO, width=1.3, dash='dash'),
                             name='Aleatório (AUROC = 0,500)', hoverinfo='skip', legendrank=1,
                             showlegend=regiao is None))  # no detalhe a diagonal fica fora da região
    fig.add_trace(go.Scatter(x=g, y=d['media'], mode='lines', line=dict(color=AZUL, width=3),
                             name=f'ROC média (AUROC = {numero(d["auc_media"], 4)} ± {numero(d["auc_dp"], 4)})',
                             hovertemplate='FPR %{x:.3f}<br>TPR média %{y:.3f}<extra></extra>', legendrank=2))
    for rank, (s, cor, traco, rotulo) in enumerate(((d['melhor'], VERDE, 'solid', 'Melhor modelo'),
                                                     (d['pior'], VERMELHO, 'dash', 'Pior modelo')), start=3):
        x, y = _afinar(curvas[s]['fpr'], curvas[s]['tpr'], 0.0004 if regiao is None else 0.00005)
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line_shape='hv', line=dict(color=cor, width=1.8, dash=traco),
                                 name=f'{rotulo} (seed {s}) – AUROC = {numero(auroc[s], 4)}', legendrank=rank,
                                 hovertemplate=f'Seed {s}<br>FPR %{{x:.4f}}<br>TPR %{{y:.4f}}<extra></extra>'))
    titulo = ('Curva ROC: previsão de ligações (3 modelos avaliados)' if regiao is None
              else 'Curva ROC: detalhe até 10% de falsos positivos (3 modelos)')
    largura, altura = int(4.6 * PX_POL), int(4.0 * PX_POL)
    tema(fig, largura, altura, margem=dict(l=52, r=14, t=30, b=42),
         legenda=dict(x=0.985, y=0.025, xanchor='right', yanchor='bottom', bgcolor='rgba(255,255,255,0.95)',
                      bordercolor='#D1D5DB', borderwidth=0.8, font=dict(size=TAM['legenda'] - 0.5), tracegroupgap=0,
                      traceorder='normal'))
    fig.update_layout(title=dict(text=titulo, x=0.5, xanchor='center', y=0.985, yanchor='top',
                                 font=dict(size=TAM['titulo'] + 1, color=TINTA)))
    margem_x, margem_y = 0.05 * (x1 - x0), 0.05 * (y1 - y0)
    passo_x = 0.2 if regiao is None else 0.02
    passo_y = 0.2 if regiao is None else 0.1
    fig.update_xaxes(range=[x0 - margem_x, x1 + margem_x], tickvals=np.round(np.arange(x0, x1 + 1e-9, passo_x), 3),
                     tickformat='.1f' if regiao is None else '.2f', title_text='Taxa de falsos positivos',
                     mirror=True, linecolor=MOLDURA, linewidth=1, ticks='outside', tickcolor=MOLDURA)
    fig.update_yaxes(range=[y0 - margem_y, y1 + margem_y], tickvals=np.round(np.arange(y0, y1 + 1e-9, passo_y), 3),
                     tickformat='.1f', title_text='Taxa de verdadeiros positivos',
                     mirror=True, linecolor=MOLDURA, linewidth=1, ticks='outside', tickcolor=MOLDURA)
    return fig


def _validacao_roc(d):
    recalc = {s: d['curvas'][s]['auroc'] for s in SEEDS}
    diferenca = max(abs(recalc[s] - d['auroc'][s]) for s in SEEDS)
    return dict(
        curvas=len(SEEDS), pares_por_seed={s: d['curvas'][s]['positivos'] + d['curvas'][s]['negativos'] for s in SEEDS},
        auroc_oficial=d['auroc'], auroc_recalculado_das_curvas=recalc, maior_diferenca_auroc=diferenca,
        auroc_medio=d['auc_media'], auroc_dp_amostral=d['auc_dp'], auc_da_curva_media_interpolada=d['auc_curva_media'],
        melhor_seed=d['melhor'], pior_seed=d['pior'], diferenca_melhor_pior=d['auroc'][d['melhor']] - d['auroc'][d['pior']],
        banda_em_0_1=bool((d['baixo'] >= 0).all() and (d['cima'] <= 1).all()), dp_maximo_tpr=float(d['dp'].max()),
        interpolacao='TPR máximo por FPR distinto; interpolação linear numa grade de 1.001 FPR (0 a 1); só para média e banda',
        ok=bool(diferenca < 1e-5))


def _dados_csv_roc(d):
    linhas = [pd.DataFrame({'curva': 'media_interpolada', 'fpr': d['grade'], 'tpr': d['media'], 'dp': d['dp'],
                            'banda_inferior': d['baixo'], 'banda_superior': d['cima']})]
    for s in SEEDS:
        linhas.append(pd.DataFrame({'curva': f'seed_{s}', 'fpr': d['curvas'][s]['fpr'], 'tpr': d['curvas'][s]['tpr']}))
    return pd.concat(linhas, ignore_index=True)


def g3():
    d = _dados_roc()
    return Figura('G3_curva_roc',
                  f'Curva ROC da previsão de ligações no teste reservado (22.907 positivos e 22.907 negativos por modelo, '
                  f'grafo de contexto). Cinza: os 3 modelos (seeds 42, 73 e 2026), que compartilham a mesma divisão e os '
                  f'mesmos pares de teste e diferem só na inicialização e no treino; a variação mede, portanto, a '
                  f'sensibilidade à seed. Azul: média vertical das curvas interpoladas numa grade comum de FPR, com faixa '
                  f'de ±1 desvio padrão amostral (n = 3, limitada a [0, 1]); a faixa é quase imperceptível porque o maior '
                  f'DP da TPR entre seeds é {numero(float(d["dp"].max()), 3)}. A legenda traz o AUROC médio entre modelos '
                  f'± DP; a área sob a curva média interpolada é {numero(d["auc_curva_media"], 4)}. Melhor e pior modelo '
                  f'pelo AUROC oficial (diferença de {numero(d["auroc"][d["melhor"]] - d["auroc"][d["pior"]], 4)}). '
                  f'Curvas empíricas em degraus. A região até 10% de falsos positivos está ampliada na figura de '
                  f'detalhe (G3_curva_roc_detalhe, suplemento).', _figura_roc(d), _dados_csv_roc(d), larga=False, validacao=_validacao_roc(d))


def g3_detalhe():
    d = _dados_roc()
    return Figura('G3_curva_roc_detalhe',
                  'Detalhe da curva ROC (G3) na região de até 10% de falsos positivos (TPR de 0,4 a 1,0), com os '
                  'mesmos elementos e escala linear. Os eixos foram recortados para ampliar a região: as curvas '
                  'continuam até FPR = 1.', _figura_roc(d, regiao=(0, 0.10, 0.40, 1.0)), _dados_csv_roc(d), larga=False,
                  validacao=_validacao_roc(d))


# S2 ---------------------------------------------------------------------------------------------

def s2():
    go, _ = _go()
    curvas = _curvas_completas()
    oficial = json.loads((RESULTADOS / 'avaliacao_cobertura.json').read_text())
    ap = {s: oficial[str(s)]['ap'] for s in SEEDS}
    melhor, pior = max(ap, key=ap.get), min(ap, key=ap.get)
    valores = np.array(list(ap.values()))
    fig = go.Figure()
    zoom = dict(x0=0.80, x1=1.0, y0=0.70, y1=1.0)

    def tracar(eixo, legenda):
        kw = dict(xaxis=eixo[0], yaxis=eixo[1])
        if not legenda:
            fundo_inset(fig, zoom, kw)
        fig.add_trace(go.Scatter(x=[0, 1], y=[0.5, 0.5], mode='lines', line=dict(color=PRETO, width=1, dash='dash'),
                                 name='Aleatório (precisão = 0,5)', showlegend=legenda, hoverinfo='skip', **kw))
        for i, s in enumerate(SEEDS):
            x, y = _afinar(curvas[s]['tpr'][1:], curvas[s]['precisao'][1:])
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color=rgba(ARDOSIA, 0.6), width=0.8),
                                     name='Seeds individuais (n = 3)', legendgroup='ind', showlegend=legenda and i == 0,
                                     hoverinfo='skip', **kw))
        for s, cor, traco, rotulo in ((melhor, VERDE, 'solid', 'Maior AP'), (pior, VERMELHO, 'dash', 'Menor AP')):
            x, y = _afinar(curvas[s]['tpr'][1:], curvas[s]['precisao'][1:])
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color=cor, width=1.4, dash=traco),
                                     name=f'{rotulo} (seed {s}) · AP = {numero(ap[s], 4)}', showlegend=legenda,
                                     hovertemplate=f'Seed {s}<br>revocação %{{x:.3f}}<br>precisão %{{y:.3f}}<extra></extra>',
                                     **kw))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='rgba(0,0,0,0)'),
                                 name=f'AP médio = {numero(valores.mean(), 4)} ± {numero(valores.std(ddof=1), 4)} (DP)',
                                 showlegend=legenda, **kw))

    tracar(('x', 'y'), True)
    tracar(('x2', 'y2'), False)
    fig.add_shape(type='rect', xref='x', yref='y', line=dict(color=TINTA_2, width=0.8), **zoom)
    tema(fig, int(4.6 * PX_POL), int(4.25 * PX_POL), margem=dict(l=50, r=10, t=12, b=40),
         legenda=dict(x=0.02, y=0.02, xanchor='left', yanchor='bottom', bgcolor='rgba(255,255,255,0.94)',
                      bordercolor=GRADE, borderwidth=0.8, font=dict(size=TAM['legenda'] - 0.5), tracegroupgap=0))
    fig.update_layout(
        xaxis=dict(range=[-0.02, 1.02], tickvals=np.linspace(0, 1, 6), tickformat='.1f', title_text='Revocação'),
        yaxis=dict(range=[-0.02, 1.02], tickvals=np.linspace(0, 1, 6), tickformat='.1f', title_text='Precisão'),
        # Quadro entre a linha aleatória (0,5) e o platô das curvas, longe da queda final (revocação > 0,8).
        xaxis2=dict(domain=[0.14, 0.62], anchor='y2', range=[zoom['x0'], zoom['x1']], tickvals=[0.8, 0.9, 1],
                    tickformat='.1f'),
        yaxis2=dict(domain=[0.60, 0.88], anchor='x2', range=[zoom['y0'], zoom['y1']], tickvals=[0.7, 0.8, 0.9, 1],
                    tickformat='.1f'))
    fig.update_xaxes(selector=dict(anchor='y2'), linecolor=TINTA_2, mirror=True, tickfont=dict(size=TAM['eixo'] - 1))
    fig.update_yaxes(selector=dict(anchor='x2'), linecolor=TINTA_2, mirror=True, tickfont=dict(size=TAM['eixo'] - 1))
    recalc = {s: curvas[s]['ap'] for s in SEEDS}
    dados = pd.concat([pd.DataFrame({'seed': s, 'revocacao': curvas[s]['tpr'], 'precisao': curvas[s]['precisao']})
                       for s in SEEDS], ignore_index=True)
    validacao = dict(ap_oficial=ap, ap_recalculado=recalc, maior_diferenca=max(abs(recalc[s] - ap[s]) for s in SEEDS),
                     definicao='AP = soma de (R_n − R_{n−1}) · P_n por limiar (sem trapézio)', melhor_seed=melhor,
                     pior_seed=pior, ok=max(abs(recalc[s] - ap[s]) for s in SEEDS) < 1e-4)
    return Figura('S2_precisao_revocacao',
                  'Curva precisão–revocação das ligações reservadas, por seed (mesmos pares de teste nas três). AP é a '
                  'soma por limiar de (R_n − R_{n−1}) · P_n, a mesma definição da T2, não a área trapezoidal. Não se '
                  'desenha uma curva média: a média vertical de precisões não corresponde a nenhum AP. Linha tracejada: '
                  'precisão de um classificador aleatório com negativos 1:1. O quadro mostra a região ampliada.',
                  fig, dados, larga=False, validacao=validacao)


# G4 ---------------------------------------------------------------------------------------------

def g4():
    go, make_subplots = _go()
    traj = pd.read_csv(RESULTADOS / 'pareado/trajetorias.csv').query("papel == 'alvo'")
    gnns = ['GNN híbrida', 'Similaridade da GNN']
    traj = traj[traj.metodo.isin(gnns)]
    real = pd.read_csv(RESULTADOS / 'posicoes_dia_real.csv')
    rel = set(real.wallet_id[real['Relacional_0'] <= 100])
    sobre = pd.DataFrame([dict(metodo=m, seed=s, comuns=len(set(real.wallet_id[real[f'{m}_{s}'] <= 100]) & rel))
                          for m in gnns for s in SEEDS])
    fig = make_subplots(rows=1, cols=2, column_widths=[0.72, 0.28], horizontal_spacing=0.11, subplot_titles=[' '] * 2)
    paineis(fig, [('a', 'Posição dos 50 alvos no teste pareado'), ('b', 'Top 100 comum ao relacional')])
    deslocamento = {'GNN híbrida': -0.19, 'Similaridade da GNN': 0.19}
    for metodo in gnns:
        m = METODOS[metodo]
        d = traj[traj.metodo == metodo]
        x = d.dose.map({k: i for i, k in enumerate(DOSES)}) + deslocamento[metodo]
        fig.add_trace(go.Box(x=x, y=d.posicao, name=m['rotulo'], legendgroup=metodo, boxpoints='all', jitter=0.55,
                             pointpos=0, width=0.3, marker=dict(color=rgba(m['cor'], 0.22), size=2.2),
                             line=dict(color=m['cor'], width=1.1), fillcolor='rgba(255,255,255,0)',
                             offsetgroup=metodo, alignmentgroup='a',
                             hovertemplate=f'{m["rotulo"]}<br>posição %{{y}}<extra></extra>'), row=1, col=1)
        s = sobre[sobre.metodo == metodo]
        fig.add_trace(go.Scatter(x=[f'{v}' for v in s.seed], y=s.comuns, mode='markers+text', name=m['rotulo'],
                                 legendgroup=metodo, showlegend=False, text=s.comuns,
                                 textposition='top center' if metodo == 'GNN híbrida' else 'bottom center',
                                 textfont=dict(size=TAM['nota'], color=TINTA),
                                 marker=dict(color=m['cor'] if 'open' not in m['marcador'] else 'white', size=8,
                                             symbol=m['marcador'].replace('-open', ''),
                                             line=dict(color=m['cor'], width=1.4)),
                                 hovertemplate=f'{m["rotulo"]}, seed %{{x}}: %{{y}} de 100<extra></extra>'), row=1, col=2)
    for corte in CORTES:
        fig.add_hline(y=corte, line=dict(color=TINTA_2, width=0.8, dash='dot'), row=1, col=1)
        fig.add_annotation(x=1, xref='x domain', y=np.log10(corte), yref='y', text=f'top {corte}', showarrow=False,
                           xanchor='right', yanchor='bottom', font=dict(size=TAM['nota'], color=TINTA_2))
    fig.update_layout(boxmode='overlay', boxgap=0.1)
    fig.update_xaxes(tickvals=[0, 1, 2, 3], ticktext=list(DOSES.values()), range=[-0.5, 3.5], showgrid=False,
                     title_text='Novos mercados em comum por grupo', row=1, col=1)
    fig.update_yaxes(type='log', range=[np.log10(N_CARTEIRAS) + 0.08, np.log10(0.8)], **LOG_TICKS,
                     title_text='Posição no ranking (log; 1 no alto)', row=1, col=1)
    fig.update_xaxes(type='category', title_text='Seed', row=1, col=2)
    fig.update_yaxes(range=[-16, 118], tickvals=[0, 25, 50, 75, 100], title_text='Carteiras em comum (de 100)', row=1, col=2)
    tema(fig, PAGINA, 330, margem=dict(l=54, r=14, t=44, b=40))
    fig.update_layout(legend=dict(y=1.11))
    caixas = traj.groupby(['metodo', 'dose']).posicao.describe().reset_index()
    validacao = dict(pontos_por_caixa=sorted(traj.groupby(['metodo', 'dose']).size().unique().tolist()),
                     pontos_omitidos=0, sobreposicao=sobre.to_dict('records'),
                     ok=bool(traj.groupby(['metodo', 'dose']).size().eq(450).all()))
    return Figura('G4_comparacao_gnns',
                  'GNN híbrida versus similaridade isolada: mesmos embeddings e mesmo treino (mesma AUROC); só muda o '
                  'score de carteira. (a) Todas as posições dos 50 alvos em cada ranking do teste pareado (3 sorteios '
                  '× 3 seeds = 450 por caixa); caixa: quartis e mediana; bigodes: 1,5 × IQR; todos os pontos são '
                  'mostrados. (b) Carteiras do top 100 de cada GNN presentes também no top 100 do relacional no dia real.',
                  fig, pd.concat([caixas.assign(painel='a'), sobre.assign(painel='b')], ignore_index=True),
                  validacao=validacao)


# G5 ---------------------------------------------------------------------------------------------

def g5():
    go, make_subplots = _go()
    traj = pd.read_csv(RESULTADOS / 'pareado/trajetorias.csv').query("metodo == 'GNN híbrida' and seed == 42")
    fig = make_subplots(rows=1, cols=3, column_widths=[0.37, 0.37, 0.26], horizontal_spacing=0.06,
                        subplot_titles=[' '] * 3, shared_yaxes=False)
    paineis(fig, [('a', 'Alvos'), ('b', 'Controles'), ('c', 'Deslocamento pareado')])
    cores = {'alvo': AZUL, 'controle': ARDOSIA}
    idx = {k: i for i, k in enumerate(DOSES)}
    for coluna, papel in enumerate(('alvo', 'controle'), start=1):
        d = traj[traj.papel == papel]
        cor = cores[papel]
        xs, ys = [], []
        for _, w in d.sort_values('dose').groupby(['realizacao', 'wallet_id']):
            xs += list(w.dose.map(idx)) + [None]
            ys += list(w.posicao) + [None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', line=dict(color=rgba(cor, 0.10), width=0.6),
                                 hoverinfo='skip', showlegend=False), row=1, col=coluna)
        q = d.groupby('dose').posicao.quantile([0.25, 0.5, 0.75]).unstack()
        x = [idx[k] for k in q.index]
        fig.add_trace(go.Scatter(x=x + x[::-1], y=list(q[0.75]) + list(q[0.25])[::-1], fill='toself', mode='lines',
                                 fillcolor=rgba(cor, 0.22), line=dict(width=0), hoverinfo='skip',
                                 name='Intervalo interquartil (25–75%)', legendgroup='iqr',
                                 showlegend=coluna == 1), row=1, col=coluna)
        fig.add_trace(go.Scatter(x=x, y=q[0.5], mode='lines+markers', line=dict(color=cor, width=2.2),
                                 marker=dict(color=cor, size=6, line=dict(color='white', width=0.8)),
                                 name='Mediana', legendgroup='med', showlegend=coluna == 1,
                                 hovertemplate='Mediana: posição %{y:.0f}<extra></extra>'), row=1, col=coluna)
        fig.add_hline(y=100, line=dict(color=TINTA_2, width=0.8, dash='dot'), row=1, col=coluna)
    fig.add_annotation(x=0, xref='x domain', y=np.log10(100), yref='y', text='top 100', showarrow=False,
                       xanchor='left', yanchor='bottom', font=dict(size=TAM['nota'], color=TINTA_2))
    # (c) razão de posições original/20, por carteira: >1 = subiu
    largo = traj.pivot_table(index=['realizacao', 'wallet_id', 'papel'], columns='dose', values='posicao').reset_index()
    largo['log_razao'] = np.log10(largo[0] / largo[20])
    for papel, nome in (('alvo', 'Alvos'), ('controle', 'Controles')):
        d = largo[largo.papel == papel]
        fig.add_trace(go.Box(x=d.log_razao, y=[nome] * len(d), orientation='h', boxpoints='all', jitter=0.6,
                             pointpos=0, marker=dict(color=rgba(cores[papel], 0.35), size=2.6),
                             line=dict(color=cores[papel], width=1.1), fillcolor='rgba(255,255,255,0)', showlegend=False,
                             hovertemplate=f'{nome}: log10(original/20) = %{{x:.2f}}<extra></extra>'), row=1, col=3)
    fig.add_vline(x=0, line=dict(color=TINTA_2, width=0.8), row=1, col=3)
    log = dict(type='log', range=[np.log10(N_CARTEIRAS) + 0.08, np.log10(0.8)], **LOG_TICKS)
    fig.update_yaxes(**log, row=1, col=1)
    fig.update_yaxes(**log, row=1, col=2, showticklabels=False)
    fig.update_yaxes(title_text='Posição no ranking (log; 1 no alto)', row=1, col=1)
    for c in (1, 2):
        fig.update_xaxes(tickvals=[0, 1, 2, 3], ticktext=list(DOSES.values()), range=[-0.25, 3.25], showgrid=False,
                         title_text='Novos mercados', row=1, col=c)
    fig.update_xaxes(title_text='log<sub>10</sub>(posição original / com 20)', tickvals=[-1, 0, 1, 2], row=1, col=3)
    fig.update_yaxes(row=1, col=3, showgrid=False)
    tema(fig, PAGINA, 300, margem=dict(l=54, r=10, t=44, b=40))
    fig.update_layout(legend=dict(y=1.12))
    validacao = dict(carteiras={p: int((largo.papel == p).sum()) for p in ('alvo', 'controle')},
                     pares_completos=bool(largo[[0, 4, 10, 20]].notna().all().all()),
                     mediana_log_razao={p: float(largo[largo.papel == p].log_razao.median()) for p in ('alvo', 'controle')},
                     ok=bool(largo[[0, 4, 10, 20]].notna().all().all()))
    return Figura('G5_trajetorias_pareadas',
                  'Trajetórias pareadas no teste pareado (GNN híbrida, seed 42, 3 sorteios: 150 alvos e 150 controles). '
                  '(a, b) Linhas finas: cada carteira; faixa: intervalo interquartil entre carteiras; linha forte: '
                  'mediana. (c) Deslocamento de cada carteira entre o grafo original e 20 mercados, em log10 da razão '
                  'das posições (positivo = subiu); caixa: quartis, bigodes 1,5 × IQR, todos os pontos mostrados.',
                  fig, traj, validacao=validacao)


# G6 ---------------------------------------------------------------------------------------------

def g6():
    go, make_subplots = _go()
    cob = pd.read_csv(RESULTADOS / 'cobertura_total.csv')
    s = 42
    d = cob.assign(contexto=cob[f'contexto_{s}'], total=cob[f'total_{s}'])
    d = d[(d.contexto <= 100) | (d.total <= 100)].copy()
    d['grupo'] = np.select([(d.contexto <= 100) & (d.total <= 100), d.total <= 100], ['permaneceu', 'entrou'], 'saiu')
    contagem = d.grupo.value_counts().to_dict()
    casos = pd.read_csv(EXTRA / 'casos_dia_real.csv').query("papel == 'caso'")
    lider = int(d.loc[d.total.idxmin(), 'wallet_id'])
    lider_ctx = int(d.loc[d.contexto.idxmin(), 'wallet_id'])
    fig = make_subplots(rows=1, cols=2, column_widths=[0.36, 0.64], horizontal_spacing=0.16, subplot_titles=[' '] * 2)
    paineis(fig, [('a', 'Composição do top 100'), ('b', 'Carteiras que entraram ou saíram, e os casos da T6')])
    barras = [('Contexto', [('permaneceu', contagem['permaneceu']), ('saiu', contagem['saiu'])]),
              ('Cobertura total', [('permaneceu', contagem['permaneceu']), ('entrou', contagem['entrou'])])]
    estilo = {'permaneceu': ('Permaneceram', '#CBD5E1'), 'saiu': ('Saíram', AMBAR), 'entrou': ('Entraram', AZUL)}
    vistos = set()
    for rotulo, partes in barras:
        for grupo, n in partes:
            nome, cor = estilo[grupo]
            fig.add_trace(go.Bar(y=[rotulo], x=[n], orientation='h', name=nome, marker=dict(color=cor, line=dict(color='white', width=1)),
                                 text=[n], textposition='inside', insidetextanchor='middle',
                                 textfont=dict(size=TAM['nota'], color=TINTA if grupo == 'permaneceu' else 'white'),
                                 legendgroup=grupo, showlegend=grupo not in vistos, width=0.55,
                                 hovertemplate=f'{nome}: %{{x}}<extra></extra>'), row=1, col=1)
            vistos.add(grupo)
    fig.update_layout(barmode='stack')
    mov = d[d.grupo != 'permaneceu'].copy()
    destaque = set(casos.wallet_id) | {lider, lider_ctx}
    mov = pd.concat([mov, d[d.wallet_id.isin(destaque) & ~d.wallet_id.isin(mov.wallet_id)]])
    mov['rotulo'] = mov.apply(lambda r: f'{int(r.wallet_id)}' + (' (T6)' if r.wallet_id in set(casos.wallet_id) else '')
                              + (' líder' if r.wallet_id == lider else '') + (' líder ctx.' if r.wallet_id == lider_ctx else ''),
                              axis=1)
    mov = mov.sort_values(['grupo', 'total'], key=lambda c: c.map({'permaneceu': 0, 'entrou': 1, 'saiu': 2}) if c.name == 'grupo' else c)
    for _, r in mov.iterrows():
        nome, cor = estilo[r.grupo]
        cor = TINTA_2 if r.grupo == 'permaneceu' else cor
        fig.add_trace(go.Scatter(x=[r.contexto, r.total], y=[r.rotulo, r.rotulo], mode='lines',
                                 line=dict(color=cor, width=1.4), showlegend=False, hoverinfo='skip'), row=1, col=2)
        fig.add_trace(go.Scatter(x=[r.contexto], y=[r.rotulo], mode='markers', showlegend=False,
                                 marker=dict(color='white', size=6, line=dict(color=cor, width=1.3)),
                                 hovertemplate=f'{int(r.wallet_id)}: contexto %{{x}}<extra></extra>'), row=1, col=2)
        fig.add_trace(go.Scatter(x=[r.total], y=[r.rotulo], mode='markers', showlegend=False,
                                 marker=dict(color=cor, size=6, line=dict(color='white', width=0.6)),
                                 hovertemplate=f'{int(r.wallet_id)}: cobertura total %{{x}}<extra></extra>'), row=1, col=2)
    fig.add_vline(x=100, line=dict(color=TINTA_2, width=0.8, dash='dot'), row=1, col=2)
    fig.add_annotation(x=np.log10(100), xref='x2', y=1, yref='y2 domain', text='top 100', showarrow=False,
                       xanchor='left', yanchor='bottom', xshift=2, font=dict(size=TAM['nota'], color=TINTA_2))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name='posição no contexto',
                             marker=dict(color='white', size=6, line=dict(color=TINTA_2, width=1.3))), row=1, col=2)
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name='posição na cobertura total',
                             marker=dict(color=TINTA_2, size=6)), row=1, col=2)
    fig.update_xaxes(range=[0, 100], tickvals=[0, 25, 50, 75, 100], title_text='Carteiras', row=1, col=1)
    fig.update_yaxes(autorange='reversed', row=1, col=1, showgrid=False)
    fig.update_xaxes(type='log', range=[np.log10(0.8), np.log10(mov[['contexto', 'total']].max().max() * 1.3)],
                     tickvals=[1, 3, 10, 30, 100, 300], title_text='Posição no ranking (log)', row=1, col=2)
    fig.update_yaxes(autorange='reversed', row=1, col=2, tickfont=dict(size=TAM['eixo'] - 0.5))
    tema(fig, PAGINA, 320, margem=dict(l=86, r=10, t=44, b=40))
    fig.update_layout(legend=dict(y=1.12))
    oficial = json.loads((RESULTADOS / 'avaliacao_cobertura.json').read_text())[str(s)]
    validacao = dict(contagens=contagem, permaneceram_oficial=oficial['top100_comuns'],
                     lider_contexto=lider_ctx, lider_total=lider,
                     ok=contagem['permaneceu'] == oficial['top100_comuns'] and lider == oficial['primeiro_total']
                     and lider_ctx == oficial['primeiro_contexto'])
    return Figura('G6_entradas_saidas_top100',
                  'Top 100 do dia real com as ligações de contexto e com a cobertura total (GNN híbrida, seed 42). '
                  f'(a) {contagem["permaneceu"]} carteiras permaneceram; {contagem["saiu"]} saíram e {contagem["entrou"]} '
                  'entraram. (b) Posição no contexto (vazado) e na cobertura total (cheio) das carteiras que entraram '
                  'ou saíram, dos casos da T6 e dos líderes. Não há fluxo entre carteiras: cada linha é a mesma carteira.',
                  fig, d[['wallet_id', 'contexto', 'total', 'grupo']].sort_values('total'), validacao=validacao)


# S1 ---------------------------------------------------------------------------------------------

def s1():
    go, make_subplots = _go()
    traj = pd.read_csv(RESULTADOS / 'pareado/trajetorias.csv').query("papel == 'alvo' and dose > 0")
    k = np.arange(1, N_CARTEIRAS + 1)
    linhas = []
    for (dose, metodo), d in traj.groupby(['dose', 'metodo']):
        por_ranking = []
        for _, r in d.groupby(['realizacao', 'seed']):
            pos = np.sort(r.posicao.to_numpy())
            por_ranking.append(100 * np.searchsorted(pos, k, side='right') / len(pos))
        media = np.mean(por_ranking, axis=0)
        mudou = np.r_[True, np.diff(media) != 0, True][:-1] | (k == N_CARTEIRAS)
        linhas.append(pd.DataFrame({'dose': dose, 'metodo': metodo, 'k': k[mudou], 'percentual': media[mudou],
                                    'rankings': len(por_ranking)}))
    curvas = pd.concat(linhas, ignore_index=True)
    fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.025, subplot_titles=[' '] * 3)
    paineis(fig, [('a', '4 novos mercados'), ('b', '10 novos mercados'), ('c', '20 novos mercados')])
    for coluna, dose in enumerate((4, 10, 20), start=1):
        for metodo in ORDEM:
            m = METODOS[metodo]
            d = curvas[(curvas.dose == dose) & (curvas.metodo == metodo)]
            fig.add_trace(go.Scatter(x=d.k, y=d.percentual, mode='lines', line_shape='hv', name=m['rotulo'],
                                     legendgroup=metodo, showlegend=coluna == 1,
                                     line=dict(color=m['cor'], width=1.6 if metodo != 'Relacional' else 1.3,
                                               dash=m['traco']),
                                     hovertemplate=f'{m["rotulo"]}: %{{y:.1f}}% até a posição %{{x}}<extra></extra>'),
                          row=1, col=coluna)
        for corte in CORTES:
            fig.add_vline(x=corte, line=dict(color=TINTA_2, width=0.7, dash='dot'), row=1, col=coluna)
        if coluna == 1:  # só no painel a, onde o canto superior está livre; a legenda identifica as linhas
            for corte in CORTES:
                fig.add_annotation(x=np.log10(corte), xref='x', y=0.98, yref='y domain', text=str(corte),
                                   showarrow=False, xanchor='right', yanchor='top', xshift=-2,
                                   font=dict(size=TAM['nota'] - 0.5, color=TINTA_2))
    fig.update_xaxes(type='log', range=[0, np.log10(N_CARTEIRAS)], **LOG_TICKS, title_text='Profundidade k (log)')
    fig.update_yaxes(range=[0, 104], tickvals=[0, 25, 50, 75, 100], ticksuffix='%')
    fig.update_yaxes(title_text='Alvos até a posição k', col=1)
    tema(fig, PAGINA, 250, margem=dict(l=54, r=10, t=44, b=40))
    fig.update_layout(legend=dict(y=1.14))
    final = curvas.groupby(['dose', 'metodo']).percentual.last()
    monotona = all((np.diff(g.percentual) >= -1e-12).all() for _, g in curvas.groupby(['dose', 'metodo']))
    validacao = dict(rankings_por_curva=sorted(curvas.rankings.unique().tolist()), monotona=monotona,
                     termina_em_100=bool(np.allclose(final, 100)), ok=monotona and bool(np.allclose(final, 100)))
    return Figura('S1_recuperacao_acumulada',
                  'Recuperação acumulada pela profundidade do ranking no teste pareado: percentual dos 50 alvos até a '
                  'posição k, média dos rankings (3 sorteios × 3 seeds; 3 rankings no relacional), em degraus exatos. '
                  'Linhas pontilhadas: cortes 100 e 541. O relacional (tracejado) quase coincide com a GNN híbrida.',
                  fig, curvas, validacao=validacao)


# S3 ---------------------------------------------------------------------------------------------

def s3():
    go, make_subplots = _go()
    hist = pd.read_csv(EXTRA / 'historico_treino.csv')
    treino = pd.read_csv(EXTRA / 'treino_por_seed.csv')
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1, subplot_titles=[' '] * 2)
    paineis(fig, [('a', 'Perda de ligações por época'), ('b', 'AUROC de validação por época')])
    for i, seed in enumerate(SEEDS):
        d = hist[hist.seed == seed]
        for coluna, (campo, nome, cor) in enumerate((('link_loss', 'Treino', AZUL), ('val_loss', 'Validação', AMBAR))):
            fig.add_trace(go.Scatter(x=d.epoch, y=d[campo], mode='lines', name=nome, legendgroup=campo,
                                     showlegend=i == 0, line=dict(color=cor, width=1.1, dash='solid' if campo == 'link_loss' else 'dash'),
                                     hovertemplate=f'{nome}, seed {seed}<br>época %{{x}}: %{{y:.4f}}<extra></extra>'),
                          row=1, col=1)
        fig.add_trace(go.Scatter(x=d.epoch, y=d.val_auc, mode='lines', showlegend=False,
                                 line=dict(color=AMBAR, width=1.1, dash='dash'),
                                 hovertemplate=f'Seed {seed}<br>época %{{x}}: %{{y:.4f}}<extra></extra>'), row=1, col=2)
    melhor = treino.melhor_epoca.unique()
    for c in (1, 2):
        fig.add_vline(x=100, line=dict(color=TINTA_2, width=0.8, dash='dot'), row=1, col=c)
    fig.add_annotation(x=99, xref='x2', y=0.70, yref='y2', text='melhor época = 100 (teto)<br>nas 3 seeds; paciência 15<br>não atingida',
                       showarrow=False, xanchor='right', align='right', font=dict(size=TAM['nota'] - 0.5, color=TINTA_2))
    fig.update_xaxes(range=[0, 102], tickvals=[1, 20, 40, 60, 80, 100], title_text='Época')
    fig.update_yaxes(title_text='Entropia cruzada (ligações)', row=1, col=1)
    fig.update_yaxes(title_text='AUROC', range=[0.5, 1.0], tickformat='.1f', row=1, col=2)
    tema(fig, PAGINA, 250, margem=dict(l=54, r=10, t=44, b=40))
    fig.update_layout(legend=dict(y=1.14))
    validacao = dict(epocas_por_seed=hist.groupby('seed').epoch.max().to_dict(), melhor_epoca=melhor.tolist(),
                     linhas=len(hist), suavizacao='nenhuma', ok=len(hist) == 300)
    return Figura('S3_perdas_treino',
                  'Treino da GNN híbrida, uma linha por seed (3), sem suavização. (a) Perda de ligações no treino '
                  '(contínua; sem o termo de contraste de peso 0,1) e na validação (tracejada). (b) AUROC de validação. '
                  'A melhor época foi a última (100) nas três seeds: a parada antecipada (paciência 15) não foi '
                  'atingida, e a perda de validação ainda caía.', fig, hist, validacao=validacao)


# S4 ---------------------------------------------------------------------------------------------

def s4():
    go, make_subplots = _go()
    resumo = pd.read_csv(ROOT / 'benchmark/resumo.csv')
    etapas = pd.read_csv(ROOT / 'benchmark/etapas.csv')
    hardware = json.loads((ROOT / 'benchmark/hardware.json').read_text(encoding='utf-8'))
    resumo = resumo[resumo.etapa != 'total']
    total = etapas[etapas.etapa == 'total'].segundos
    grupo = lambda e: 'Treino da GNN' if e.startswith('treino') else (
        'Scores de carteira' if e.split('_')[0] in ('relacional', 'hibrido', 'similaridade') else 'Dados e avaliação')
    cores = {'Dados e avaliação': ARDOSIA, 'Treino da GNN': AZUL, 'Scores de carteira': AMBAR}
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.04, subplot_titles=[' '] * 2)
    paineis(fig, [('a', 'Tempo de parede (s)'), ('b', 'Pico de memória residente (MiB)')])
    vistos = set()
    for coluna, (med, campo) in enumerate((('segundos_mediana', 'segundos'), ('rss_mib_mediana', 'rss_mib')), start=1):
        for _, r in resumo.iterrows():
            g = grupo(r.etapa)
            fig.add_trace(go.Bar(y=[r.rotulo], x=[r[med]], orientation='h', marker=dict(color=rgba(cores[g], 0.85)),
                                 width=0.6, name=g, legendgroup=g, showlegend=coluna == 1 and g not in vistos,
                                 hovertemplate=f'{r.rotulo}: mediana %{{x:.1f}}<extra></extra>'), row=1, col=coluna)
            vistos.add(g)
            reps = etapas[etapas.etapa == r.etapa][campo]
            fig.add_trace(go.Scatter(y=[r.rotulo] * len(reps), x=reps, mode='markers', showlegend=False,
                                     marker=dict(color=TINTA, size=3.5, symbol='line-ns-open', line=dict(width=1.2)),
                                     hovertemplate='repetição: %{x:.1f}<extra></extra>'), row=1, col=coluna)
            fig.add_annotation(x=max(reps.max(), r[med]), y=r.rotulo, xref=f'x{"" if coluna == 1 else 2}',
                               yref='y', text=numero(r[med], 1 if coluna == 1 else 0), showarrow=False,
                               xanchor='left', xshift=4, font=dict(size=TAM['nota'] - 0.5, color=TINTA))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name='cada repetição (n = 3)',
                             marker=dict(color=TINTA, size=6, symbol='line-ns-open', line=dict(width=1.2))), row=1, col=1)
    fig.update_yaxes(autorange='reversed', showgrid=False, tickfont=dict(size=TAM['eixo'] - 0.5))
    fig.update_xaxes(rangemode='tozero')
    fig.update_xaxes(range=[0, etapas.segundos[etapas.etapa != 'total'].max() * 1.18], row=1, col=1)
    fig.update_xaxes(range=[0, etapas.rss_mib.max() * 1.15], row=1, col=2)
    tema(fig, PAGINA, 330, margem=dict(l=170, r=14, t=48, b=30))
    fig.update_layout(legend=dict(y=1.11))
    cpu = hardware.get('cpu', {}).get('Model name', '')
    validacao = dict(etapas=len(resumo), repeticoes=int(resumo.repeticoes.min()),
                     mediana_confere=bool(np.allclose(etapas[etapas.etapa != 'total'].groupby('etapa').segundos.median()
                                                      .reindex(resumo.etapa), resumo.segundos_mediana)),
                     pipeline_total_mediana_s=float(total.median()), ok=True)
    validacao['ok'] = validacao['mediana_confere']
    return Figura('S4_tempos_memoria',
                  f'Custo por etapa do pipeline do dia real: barra = mediana de {int(resumo.repeticoes.min())} repetições; '
                  f'traços = cada repetição (a primeira inclui cache frio). Pipeline completo: mediana de '
                  f'{numero(total.median() / 60, 1)} min. Máquina: {cpu}, {hardware.get("nucleos_logicos")} núcleos '
                  f'lógicos, {numero(hardware.get("memoria_total_mib", 0) / 1024, 1)} GiB, 2 threads por processo.',
                  fig, resumo, validacao=validacao)


# Saídas -----------------------------------------------------------------------------------------

GERADORES = (g1, g2, g3, g3_detalhe, g4, g5, g6, s1, s2, s3, s4)


def rotulo(nome):
    """Rótulo LaTeX único: fig:g3, fig:g3-detalhe, fig:s1..."""
    return nome.split('_')[0].lower() + ('-detalhe' if nome.endswith('_detalhe') else '')


def escrever_tex(figuras, caminho):
    linhas = ['% Versão de publicação. Gerado por experimentos/conteudo/graficos_publicacao.py.',
              '% Requer \\usepackage{graphicx}. Figuras já estão no tamanho de impressão (7,0 ou 3,5/4,6 pol.).',
              '% Ajuste o caminho com \\graphicspath{{entrega/graficos_publicacao/}}.', '']
    nativas = {'G3_curva_roc', 'G3_curva_roc_detalhe', 'S2_precisao_revocacao'}  # desenhadas a 4,6 pol.: sem reduzir a fonte
    for f in figuras:
        if f.nome in nativas:
            ambiente, largura = 'figure*', '4.6in'
        else:
            ambiente = 'figure*' if f.larga else 'figure'
            largura = '\\textwidth' if f.larga else '\\columnwidth'
        linhas += [f'\\begin{{{ambiente}}}[htbp]', '\\centering', f'\\includegraphics[width={largura}]{{{f.nome}.pdf}}',
                   f'\\caption{{{tex(f.legenda)}}}', f'\\label{{fig:{rotulo(f.nome)}}}',
                   f'\\end{{{ambiente}}}', '']
    linhas += ['% Alternativa: ROC e precisão-revocação lado a lado. Cada uma fica com 75% do tamanho desenhado',
               '% (texto de cerca de 6 pt); prefira as figuras separadas acima se o periódico exigir 7 pt.',
               '% \\begin{figure*}[htbp]\\centering',
               '% \\includegraphics[width=0.49\\textwidth]{G3_curva_roc.pdf}\\hfill',
               '% \\includegraphics[width=0.49\\textwidth]{S2_precisao_revocacao.pdf}',
               '% \\caption{(a) Curva ROC e (b) curva precisão-revocação das ligações reservadas.}',
               '% \\label{fig:roc-pr}', '% \\end{figure*}', '']
    caminho.write_text('\n'.join(linhas), encoding='utf-8', newline='\n')


def escrever_comparacao(figuras, caminho):
    def antes(nome):
        if (ORIGINAIS / f'{nome}.png').exists():
            return (f'<figure><figcaption>Antes (entrega/graficos)</figcaption>'
                    f'<img src="../graficos/{nome}.png" alt="{nome}, versão anterior"></figure>')
        return '<figure><figcaption>Antes</figcaption><p>Figura nova, sem versão anterior.</p></figure>'
    blocos = []
    for f in figuras:
        blocos.append(f'''<section><h2>{f.nome}</h2><div class="par">
{antes(f.nome)}
<figure><figcaption>Depois (entrega/graficos_publicacao)</figcaption><img class="novo" src="{f.nome}.png" alt="{f.nome}, versão de publicação"></figure>
</div><p class="legenda">{f.legenda}</p></section>''')
    html = f'''<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gráficos: antes e depois</title><style>
body{{font-family:Arial,Helvetica,sans-serif;margin:0 auto;max-width:1400px;padding:16px;background:#fff;color:#111827}}
h1{{font-size:20px}} h2{{font-size:16px;margin:28px 0 8px;border-top:1px solid #e5e7eb;padding-top:16px}}
.par{{display:grid;grid-template-columns:1fr 1fr;gap:16px}} figure{{margin:0}} img{{width:100%;border:1px solid #e5e7eb}}
figcaption{{font-size:12px;color:#4b5563;margin-bottom:4px}} .legenda{{font-size:13px;color:#4b5563;max-width:1000px}}
body.cinza img{{filter:grayscale(1)}} label{{font-size:14px}}
@media (max-width:800px){{.par{{grid-template-columns:1fr}}}}
</style></head><body><h1>Gráficos do artigo: versão anterior × versão de publicação</h1>
<label><input type="checkbox" onchange="document.body.classList.toggle('cinza',this.checked)"> Ver em tons de cinza (teste de impressão)</label>
{''.join(blocos)}</body></html>'''
    caminho.write_text(html, encoding='utf-8', newline='\n')


def gerar(destino=DESTINO, imagens=True, apenas=None):
    destino.mkdir(parents=True, exist_ok=True)
    figuras = []
    for gerador in GERADORES:
        if apenas and gerador.__name__.upper() not in apenas:
            continue
        f = gerador()
        f.dados.to_csv(destino / f'{f.nome}.csv', index=False, lineterminator='\n')
        f.figura.write_html(destino / f'{f.nome}.html', include_plotlyjs='cdn', config={'displaylogo': False})
        figuras.append(f)
        print(f'{f.nome}: validação {"ok" if f.validacao.get("ok") else "FALHOU"}', flush=True)
    if not apenas:
        escrever_tex(figuras, destino / 'graficos.tex')
        escrever_comparacao(figuras, destino / 'comparacao.html')
        relatorio = {f.nome: f.validacao for f in figuras}
        def nativo(v):  # tipos NumPy viram bool/int/float do Python (bool primeiro: np.bool_ não é int)
            if isinstance(v, np.bool_):
                return bool(v)
            if isinstance(v, np.integer):
                return int(v)
            return float(v)
        (destino / 'validacao.json').write_text(json.dumps(relatorio, indent=2, ensure_ascii=False, default=nativo) + '\n',
                                                encoding='utf-8', newline='\n')
    if imagens and figuras:
        bases = [destino / f.nome for f in figuras]
        exportar_imagens([f.figura for f in figuras], bases, escala_png=ESCALA_PNG, formatos=('.png', '.pdf', '.svg'))
        print(f'{len(figuras)} figuras em PNG (600 dpi), PDF e SVG', flush=True)
    falhas = [f.nome for f in figuras if not f.validacao.get('ok')]
    if falhas:
        raise SystemExit(f'Validação falhou: {falhas}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destino', type=Path, default=DESTINO)
    parser.add_argument('--sem-imagens', action='store_true')
    parser.add_argument('--apenas', nargs='*')
    args = parser.parse_args()
    gerar(args.destino, imagens=not args.sem_imagens, apenas={a.upper() for a in args.apenas} if args.apenas else None)
