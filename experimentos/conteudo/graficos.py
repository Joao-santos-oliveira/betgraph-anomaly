"""Gera os gráficos do artigo (G1 a G6) e do suplemento (S1 a S4) com Plotly.

Cada gráfico sai em PDF vetorial, PNG (3x) e HTML interativo, mais um CSV com os valores
desenhados (por sorteio e seed) e um fragmento LaTeX com a legenda. Lê só resultados/artigo,
resultados/artigo/extra e benchmark/ (S4, opcional).
Uso: python -m experimentos.conteudo.graficos [--sem-imagens] [--destino entrega/graficos]
"""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experimentos.conteudo.estilo import (ENTREGA, EXTRA, FONTE, GRADE, METODOS, NEUTRO, ORDEM_METODOS, RESULTADOS,
                                          ROOT, SUPERFICIE, TINTA, TINTA_SUAVE, contagem, exportar_imagens, numero, tex)

DENOMINADOR = 50
INTENSIDADES = (4, 10, 20)
CORTES = (100, 541)
N_CARTEIRAS = 27039
EIXO = '#bdbcb8'
TRACOS_SEED = {42: 'solid', 73: 'dash', 2026: 'dot'}
SEED_FICHAS = 42


@dataclass
class Grafico:
    nome: str
    legenda: str
    figura: object
    dados: pd.DataFrame
    larga: bool = True


# Estilo comum -----------------------------------------------------------------------------------

def _go():
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    return go, make_subplots


def estilizar(fig, largura, altura, legenda_y=1.12):
    fig.update_layout(
        width=largura, height=altura, paper_bgcolor=SUPERFICIE, plot_bgcolor=SUPERFICIE, separators=',.',
        font=dict(family=FONTE, size=13, color=TINTA), margin=dict(l=72, r=24, t=64, b=60),
        legend=dict(orientation='h', x=0, y=legenda_y, xanchor='left', yanchor='bottom',
                    font=dict(size=12.5, color=TINTA), bgcolor='rgba(0,0,0,0)', itemwidth=40),
        hoverlabel=dict(font=dict(family=FONTE, size=12)))
    fig.update_xaxes(showline=True, linecolor=EIXO, linewidth=1, gridcolor=GRADE, gridwidth=1, zeroline=False,
                     ticks='outside', tickcolor=EIXO, tickfont=dict(color=TINTA_SUAVE, size=12),
                     title_font=dict(size=13, color=TINTA_SUAVE))
    fig.update_yaxes(showline=True, linecolor=EIXO, linewidth=1, gridcolor=GRADE, gridwidth=1, zeroline=False,
                     ticks='outside', tickcolor=EIXO, tickfont=dict(color=TINTA_SUAVE, size=12),
                     title_font=dict(size=13, color=TINTA_SUAVE))
    fig.update_annotations(font=dict(family=FONTE, size=13.5, color=TINTA))
    return fig


def _marcador(metodo, tamanho=10, **extra):
    m = METODOS[metodo]
    return dict(color=m['cor'], symbol=m['marcador'], size=tamanho, line=dict(color=SUPERFICIE, width=1.5), **extra)


def _ler(caminho):
    if not caminho.exists():
        raise FileNotFoundError(f'{caminho} não existe: rode make dados-artigo RODADA=<pasta da execução>')
    return caminho


# Dados de recuperação por ranking ---------------------------------------------------------------

def recuperacao_por_ranking():
    """Percentual de alvos no corte em cada ranking (unidade da dispersão)."""
    misto = pd.read_csv(RESULTADOS / 'por_realizacao_seed.csv').query("fase == 'depois' and papel == 'alvo'")
    misto = misto.groupby(['metodo', 'seed', 'mercados'])[['n', 'top100', 'top541']].sum().reset_index()
    assert misto.n.eq(DENOMINADOR).all()
    misto = misto.assign(desenho='Misto', unidade='seed ' + misto.seed.astype(str))
    pareado = pd.read_csv(RESULTADOS / 'pareado' / 'por_sorteio_seed.csv').query("papel == 'alvo' and dose > 0")
    pareado = pareado.rename(columns={'dose': 'mercados'}).assign(
        desenho='Pareado', n=DENOMINADOR,
        unidade=lambda d: 'sorteio ' + d.realizacao.astype(str) + ', seed ' + d.seed.astype(str))
    linhas = []
    for frame in (misto, pareado):
        for _, r in frame.iterrows():
            for corte in CORTES:
                linhas.append(dict(desenho=r.desenho, mercados=int(r.mercados), metodo=r.metodo, unidade=r.unidade,
                                   corte=corte, alvos=int(r[f'top{corte}']), denominador=int(r.n),
                                   percentual=100 * r[f'top{corte}'] / r.n))
    return pd.DataFrame(linhas)


# G1 ---------------------------------------------------------------------------------------------

def g1():
    go, make_subplots = _go()
    dados = recuperacao_por_ranking()
    fig = make_subplots(rows=2, cols=2, shared_xaxes=True, shared_yaxes=True, horizontal_spacing=0.05,
                        vertical_spacing=0.09, column_titles=['Teste misto', 'Teste pareado'])
    deslocamento = dict(zip(ORDEM_METODOS, (-0.1, 0, 0.1)))
    posicao = {m: i for i, m in enumerate(INTENSIDADES)}
    for linha, corte in enumerate(CORTES, start=1):
        for coluna, desenho in enumerate(('Misto', 'Pareado'), start=1):
            parte = dados[(dados.desenho == desenho) & (dados.corte == corte)]
            for metodo in ORDEM_METODOS:
                d = parte[parte.metodo == metodo]
                x = d.mercados.map(posicao) + deslocamento[metodo]
                jitter = np.random.default_rng(7).uniform(-0.035, 0.035, len(d))
                fig.add_trace(go.Scatter(
                    x=x + jitter, y=d.percentual, mode='markers', showlegend=False, opacity=0.45,
                    marker=_marcador(metodo, 6), customdata=d[['unidade', 'alvos']],
                    hovertemplate=f'{METODOS[metodo]["rotulo"]}<br>%{{customdata[0]}}<br>'
                                  '%{customdata[1]}/50 alvos (%{y:.0f}%)<extra></extra>'), row=linha, col=coluna)
                media = d.groupby('mercados').percentual.mean()
                fig.add_trace(go.Scatter(
                    x=[posicao[m] + deslocamento[metodo] for m in media.index], y=media.values, mode='lines+markers',
                    name=METODOS[metodo]['rotulo'], legendgroup=metodo, showlegend=(linha == 1 and coluna == 1),
                    line=dict(color=METODOS[metodo]['cor'], width=2), marker=_marcador(metodo, 10),
                    hovertemplate=f'{METODOS[metodo]["rotulo"]}: média %{{y:.1f}}%<extra></extra>'),
                    row=linha, col=coluna)
    fig.update_xaxes(tickvals=list(range(3)), ticktext=[str(m) for m in INTENSIDADES], range=[-0.5, 2.5])
    fig.update_xaxes(title_text='Novos mercados em comum por grupo', row=2)
    fig.update_yaxes(range=[-5, 105], ticksuffix='%', dtick=25)
    fig.update_yaxes(title_text='Top 100: alvos no corte', row=1, col=1)
    fig.update_yaxes(title_text='Top 541: alvos no corte', row=2, col=1)
    estilizar(fig, 980, 640, legenda_y=1.06)
    fig.update_layout(margin=dict(t=80))
    return Grafico('G1_recuperacao_intensidade',
                   'Recuperação dos alvos artificiais por intensidade. Percentual dos 50 alvos no top 100 (acima) e no '
                   'top 541 (abaixo). Pontos claros: cada ranking (misto: soma dos três sorteios por seed; pareado: '
                   'cada sorteio e seed); marcadores ligados: média. O relacional é determinístico (um ranking por '
                   'sorteio). Mesma escala nos quatro painéis.', fig, dados)


# G2 ---------------------------------------------------------------------------------------------

def g2():
    go, _ = _go()
    p = pd.read_csv(RESULTADOS / 'pareado' / 'por_sorteio_seed.csv').query("metodo == 'GNN híbrida'")
    por_sorteio = p.groupby(['realizacao', 'dose', 'papel']).top100.mean().reset_index()
    resumo = por_sorteio.groupby(['dose', 'papel']).top100.agg(['mean', 'min', 'max']).reset_index()
    rotulos = {0: 'Original', 4: '4', 10: '10', 20: '20'}
    fig = go.Figure()
    estilos = {'alvo': ('Alvos', METODOS['GNN híbrida']['cor'], 'circle'), 'controle': ('Controles', NEUTRO, 'square')}
    for papel, (nome, cor, simbolo) in estilos.items():
        d = resumo[resumo.papel == papel].sort_values('dose')
        x = d.dose.map(rotulos)
        fig.add_trace(go.Scatter(
            x=x, y=d['mean'], mode='lines+markers', name=nome, line=dict(color=cor, width=2),
            marker=dict(color=cor, symbol=simbolo, size=10, line=dict(color=SUPERFICIE, width=1.5)),
            error_y=dict(type='data', symmetric=False, array=d['max'] - d['mean'], arrayminus=d['mean'] - d['min'],
                         color=cor, thickness=1.5, width=5),
            customdata=d[['min', 'max']],
            hovertemplate=f'{nome}: %{{y:.2f}} de 50<br>sorteios: %{{customdata[0]:.0f}} a %{{customdata[1]:.0f}}'
                          '<extra></extra>'))
        ultimo = d.iloc[-1]
        fig.add_annotation(x=3, y=ultimo['mean'], text=f'{nome}: {contagem(ultimo["mean"])}',
                           showarrow=False, xanchor='left', xshift=14, font=dict(color=TINTA, size=12.5))
    fig.update_yaxes(range=[-2, 52], dtick=10, title_text='Carteiras no top 100 (de 50)')
    fig.update_xaxes(type='category', title_text='Novos mercados em comum por grupo', range=[-0.3, 3.9])
    estilizar(fig, 720, 460)
    dados = resumo.rename(columns={'mean': 'media_top100', 'min': 'minimo_sorteio', 'max': 'maximo_sorteio'})
    return Grafico('G2_alvos_vs_controles',
                   'Alvos e controles no teste pareado (GNN híbrida). Média de carteiras no top 100, de 50 em cada '
                   'categoria. Unidade: sorteio (média das três seeds); as barras vão do menor ao maior dos três '
                   'sorteios. Controles têm o mesmo grau inicial do alvo e ligações intactas.',
                   fig, dados, larga=False)


# G3 e S2: curvas das ligações reservadas ---------------------------------------------------------

def curvas_ligacoes():
    scores = pd.read_csv(_ler(EXTRA / 'scores_teste.csv.gz'))
    linhas, metricas = [], {}
    for seed, d in scores.groupby('seed'):
        ordem = np.argsort(-d.score.to_numpy(), kind='mergesort')
        y = d.rotulo.to_numpy()[ordem]
        s = d.score.to_numpy()[ordem]
        tp, fp = np.cumsum(y), np.cumsum(1 - y)
        # Só os limiares distintos (empates no mesmo ponto da curva).
        corte = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
        tp, fp = tp[corte], fp[corte]
        tpr, fpr = np.r_[0, tp / y.sum()], np.r_[0, fp / (1 - y).sum()]
        precisao = np.r_[1, tp / (tp + fp)]
        auc = float(np.trapezoid(tpr, fpr))
        ap = float(np.sum(np.diff(tpr) * precisao[1:]))
        metricas[int(seed)] = dict(auroc=auc, ap=ap)
        manter = np.unique(np.r_[np.linspace(0, len(tpr) - 1, 2500).astype(int), np.flatnonzero(fpr <= 0.1)[::4]])
        linhas.append(pd.DataFrame({'seed': int(seed), 'fpr': fpr[manter], 'tpr': tpr[manter],
                                    'precisao': precisao[manter]}))
    return pd.concat(linhas, ignore_index=True), metricas


def g3():
    go, make_subplots = _go()
    curvas, metricas = curvas_ligacoes()
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1,
                        column_titles=['Curva completa', 'Detalhe: até 10% de falsos positivos'])
    cor = METODOS['GNN híbrida']['cor']
    for coluna in (1, 2):
        limite = 1 if coluna == 1 else 0.1
        fig.add_trace(go.Scatter(x=[0, limite], y=[0, limite], mode='lines', line=dict(color=NEUTRO, width=1),
                                 name='Aleatório', showlegend=coluna == 1, hoverinfo='skip'), row=1, col=coluna)
        for seed, d in curvas.groupby('seed'):
            d = d[d.fpr <= limite]
            fig.add_trace(go.Scatter(
                x=d.fpr, y=d.tpr, mode='lines', legendgroup=str(seed), showlegend=coluna == 1,
                name=f'Seed {seed} · AUROC {numero(metricas[seed]["auroc"], 4)}',
                line=dict(color=cor, width=2, dash=TRACOS_SEED[seed]),
                hovertemplate=f'Seed {seed}<br>FPR %{{x:.3f}}<br>TPR %{{y:.3f}}<extra></extra>'), row=1, col=coluna)
    fig.update_xaxes(title_text='Taxa de falsos positivos')
    fig.update_yaxes(title_text='Taxa de verdadeiros positivos', row=1, col=1)
    fig.update_xaxes(range=[0, 1], dtick=0.2, row=1, col=1)
    fig.update_yaxes(range=[0, 1.01], dtick=0.2, row=1, col=1)
    fig.update_xaxes(range=[0, 0.1], dtick=0.02, row=1, col=2)
    fig.update_yaxes(range=[0, 1.01], dtick=0.2, row=1, col=2)
    estilizar(fig, 980, 480, legenda_y=1.13)
    fig.update_layout(margin=dict(t=92))
    dados = curvas.drop(columns='precisao').merge(pd.DataFrame(metricas).T.rename_axis('seed').reset_index(), on='seed')
    return Grafico('G3_curva_roc',
                   'Curva ROC da previsão de ligações no teste reservado (22.907 positivos e 22.907 negativos), com o '
                   'grafo de contexto: uma curva por seed. A ROC avalia a previsão de ligações; a recuperação '
                   'artificial (G1) avalia o ranking de carteiras.', fig, dados)


def s2():
    go, _ = _go()
    curvas, metricas = curvas_ligacoes()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0.5, 0.5], mode='lines', line=dict(color=NEUTRO, width=1),
                             name='Aleatório (prevalência 0,5)', hoverinfo='skip'))
    for seed, d in curvas.groupby('seed'):
        d = d.iloc[1:]
        fig.add_trace(go.Scatter(x=d.tpr, y=d.precisao, mode='lines',
                                 name=f'Seed {seed} · AP {numero(metricas[seed]["ap"], 4)}',
                                 line=dict(color=METODOS['GNN híbrida']['cor'], width=2, dash=TRACOS_SEED[seed]),
                                 hovertemplate='Revocação %{x:.3f}<br>Precisão %{y:.3f}<extra></extra>'))
    fig.update_xaxes(title_text='Revocação', range=[0, 1], dtick=0.2)
    fig.update_yaxes(title_text='Precisão', range=[0.45, 1.01], dtick=0.1)
    estilizar(fig, 720, 460, legenda_y=1.04)
    fig.update_layout(margin=dict(t=96), legend=dict(itemwidth=30))
    dados = curvas.drop(columns='fpr').rename(columns={'tpr': 'revocacao'})
    return Grafico('S2_precisao_revocacao',
                   'Curva precisão–revocação das ligações reservadas, por seed. Linha cinza: precisão de um '
                   'classificador aleatório com negativos 1:1.', fig, dados, larga=False)


# G4 ---------------------------------------------------------------------------------------------

def g4():
    go, make_subplots = _go()
    traj = pd.read_csv(RESULTADOS / 'pareado' / 'trajetorias.csv').query("papel == 'alvo'")
    gnns = ['GNN híbrida', 'Similaridade da GNN']
    traj = traj[traj.metodo.isin(gnns)]
    real = pd.read_csv(RESULTADOS / 'posicoes_dia_real.csv')
    relacional = set(real.wallet_id[real['Relacional_0'] <= 100])
    sobreposicao = pd.DataFrame([dict(metodo=m, seed=s, top100_comum_relacional=len(
        set(real.wallet_id[real[f'{m}_{s}'] <= 100]) & relacional)) for m in gnns for s in (42, 73, 2026)])
    fig = make_subplots(rows=1, cols=2, column_widths=[0.66, 0.34], horizontal_spacing=0.12,
                        column_titles=['Posição dos alvos no teste pareado', 'Dia real: top 100 em comum com o relacional'])
    rotulos = {0: 'Original', 4: '4', 10: '10', 20: '20'}
    for metodo in gnns:
        d = traj[traj.metodo == metodo]
        m = METODOS[metodo]
        fig.add_trace(go.Box(x=d.dose.map(rotulos), y=d.posicao, name=m['rotulo'], legendgroup=metodo,
                             marker=dict(color=m['cor'], size=4, symbol=m['marcador']), line=dict(width=1.5),
                             fillcolor='rgba(0,0,0,0)', boxpoints=False, offsetgroup=metodo,
                             hovertemplate=f'{m["rotulo"]}<br>%{{x}} mercados<br>posição %{{y}}<extra></extra>'),
                      row=1, col=1)
        s = sobreposicao[sobreposicao.metodo == metodo]
        fig.add_trace(go.Bar(x=[f'Seed {v}' for v in s.seed], y=s.top100_comum_relacional, name=m['rotulo'],
                             legendgroup=metodo, showlegend=False, marker=dict(color=m['cor']), width=0.32,
                             offsetgroup=metodo, text=s.top100_comum_relacional, textposition='outside',
                             textfont=dict(color=TINTA, size=12),
                             hovertemplate=f'{m["rotulo"]}: %{{y}} de 100<extra></extra>'), row=1, col=2)
    for corte in CORTES:
        fig.add_hline(y=corte, line=dict(color=NEUTRO, width=1), row=1, col=1)
        # Em eixo log, a anotação usa o expoente (log10), não o valor.
        fig.add_annotation(x=1, xref='x domain', y=np.log10(corte), yref='y', text=f'top {corte}', showarrow=False,
                           xanchor='right', yanchor='bottom', font=dict(size=11.5, color=TINTA_SUAVE))
    fig.update_layout(boxmode='group', bargap=0.35)
    fig.update_yaxes(type='log', range=[np.log10(N_CARTEIRAS) + 0.08, np.log10(0.8)],
                     title_text='Posição no ranking (escala log)', row=1, col=1,
                     tickvals=[1, 10, 100, 1000, 10000], ticktext=['1', '10', '100', '1.000', '10.000'])
    fig.update_xaxes(type='category', title_text='Novos mercados em comum por grupo', row=1, col=1)
    fig.update_yaxes(range=[0, 112], title_text='Carteiras em comum (de 100)', row=1, col=2)
    estilizar(fig, 980, 500)
    fig.update_layout(margin=dict(t=92))
    caixas = traj.groupby(['metodo', 'dose']).posicao.describe(percentiles=[.25, .5, .75]).reset_index()
    dados = pd.concat([caixas.assign(painel='posicao_alvos'), sobreposicao.assign(painel='dia_real')], ignore_index=True)
    return Grafico('G4_comparacao_gnns',
                   'GNN híbrida versus similaridade isolada. As duas usam os mesmos embeddings e o mesmo treino '
                   '(mesma AUROC); só muda o score de carteira. À esquerda, distribuição das posições dos 50 alvos '
                   'em cada ranking do teste pareado (3 sorteios × 3 seeds; caixa: quartis; posição 1 no alto). À '
                   'direita, carteiras do top 100 de cada GNN que também estão no top 100 do relacional no dia real.',
                   fig, dados)


# G5 ---------------------------------------------------------------------------------------------

def g5():
    go, make_subplots = _go()
    traj = pd.read_csv(RESULTADOS / 'pareado' / 'trajetorias.csv').query(
        "metodo == 'GNN híbrida' and seed == @SEED_FICHAS")
    rotulos = {0: 'Original', 4: '4', 10: '10', 20: '20'}
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.04,
                        column_titles=['Alvos', 'Controles'])
    cores = {'alvo': (METODOS['GNN híbrida']['cor'], 'rgba(42,120,214,0.16)'),
             'controle': ('#52514e', 'rgba(138,137,134,0.20)')}
    for coluna, papel in enumerate(('alvo', 'controle'), start=1):
        d = traj[traj.papel == papel]
        forte, fraca = cores[papel]
        xs, ys = [], []
        for _, w in d.sort_values('dose').groupby(['realizacao', 'wallet_id']):
            xs += list(w.dose.map(rotulos)) + [None]
            ys += list(w.posicao) + [None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', line=dict(color=fraca, width=1), hoverinfo='skip',
                                 showlegend=False), row=1, col=coluna)
        mediana = d.groupby('dose').posicao.median()
        fig.add_trace(go.Scatter(x=mediana.index.map(rotulos), y=mediana.values, mode='lines+markers',
                                 name=f'Mediana dos {"alvos" if papel == "alvo" else "controles"}',
                                 line=dict(color=forte, width=2.5),
                                 marker=dict(color=forte, size=9, line=dict(color=SUPERFICIE, width=1.5)),
                                 hovertemplate='Mediana: posição %{y:.0f}<extra></extra>'), row=1, col=coluna)
        fig.add_hline(y=100, line=dict(color=NEUTRO, width=1), row=1, col=coluna)
    fig.add_annotation(x=0, y=np.log10(100), xref='x', yref='y', text='top 100', showarrow=False, yshift=-9,
                       xanchor='left', font=dict(size=11.5, color=TINTA_SUAVE))
    fig.update_yaxes(type='log', range=[np.log10(N_CARTEIRAS) + 0.08, np.log10(0.8)],
                     tickvals=[1, 10, 100, 1000, 10000], ticktext=['1', '10', '100', '1.000', '10.000'])
    fig.update_yaxes(title_text='Posição no ranking (escala log)', row=1, col=1)
    fig.update_xaxes(type='category', title_text='Novos mercados em comum por grupo')
    estilizar(fig, 980, 500)
    fig.update_layout(margin=dict(t=92))
    dados = traj[['realizacao', 'wallet_id', 'grupo', 'papel', 'grau', 'dose', 'posicao']]
    return Grafico('G5_trajetorias_pareadas',
                   'Trajetória de cada carteira do teste pareado ao longo das intensidades (GNN híbrida, seed 42, '
                   'três sorteios: 150 alvos e 150 controles). Linhas finas: carteiras; linha forte: mediana. '
                   'As intensidades são cumulativas e os controles mantêm suas ligações.', fig, dados)


# G6 ---------------------------------------------------------------------------------------------

def g6():
    go, _ = _go()
    cob = pd.read_csv(RESULTADOS / 'cobertura_total.csv')
    s = SEED_FICHAS
    ctx, tot = cob[f'contexto_{s}'], cob[f'total_{s}']
    d = cob[(ctx <= 100) | (tot <= 100)].assign(contexto=ctx, total=tot)
    d['grupo'] = np.select([(d.contexto <= 100) & (d.total <= 100), d.total <= 100], ['permaneceu', 'entrou'], 'saiu')
    casos = pd.read_csv(EXTRA / 'casos_dia_real.csv').query("papel == 'caso'") if (EXTRA / 'casos_dia_real.csv').exists() else None
    estilos = {'permaneceu': ('Permaneceram', 'rgba(138,137,134,0.45)', 1),
               'saiu': ('Saíram', METODOS['Relacional']['cor'], 2),
               'entrou': ('Entraram', METODOS['GNN híbrida']['cor'], 2)}
    fig = go.Figure()
    for grupo, (nome, cor, largura) in estilos.items():
        parte = d[d.grupo == grupo]
        xs, ys, texto = [], [], []
        for _, r in parte.iterrows():
            xs += ['Contexto', 'Cobertura total', None]
            ys += [r.contexto, r.total, None]
            texto += [r.wallet_id, r.wallet_id, None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines+markers', name=f'{nome} ({len(parte)})',
                                 line=dict(color=cor, width=largura),
                                 marker=dict(color=cor, size=6 if grupo == 'permaneceu' else 8,
                                             line=dict(color=SUPERFICIE, width=1)),
                                 text=texto, hovertemplate='Carteira %{text}<br>posição %{y}<extra></extra>'))
    fig.add_hline(y=100, line=dict(color=NEUTRO, width=1))
    fig.add_annotation(x=0, xref='x domain', y=2, yref='y', text='top 100', showarrow=False, xanchor='left',
                       yanchor='top', font=dict(size=11.5, color=TINTA_SUAVE))
    if casos is not None:
        for _, c in casos.iterrows():
            r = d[d.wallet_id == c.wallet_id].iloc[0]
            rotulo = 'permaneceu' if c.tipo == 'permaneceu' else 'entrou'
            fig.add_annotation(x='Cobertura total', y=np.log10(r.total), text=f'{c.wallet_id} ({rotulo}): '
                               f'{numero(r.contexto)} → {numero(r.total)}', showarrow=True, arrowhead=0, ax=60, ay=0,
                               arrowcolor=TINTA_SUAVE, xanchor='left', font=dict(size=12, color=TINTA))
    fig.update_yaxes(type='log', range=[np.log10(max(d.contexto.max(), d.total.max()) * 1.25), np.log10(0.8)],
                     title_text='Posição no ranking (escala log)',
                     tickvals=[1, 3, 10, 30, 100, 300], ticktext=['1', '3', '10', '30', '100', '300'])
    fig.update_xaxes(range=[-0.25, 1.9], showgrid=False)
    estilizar(fig, 720, 560, legenda_y=1.03)
    dados = d[['wallet_id', 'contexto', 'total', 'grupo', 'mercados_contexto', 'mercados_total']].sort_values('total')
    return Grafico('G6_entradas_saidas_top100',
                   'Top 100 do dia real com as ligações de contexto e com a cobertura total (GNN híbrida, seed 42). '
                   'Cada linha liga as duas posições de uma carteira. Os casos da ficha T6 estão destacados.',
                   fig, dados, larga=False)


# S1 ---------------------------------------------------------------------------------------------

def s1():
    go, make_subplots = _go()
    traj = pd.read_csv(RESULTADOS / 'pareado' / 'trajetorias.csv').query("papel == 'alvo' and dose in [10, 20]")
    k = np.unique(np.r_[np.round(np.logspace(0, np.log10(N_CARTEIRAS), 160)).astype(int), CORTES])
    linhas = []
    for (dose, metodo, real, seed), d in traj.groupby(['dose', 'metodo', 'realizacao', 'seed']):
        pos = np.sort(d.posicao.to_numpy())
        linhas.append(pd.DataFrame({'dose': dose, 'metodo': metodo, 'realizacao': real, 'seed': seed, 'k': k,
                                    'percentual': 100 * np.searchsorted(pos, k, side='right') / len(pos)}))
    curvas = pd.concat(linhas, ignore_index=True)
    media = curvas.groupby(['dose', 'metodo', 'k']).percentual.mean().reset_index()
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.04,
                        column_titles=['10 novos mercados', '20 novos mercados'])
    for coluna, dose in enumerate((10, 20), start=1):
        for metodo in ORDEM_METODOS:
            d = media[(media.dose == dose) & (media.metodo == metodo)]
            fig.add_trace(go.Scatter(x=d.k, y=d.percentual, mode='lines', name=METODOS[metodo]['rotulo'],
                                     legendgroup=metodo, showlegend=coluna == 1,
                                     line=dict(color=METODOS[metodo]['cor'], width=2,
                                               dash='solid' if metodo != 'Relacional' else 'dash'),
                                     hovertemplate=f'{METODOS[metodo]["rotulo"]}<br>até a posição %{{x}}: '
                                                   '%{y:.1f}%<extra></extra>'), row=1, col=coluna)
        for corte in CORTES:
            fig.add_vline(x=corte, line=dict(color=NEUTRO, width=1), row=1, col=coluna)
            fig.add_annotation(x=np.log10(corte), y=104, xref=f'x{coluna if coluna > 1 else ""}', yref='y',
                               text=f'top {corte}', showarrow=False, xanchor='right', xshift=-3,
                               font=dict(size=11, color=TINTA_SUAVE))
    fig.update_xaxes(type='log', title_text='Profundidade do ranking (escala log)',
                     tickvals=[1, 10, 100, 1000, 10000], ticktext=['1', '10', '100', '1.000', '10.000'])
    fig.update_yaxes(range=[0, 108], ticksuffix='%', dtick=25)
    fig.update_yaxes(title_text='Alvos até a posição', row=1, col=1)
    estilizar(fig, 980, 460)
    fig.update_layout(margin=dict(t=92))
    return Grafico('S1_recuperacao_acumulada',
                   'Recuperação acumulada por profundidade do ranking no teste pareado: percentual dos 50 alvos até '
                   'cada posição, média dos rankings (3 sorteios × 3 seeds; relacional tracejado, por sobrepor a '
                   'GNN híbrida).', fig, curvas)


# S3 ---------------------------------------------------------------------------------------------

def s3():
    go, make_subplots = _go()
    hist = pd.read_csv(_ler(EXTRA / 'historico_treino.csv'))
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1,
                        column_titles=['Perda por época', 'AUROC de validação por época'])
    series = {'link_loss': ('Treino (ligações)', METODOS['GNN híbrida']['cor']),
              'val_loss': ('Validação', METODOS['Relacional']['cor'])}
    for coluna_dado, (nome, cor) in series.items():
        for seed, d in hist.groupby('seed'):
            fig.add_trace(go.Scatter(x=d.epoch, y=d[coluna_dado], mode='lines', name=f'{nome} · seed {seed}',
                                     legendgroup=coluna_dado, line=dict(color=cor, width=2, dash=TRACOS_SEED[seed]),
                                     hovertemplate=f'{nome}, seed {seed}<br>época %{{x}}: %{{y:.4f}}<extra></extra>'),
                          row=1, col=1)
    for seed, d in hist.groupby('seed'):
        fig.add_trace(go.Scatter(x=d.epoch, y=d.val_auc, mode='lines', showlegend=False,
                                 line=dict(color=METODOS['Relacional']['cor'], width=2, dash=TRACOS_SEED[seed]),
                                 hovertemplate=f'Seed {seed}<br>época %{{x}}: %{{y:.4f}}<extra></extra>'), row=1, col=2)
    fig.update_xaxes(title_text='Época')
    fig.update_yaxes(title_text='Perda (entropia cruzada)', row=1, col=1)
    fig.update_yaxes(title_text='AUROC', row=1, col=2)
    estilizar(fig, 980, 460, legenda_y=1.12)
    fig.update_layout(margin=dict(t=104), legend=dict(itemwidth=30, font=dict(size=11.5)))
    return Grafico('S3_perdas_treino',
                   'Treino da GNN híbrida por seed (traço: 42 contínuo, 73 tracejado, 2026 pontilhado). A perda de '
                   'treino inclui só o termo de ligações; o contraste (peso 0,1) fica de fora.', fig, hist)


# S4 ---------------------------------------------------------------------------------------------

def s4():
    go, make_subplots = _go()
    resumo = pd.read_csv(_ler(ROOT / 'benchmark' / 'resumo.csv'))
    hardware = json.loads((ROOT / 'benchmark' / 'hardware.json').read_text(encoding='utf-8'))
    resumo = resumo[resumo.etapa != 'total'].iloc[::-1]
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.04,
                        column_titles=['Tempo de parede (s)', 'Pico de memória (MiB)'])
    cor = METODOS['GNN híbrida']['cor']
    for coluna, (med, mn, mx, fmt) in enumerate((('segundos_mediana', 'segundos_min', 'segundos_max', '{:.1f}'),
                                                 ('rss_mib_mediana', 'rss_mib_min', 'rss_mib_max', '{:.0f}')), start=1):
        fig.add_trace(go.Bar(y=resumo.rotulo, x=resumo[med], orientation='h', marker=dict(color=cor), width=0.55,
                             showlegend=False,
                             error_x=dict(type='data', symmetric=False, array=resumo[mx] - resumo[med],
                                          arrayminus=resumo[med] - resumo[mn], color=TINTA_SUAVE, thickness=1.2),
                             customdata=resumo[[mn, mx]],
                             hovertemplate='%{y}: mediana %{x:.1f} (%{customdata[0]:.1f} a %{customdata[1]:.1f})'
                                           '<extra></extra>'), row=1, col=coluna)
        # Rótulo da mediana depois do fim da barra de erro, para não colidir com ela.
        fig.add_trace(go.Scatter(x=resumo[mx], y=resumo.rotulo, mode='text', showlegend=False, hoverinfo='skip',
                                 text=[numero(v, 1 if coluna == 1 else 0) for v in resumo[med]],
                                 textposition='middle right', textfont=dict(color=TINTA, size=11.5),
                                 cliponaxis=False), row=1, col=coluna)
    cpu = hardware.get('cpu', {}).get('Model name', hardware.get('maquina', ''))
    fig.update_xaxes(rangemode='tozero')
    estilizar(fig, 980, 120 + 34 * len(resumo))
    fig.update_layout(margin=dict(l=210, t=64, r=60))
    legenda = (f'Custo por etapa do pipeline do dia real: mediana de {int(resumo.repeticoes.iloc[0])} repetições '
               f'(barras: mínimo e máximo). Máquina: {cpu}, {hardware.get("nucleos_logicos")} núcleos lógicos, '
               f'{numero(hardware.get("memoria_total_mib", 0) / 1024, 1)} GiB; 2 threads por processo.')
    return Grafico('S4_tempos_memoria', legenda, fig, resumo.iloc[::-1])


# Saídas -----------------------------------------------------------------------------------------

GERADORES = (g1, g2, g3, g4, g5, g6, s1, s2, s3, s4)


def escrever_tex(graficos, caminho):
    linhas = ['% Gerado por experimentos/conteudo/graficos.py. Requer \\usepackage{graphicx}.']
    for g in graficos:
        ambiente = 'figure*' if g.larga else 'figure'
        largura = '\\textwidth' if g.larga else '\\columnwidth'
        linhas += [f'\\begin{{{ambiente}}}[htbp]', '\\centering', f'\\includegraphics[width={largura}]{{{g.nome}.pdf}}',
                   f'\\caption{{{tex(g.legenda)}}}', f'\\label{{fig:{g.nome.split("_")[0].lower()}}}',
                   f'\\end{{{ambiente}}}', '']
    caminho.write_text('\n'.join(linhas), encoding='utf-8', newline='\n')


def exportar(graficos, destino):
    exportar_imagens([g.figura for g in graficos], [destino / g.nome for g in graficos])


def gerar(destino, imagens=True, apenas=None):
    destino.mkdir(parents=True, exist_ok=True)
    feitos = []
    for gerador in GERADORES:
        if apenas and gerador.__name__.upper() not in apenas:
            continue
        try:
            g = gerador()
        except FileNotFoundError as erro:
            print(f'{gerador.__name__.upper()}: pulado ({erro})', flush=True)
            continue
        g.dados.to_csv(destino / f'{g.nome}.csv', index=False, lineterminator='\n')
        g.figura.write_html(destino / f'{g.nome}.html', include_plotlyjs='cdn', full_html=True,
                            config={'displaylogo': False})
        feitos.append(g)
        print(f'{g.nome}: {len(g.dados)} linhas de dados', flush=True)
    if not apenas:  # uma lista parcial não sobrescreve o índice completo
        escrever_tex(feitos, destino / 'graficos.tex')
    if imagens and feitos:
        exportar(feitos, destino)
        print(f'{len(feitos)} gráficos em PNG e PDF', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destino', type=Path, default=ENTREGA / 'graficos')
    parser.add_argument('--sem-imagens', action='store_true', help='só CSV, HTML e LaTeX (sem navegador para o kaleido)')
    parser.add_argument('--apenas', nargs='*', help='ex.: --apenas G1 G3')
    args = parser.parse_args()
    gerar(args.destino, imagens=not args.sem_imagens, apenas={a.upper() for a in args.apenas} if args.apenas else None)
