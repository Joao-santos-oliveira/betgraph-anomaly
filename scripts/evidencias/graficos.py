"""Figuras 1 e 2: alvos recuperados por cenário e decoys em painel próprio.

Fonte: as mesmas métricas da tabela comparativa (`resultados/v3/metricas.csv` no
commit fixado em `tabela.py`), filtradas para `fase=injetado` e
`universo=completo`. Cada ponto é uma execução: o símbolo indica a realização
da intervenção e, nas redes, há um ponto por seed. A barra é a média.

Saídas em `figuras/`: HTML interativo, PDF e PNG de cada figura, e o CSV com os
pontos desenhados.

A exportação para PDF/PNG usa o kaleido 1.x, que precisa de um navegador
Chromium. No Windows, o Edge é usado automaticamente se BROWSER_PATH não estiver
definido. No Linux, use `plotly_get_chrome` ou defina BROWSER_PATH.

Uso:
    python scripts/evidencias/graficos.py [--saida figuras] [--sem-imagens]
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.evidencias import tabela  # noqa: E402

ROOT = tabela.ROOT
EDGE = Path(r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe')

# Paleta categórica de referência (modo claro), validada para 4 posições adjacentes.
# Dois tons ficam abaixo de 3:1 sobre o fundo: os nomes no eixo e os rótulos de média
# garantem que a identidade não dependa só da cor.
CORES = {'A': '#2a78d6', 'E_features': '#eb6834', 'E_v2': '#1baf7a', 'C': '#eda100'}
FUNDO, TEXTO, TEXTO_2, GRADE = '#ffffff', '#0b0b0b', '#52514e', '#e4e3df'
ROTULOS = {'A': 'Relacional<br>direto', 'E_features': 'Atributos +<br>IsolationForest',
           'E_v2': 'Híbrido<br>GNN v2', 'C': 'GNN v3<br>(pares)'}
SIMBOLOS = {30101: 'circle', 30201: 'diamond'}
DESLOCAMENTO_REALIZACAO = {30101: -0.18, 30201: 0.18}
DESLOCAMENTO_SEED = {0: 0.0, 42: -0.09, 73: 0.0, 2026: 0.09}
MEIA_BARRA = 0.31  # todos os pontos ficam dentro da largura da barra (0,62)
Y_MEDIA = 44  # acima do maior ponto observado (34) e abaixo da linha do denominador
FONTE = 'Arial'


def dados(metricas):
    """Pontos desenhados: uma linha por execução (método × cenário × realização × seed)."""
    base = metricas[(metricas.fase == 'injetado') & (metricas.universo == 'completo')
                    & metricas.metodo.isin(list(tabela.PRINCIPAIS))].copy()
    base['posicao'] = base.metodo.map({m: i for i, m in enumerate(tabela.PRINCIPAIS)})
    base['x'] = (base.posicao + base.realizacao.map(DESLOCAMENTO_REALIZACAO)
                 + base.seed_modelo.map(DESLOCAMENTO_SEED))
    if base.x.isna().any():
        raise ValueError('Realização ou seed sem posição definida no gráfico')
    colunas = ['cenario', 'metodo', 'realizacao', 'seed_modelo', 'n', 'alvos', 'top100', 'AP', 'x']
    ordem = ['cenario', 'posicao', 'realizacao', 'seed_modelo']
    return base.sort_values(ordem)[colunas].reset_index(drop=True)


def _num(valor, casas=1):
    return f'{valor:.{casas}f}'.replace('.', ',')


def _painel(fig, pontos, linha, coluna, mostrar_legenda, metrica_hover):
    """Barras de média por método e pontos por execução em um painel."""
    medias = pontos.groupby('metodo').top100.mean()
    for i, metodo in enumerate(tabela.PRINCIPAIS):
        media = medias.get(metodo, 0.0)
        fig.add_trace(go.Bar(
            x=[i], y=[media], width=2 * MEIA_BARRA, marker=dict(color=CORES[metodo], opacity=0.32,
                                                       line=dict(color=CORES[metodo], width=1.5),
                                                       cornerradius=4),
            showlegend=False,
            hovertemplate=f'{tabela.PRINCIPAIS[metodo][0]}<br>média: %{{y:.2f}}<extra></extra>'),
            row=linha, col=coluna)
        fig.add_annotation(x=i, y=Y_MEDIA, text=f'média<br><b>{_num(media)}</b>', showarrow=False,
                           font=dict(size=12, color=TEXTO), row=linha, col=coluna)
    for realizacao, simbolo in SIMBOLOS.items():
        sub = pontos[pontos.realizacao == realizacao]
        seed = sub.seed_modelo.map(lambda s: 'única (sem treino)' if s == 0 else str(s))
        fig.add_trace(go.Scatter(
            x=sub.x, y=sub.top100, mode='markers', name=f'realização {realizacao}',
            legendgroup=str(realizacao), showlegend=mostrar_legenda,
            marker=dict(symbol=simbolo, size=10, color=TEXTO_2, line=dict(color=FUNDO, width=2)),
            customdata=list(zip(sub.metodo.map(lambda m: tabela.PRINCIPAIS[m][0]), seed, sub.AP)),
            hovertemplate=('%{customdata[0]}<br>realização ' + str(realizacao)
                           + '<br>seed: %{customdata[1]}<br>' + metrica_hover
                           + ': %{y} de 50<br>AP: %{customdata[2]:.4f}<extra></extra>')),
            row=linha, col=coluna)


def _eixos(fig, colunas, titulo_y):
    for c in range(1, colunas + 1):
        fig.update_xaxes(tickvals=list(range(len(tabela.PRINCIPAIS))),
                         ticktext=[ROTULOS[m] for m in tabela.PRINCIPAIS], range=[-0.6, 3.6],
                         showgrid=False, ticks='', tickfont=dict(size=12, color=TEXTO_2),
                         showline=False, row=1, col=c)
        # O eixo começa abaixo de zero para que os pontos em 0 apareçam inteiros.
        fig.update_yaxes(range=[-2, 53], tickvals=[0, 10, 20, 30, 40, 50], gridcolor=GRADE,
                         zeroline=True, zerolinecolor=TEXTO_2, zerolinewidth=1,
                         tickfont=dict(size=12, color=TEXTO_2), row=1, col=c)
        fig.add_hline(y=50, line=dict(color=TEXTO_2, width=1, dash='dot'), row=1, col=c)
    fig.update_yaxes(title=dict(text=titulo_y, font=dict(size=13, color=TEXTO)), row=1, col=1)


def _layout(fig, titulo, nota, largura, altura):
    fig.update_layout(
        title=dict(text=titulo, x=0, xanchor='left', font=dict(size=17, color=TEXTO)),
        font=dict(family=FONTE, color=TEXTO), separators=',.', barmode='overlay',
        plot_bgcolor=FUNDO, paper_bgcolor=FUNDO, width=largura, height=altura,
        margin=dict(l=70, r=30, t=125, b=185),
        legend=dict(orientation='h', x=0, xanchor='left', y=1.09, yanchor='bottom',
                    font=dict(size=12, color=TEXTO_2)),
        hoverlabel=dict(bgcolor=FUNDO, font=dict(family=FONTE, color=TEXTO)))
    fig.add_annotation(text=nota, xref='paper', yref='paper', x=0, y=-0.30, xanchor='left',
                       yanchor='top', showarrow=False, align='left',
                       font=dict(size=11.5, color=TEXTO_2))
    for anotacao in fig.layout.annotations:
        if anotacao.text in tabela.CENARIOS.values():
            anotacao.font = dict(size=14, color=TEXTO)


def nota_comum(pontos, decoys=False):
    n = int(pontos.n.iloc[0])
    alvo, curto = ('decoys', 'dos decoys') if decoys else ('alvos artificiais', 'dos alvos')
    return (f'<b>Denominador:</b> 50 {alvo} entre {n:,} carteiras (prevalência de 1%); a linha pontilhada marca 50.'.replace(',', '.')
            + f'<br><b>Realizações:</b> 30101 (●) e 30201 (◆), dois sorteios independentes {curto} e da injeção sobre a mesma coorte.'
            + f'<br><b>Variação:</b> a distância entre ● e ◆ mostra a sensibilidade à escolha {curto}; nas redes, os três pontos de cada'
            + '<br>símbolo são as seeds 42, 73 e 2026 (só a inicialização muda). Os baselines são determinísticos: um ponto por realização.'
            + '<br><b>Barra:</b> média de todas as execuções do método. Fonte: resultados/v3/metricas.csv (commit '
            + f'{tabela.COMMIT_V3}), fase injetada, universo completo.')


def figura_recuperacao(pontos):
    cenarios = list(tabela.CENARIOS)
    fig = make_subplots(rows=1, cols=len(cenarios), shared_yaxes=True, horizontal_spacing=0.04,
                        subplot_titles=[tabela.CENARIOS[c] for c in cenarios])
    for j, cenario in enumerate(cenarios, start=1):
        _painel(fig, pontos[pontos.cenario == cenario], 1, j, j == 1, 'alvos no top 100')
    _eixos(fig, len(cenarios), 'Alvos recuperados no top 100 (de 50)')
    _layout(fig, 'Alvos recuperados no top 100 por cenário', nota_comum(pontos), 1250, 640)
    return fig


def figura_decoys(pontos):
    sub = pontos[pontos.cenario == tabela.DECOYS]
    fig = make_subplots(rows=1, cols=1, subplot_titles=['S2 controle de decoys'])
    _painel(fig, sub, 1, 1, True, 'decoys no top 100')
    _eixos(fig, 1, 'Decoys no top 100 (de 50) · menor é melhor')
    nota = nota_comum(sub, decoys=True).replace(
        '<br><b>Variação:</b>', '<br>Decoys: coparticipação forte, lados balanceados e horários dispersos; destacá-los é indesejado.<br><b>Variação:</b>')
    _layout(fig, 'Decoys priorizados no top 100', nota, 900, 680)
    fig.update_layout(margin=dict(b=215))
    fig.layout.annotations[-1].y = -0.33
    return fig


def configurar_navegador():
    if not os.environ.get('BROWSER_PATH') and os.name == 'nt' and EDGE.exists():
        os.environ['BROWSER_PATH'] = str(EDGE)


def salvar(fig, destino, nome, imagens=True):
    destino.mkdir(parents=True, exist_ok=True)
    fig.write_html(destino / f'{nome}.html', include_plotlyjs='cdn', full_html=True)
    if imagens:
        fig.write_image(destino / f'{nome}.pdf')
        fig.write_image(destino / f'{nome}.png', scale=3)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--saida', type=Path, default=ROOT / 'figuras')
    parser.add_argument('--sem-imagens', action='store_true', help='gera só HTML e CSV, sem PDF/PNG')
    args = parser.parse_args(argv)
    configurar_navegador()
    metricas, _ = tabela.ler_metricas()
    pontos = dados(metricas)
    args.saida.mkdir(parents=True, exist_ok=True)
    pontos.drop(columns='x').to_csv(args.saida / 'fig1_fig2_pontos.csv', index=False, lineterminator='\n')
    salvar(figura_recuperacao(pontos), args.saida, 'fig1_recuperacao', not args.sem_imagens)
    salvar(figura_decoys(pontos), args.saida, 'fig2_decoys', not args.sem_imagens)
    print(f'Figuras gravadas em {args.saida}')


if __name__ == '__main__':
    main()
