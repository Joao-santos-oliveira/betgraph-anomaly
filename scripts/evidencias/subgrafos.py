"""Figura 3: subgrafo de um caso e de um controle com atividade semelhante.

Duas fases, para separar o que precisa de torch do que precisa de navegador:

  selecionar  (precisa de torch) lê o grafo de contexto de uma execução de
              `make comparar ESCOPO=5000`, aplica a regra de seleção e grava
              `entrega/subgrafos_selecao.json` e `figuras/fig3_subgrafos_dados.json`.
  desenhar    (precisa de plotly e kaleido) lê esse JSON e grava a figura em
              HTML, PDF e PNG.

Regra de seleção, fixada antes de olhar os grafos:
  * Caso: a carteira na posição 1 do ranking relacional da coorte de 5.000.
  * Controle: entre as carteiras fora do top 100 do mesmo ranking, a mais próxima
    do caso em z-score de log(1 + grau no contexto) e log(1 + volume no contexto),
    com caliper de 0,75 desvio-padrão em cada dimensão (o mesmo pareamento das
    fichas de revisão em experimentos/validacao_ampliada/casos.py).
Corte da vizinhança, igual para os dois:
  * Mercados: os mercados da carteira focal no contexto, no máximo 10. Primeiro
    os compartilhados com o parceiro principal, depois os de maior volume da focal.
  * Carteiras: a focal, o parceiro principal (o que mais explica o score
    relacional; sem ele, a carteira com mais mercados em comum) e até 5 outras
    carteiras com pelo menos 2 mercados entre os desenhados, por número de
    mercados em comum.
  * Conexões: todas as ligações de contexto entre as carteiras e os mercados
    desenhados. Volume e lado vêm de edge.csv para o mesmo par carteira–mercado.

Uso:
    python scripts/evidencias/subgrafos.py selecionar --execucao <pasta v3-...>
    python scripts/evidencias/subgrafos.py desenhar
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MAX_MERCADOS = 10
MAX_OUTRAS = 5
MIN_COMPARTILHADOS = 2
CALIPER = 0.75
FORA_DO_TOPO = 100
FORWARD = ('user', 'bets_on', 'event')  # mesmo tipo de aresta de src/preparar_gnn.py
DADOS = ROOT / 'figuras' / 'fig3_subgrafos_dados.json'
SELECAO = ROOT / 'entrega' / 'subgrafos_selecao.json'


# --------------------------------------------------------------------------- seleção

def carregar(execucao):
    """Pares de contexto em IDs persistentes, arestas agregadas e ranking relacional."""
    import torch
    base = torch.load(execucao / 'resultados' / 'contexto.pt', weights_only=False)
    g = base['context']
    uids, eids = g['user'].original_id.numpy(), g['event'].original_id.numpy()
    local = g[FORWARD].edge_index.numpy()
    pares = pd.DataFrame({'wallet_id': uids[local[0]], 'condition_node_id': eids[local[1]]})
    entrada = execucao / 'projeto' / 'data' / 'processed' / 'validacao_ampliada_v1' / 'novo'
    arestas = pd.read_csv(entrada / 'edge.csv')
    eventos = pd.read_csv(entrada / 'event.csv', usecols=['condition_node_id', 'question'])
    ranking = pd.read_csv(execucao / 'resultados' / 'relacional_42' / 'ranking.csv')
    return pares, (local, uids, len(uids), len(eids)), arestas, eventos, ranking


def atributos_das_ligacoes(pares, arestas):
    """Volume total e lados por par carteira–mercado, só nos pares do contexto."""
    agg = (arestas.groupby(['wallet_id', 'condition_node_id'])
           .agg(volume_usd=('volume_usd', 'sum'), n_trades=('n_trades', 'sum'),
                lados=('token_outcome', lambda s: '/'.join(sorted(map(str, s.unique())))))
           .reset_index())
    ligacoes = pares.merge(agg, on=['wallet_id', 'condition_node_id'], how='left')
    if ligacoes.volume_usd.isna().any():
        raise ValueError('Par de contexto sem aresta correspondente em edge.csv')
    return ligacoes


def relacional(grafo):
    """Recalcula o score relacional e o parceiro principal de cada carteira."""
    from experimentos.investigacao_anomalias.relacional import scores_relacionais
    local, uids, n_u, n_e = grafo
    stats = scores_relacionais(local, n_u, n_e)
    parceiro = np.where(stats['melhor_parceiro_indice'] >= 0,
                        uids[np.clip(stats['melhor_parceiro_indice'], 0, None)], -1)
    return pd.DataFrame({'wallet_id': uids, 'score': stats['surpresa_coparticipacao'],
                         'parceiro': parceiro, 'compartilhados_parceiro': stats['mercados_compartilhados'],
                         'grau': stats['grau_contexto'].astype(int)})


def parear(caso, atividade, candidatos, caliper=CALIPER):
    """Vizinho mais próximo em z-score de log(1+grau) e log(1+volume), com caliper."""
    log = np.log1p(atividade[['grau', 'volume_usd']])
    z = (log - log.mean()) / log.std(ddof=0)
    delta = z.loc[sorted(candidatos)] - z.loc[caso]
    ok = delta.abs().max(axis=1) <= caliper
    if not ok.any():
        raise ValueError('Nenhum controle dentro do caliper')
    distancia = np.sqrt((delta.loc[ok] ** 2).sum(axis=1)).sort_values(kind='stable')
    controle = int(distancia.index[0])
    return controle, {'distancia_z': float(distancia.iloc[0]),
                      'delta_grau_z': float(delta.loc[controle, 'grau']),
                      'delta_volume_z': float(delta.loc[controle, 'volume_usd']),
                      'candidatos_no_caliper': int(ok.sum()), 'caliper': caliper}


def vizinhanca(focal, parceiro, ligacoes):
    """Aplica o corte documentado e devolve carteiras, mercados, ligações e o que foi omitido."""
    mercados_de = ligacoes.groupby('wallet_id').condition_node_id.agg(set)
    proprios = ligacoes[ligacoes.wallet_id == focal].set_index('condition_node_id')
    if parceiro < 0:  # sem parceiro relacional: a carteira com mais mercados em comum
        comuns = mercados_de.drop(focal).map(lambda m: len(m & set(proprios.index)))
        parceiro = int(comuns.idxmax()) if comuns.max() > 0 else -1
    compartilhados = set(proprios.index) & mercados_de.get(parceiro, set())
    ordem = sorted(proprios.index, key=lambda m: (m not in compartilhados, -proprios.volume_usd[m], m))
    mercados = ordem[:MAX_MERCADOS]
    conjunto = set(mercados)
    comuns = mercados_de.drop(focal).map(lambda m: len(m & conjunto))
    tocam = comuns[comuns > 0]
    outras = (tocam.drop(parceiro, errors='ignore')[lambda s: s >= MIN_COMPARTILHADOS]
              .reset_index().sort_values(['condition_node_id', 'wallet_id'], ascending=[False, True])
              .wallet_id.head(MAX_OUTRAS).tolist())
    carteiras = [focal] + ([parceiro] if parceiro >= 0 else []) + [int(w) for w in outras]
    desenho = ligacoes[ligacoes.wallet_id.isin(carteiras) & ligacoes.condition_node_id.isin(conjunto)]
    return {'focal': int(focal), 'parceiro': int(parceiro), 'carteiras': [int(c) for c in carteiras],
            'mercados': [int(m) for m in mercados], 'ligacoes': desenho,
            'compartilhados_com_parceiro': len(compartilhados),
            'omitidos': {'mercados_da_focal': len(proprios) - len(mercados),
                         'carteiras_que_tocam_os_mercados': int(len(tocam) - (len(carteiras) - 1)),
                         'mercados_totais_da_focal': int(len(proprios))}}


def selecionar(execucao):
    pares, grafo, arestas, eventos, ranking = carregar(execucao)
    ligacoes = atributos_das_ligacoes(pares, arestas)
    rel = relacional(grafo).set_index('wallet_id')
    ordem = rel.reset_index().sort_values(['score', 'wallet_id'], ascending=[False, True]).wallet_id.tolist()
    if ordem != ranking.wallet_id.tolist():
        raise ValueError('Score recalculado não reproduz o ranking relacional da execução')
    posicao = {w: i + 1 for i, w in enumerate(ordem)}
    atividade = rel[['grau']].join(ligacoes.groupby('wallet_id').volume_usd.sum()).fillna(0.0)

    caso = ordem[0]
    candidatos = set(ordem[FORA_DO_TOPO:]) - {int(rel.parceiro[caso])}
    controle, pareamento = parear(caso, atividade, candidatos)

    paineis, perguntas = {}, eventos.set_index('condition_node_id').question
    for papel, focal in (('caso', caso), ('controle', controle)):
        viz = vizinhanca(focal, int(rel.parceiro[focal]), ligacoes)
        lig = viz.pop('ligacoes')
        paineis[papel] = {
            **viz, 'posicao_focal': posicao[focal], 'score_focal': float(rel.score[focal]),
            'grau_focal': int(atividade.grau[focal]), 'volume_focal_usd': float(atividade.volume_usd[focal]),
            'carteiras_info': [{'wallet_id': w, 'posicao': posicao[w], 'grau': int(atividade.grau[w])}
                               for w in viz['carteiras']],
            'mercados_info': [{'condition_node_id': m, 'pergunta': str(perguntas.get(m, ''))}
                              for m in viz['mercados']],
            'ligacoes': lig.sort_values(['wallet_id', 'condition_node_id']).to_dict('records')}
    selecao = {'execucao': execucao.name, 'universo': {'carteiras': len(ordem), 'pares_contexto': len(pares)},
               'regra': {'caso': 'posição 1 do ranking relacional (relacional_42)',
                         'controle': f'fora do top {FORA_DO_TOPO}; vizinho mais próximo em z de log(1+grau) '
                                     f'e log(1+volume) no contexto; caliper {CALIPER}',
                         'corte': f'até {MAX_MERCADOS} mercados da focal (compartilhados com o parceiro '
                                  f'primeiro, depois por volume); focal, parceiro e até {MAX_OUTRAS} '
                                  f'carteiras com ≥{MIN_COMPARTILHADOS} mercados desenhados'},
               'pareamento': pareamento,
               'resumo': {p: {k: v for k, v in d.items() if k not in ('ligacoes', 'carteiras_info', 'mercados_info')}
                          for p, d in paineis.items()}}
    SELECAO.parent.mkdir(parents=True, exist_ok=True)
    DADOS.parent.mkdir(parents=True, exist_ok=True)
    SELECAO.write_text(json.dumps(selecao, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    DADOS.write_text(json.dumps({'selecao': selecao, 'paineis': paineis}, indent=2, ensure_ascii=False,
                                default=float) + '\n', encoding='utf-8')
    print(f'Caso W{caso} (posição 1), controle W{controle} (posição {posicao[controle]}); '
          f'distância z {pareamento["distancia_z"]:.3f}')


# --------------------------------------------------------------------------- desenho

FUNDO, TEXTO, TEXTO_2, GRADE = '#ffffff', '#0b0b0b', '#52514e', '#c9c8c2'
COR_FOCAL = {'caso': '#2a78d6', 'controle': '#eb6834'}
COR_PARCEIRO = '#6250d6'
X_RANGE = [-1.05, 2.25]  # carteiras em x=0 e mercados em x=1; o resto é espaço para rótulos


def _curto(texto, n=46):
    return texto if len(texto) <= n else texto[:n - 1] + '…'


def _dec(valor, sinal=''):
    return f'{valor:{sinal}.2f}'.replace('.', ',')


def _usd(valor):
    # "USD" em vez do cifrão: dois "$" no mesmo texto ativam o modo LaTeX do Plotly.
    return f'USD {valor:,.0f}'.replace(',', '.')


def _painel(fig, painel, papel, coluna):
    import plotly.graph_objects as go
    carteiras, mercados = painel['carteiras'], painel['mercados']
    ligacoes = pd.DataFrame(painel['ligacoes'])
    grau_no_desenho = ligacoes.groupby('condition_node_id').size()
    mercados = sorted(mercados, key=lambda m: (-grau_no_desenho.get(m, 0), m))
    passo_c = 1 / max(len(carteiras) - 1, 1)
    passo_m = 1 / max(len(mercados) - 1, 1)
    y_c = {w: 1 - i * passo_c for i, w in enumerate(carteiras)}
    y_m = {m: 1 - i * passo_m for i, m in enumerate(mercados)}
    vmax = np.log1p(ligacoes.volume_usd.max())

    def cor(w):
        return COR_FOCAL[papel] if w == painel['focal'] else COR_PARCEIRO if w == painel['parceiro'] else GRADE

    # Desenha primeiro as coparticipantes, depois parceiro e focal, para que fiquem por cima.
    prioridade = ligacoes.wallet_id.map(lambda w: 2 if w == painel['focal'] else 1 if w == painel['parceiro'] else 0)
    for _, l in ligacoes.assign(_p=prioridade).sort_values(['_p', 'wallet_id']).iterrows():
        destaque = l._p > 0
        relativo = np.log1p(l.volume_usd) / vmax
        largura = (1.2 + 3.8 * relativo) if destaque else (0.6 + 1.4 * relativo)
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[y_c[l.wallet_id], y_m[l.condition_node_id]], mode='lines', showlegend=False,
            opacity=0.9 if destaque else 0.55, line=dict(color=cor(l.wallet_id), width=largura),
            hovertemplate=(f'W{l.wallet_id} → M{l.condition_node_id}<br>volume: {_usd(l.volume_usd)}'
                           f'<br>lado: {l.lados}<br>operações: {int(l.n_trades)}<extra></extra>')),
            row=1, col=coluna)

    info_c = {c['wallet_id']: c for c in painel['carteiras_info']}
    rotulos_c = []
    for w in carteiras:
        papel_w = papel if w == painel['focal'] else 'parceiro' if w == painel['parceiro'] else 'coparticipante'
        rotulos_c.append(f"<b>W{w}</b> · {papel_w}<br>posição {info_c[w]['posicao']} · {info_c[w]['grau']} mercados")
    fig.add_trace(go.Scatter(
        x=[0] * len(carteiras), y=[y_c[w] for w in carteiras], mode='markers+text', showlegend=False, cliponaxis=False,
        text=rotulos_c, textposition='middle left', textfont=dict(size=11, color=TEXTO),
        marker=dict(symbol='circle', size=[18 if w == painel['focal'] else 13 for w in carteiras],
                    color=[cor(w) if cor(w) != GRADE else TEXTO_2 for w in carteiras],
                    line=dict(color=FUNDO, width=2)),
        hovertemplate='%{text}<extra></extra>'), row=1, col=coluna)

    perguntas = {m['condition_node_id']: m['pergunta'] for m in painel['mercados_info']}
    fig.add_trace(go.Scatter(
        x=[1] * len(mercados), y=[y_m[m] for m in mercados], mode='markers+text', showlegend=False, cliponaxis=False,
        text=[f'<b>M{m}</b> · {_curto(perguntas[m])}' for m in mercados], textposition='middle right',
        textfont=dict(size=10.5, color=TEXTO),
        marker=dict(symbol='square', size=11, color=TEXTO_2, line=dict(color=FUNDO, width=2)),
        customdata=[perguntas[m] for m in mercados],
        hovertemplate='%{customdata}<extra></extra>'), row=1, col=coluna)


def _texto_painel(painel, papel):
    o = painel['omitidos']
    return (f"{'Caso' if papel == 'caso' else 'Controle'} W{painel['focal']}: posição {painel['posicao_focal']}, "
            f"{painel['grau_focal']} mercados, {_usd(painel['volume_focal_usd'])} no contexto; "
            f"{painel['compartilhados_com_parceiro']} mercados em comum com o parceiro. "
            f"Omitidos: {o['mercados_da_focal']} de {o['mercados_totais_da_focal']} mercados da focal e "
            f"{o['carteiras_que_tocam_os_mercados']} carteiras que tocam os mercados desenhados.")


def figura(dados):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    p, sel = dados['paineis'], dados['selecao']
    titulos = [f"Caso · W{p['caso']['focal']} (posição {p['caso']['posicao_focal']})",
               f"Controle · W{p['controle']['focal']} (posição {p['controle']['posicao_focal']})"]
    fig = make_subplots(rows=1, cols=2, subplot_titles=titulos, horizontal_spacing=0.03)
    _painel(fig, p['caso'], 'caso', 1)
    _painel(fig, p['controle'], 'controle', 2)
    for c in (1, 2):
        fig.update_xaxes(visible=False, range=X_RANGE, row=1, col=c)
        fig.update_yaxes(visible=False, range=[-0.06, 1.06], row=1, col=c)
    # Legenda das codificações (traços fantasmas fora da área visível).
    for nome, cor_ in [('carteira focal (caso)', COR_FOCAL['caso']), ('carteira focal (controle)', COR_FOCAL['controle']),
                       ('parceiro principal', COR_PARCEIRO), ('outras carteiras', TEXTO_2)]:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name=nome,
                                 marker=dict(symbol='circle', size=11, color=cor_)))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name='mercado',
                             marker=dict(symbol='square', size=10, color=TEXTO_2)))
    par = sel['pareamento']
    nota = ('<b>Seleção:</b> caso = posição 1 do ranking relacional na coorte de 5.000 carteiras. Controle = carteira fora do '
            'top 100 mais próxima em z-score de log(1+grau) e log(1+volume) no contexto,<br>com caliper de 0,75 DP em cada '
            f"dimensão (distância z {_dec(par['distancia_z'])}; Δgrau {_dec(par['delta_grau_z'], '+')}; Δvolume "
            f"{_dec(par['delta_volume_z'], '+')}; {par['candidatos_no_caliper']} candidatos no caliper)."
            '<br><b>Corte:</b> até 10 mercados da carteira focal (compartilhados com o parceiro principal primeiro, depois '
            'por volume), a focal, o parceiro principal e até 5 carteiras com ≥2 mercados desenhados.'
            '<br><b>Conexões:</b> ligações do grafo de contexto (o mesmo que o score relacional usa); espessura ∝ log(volume '
            'da carteira no mercado). Lado operado e nº de operações aparecem no HTML ao passar o mouse.'
            f"<br>{_texto_painel(p['caso'], 'caso')}<br>{_texto_painel(p['controle'], 'controle')}"
            '<br>IDs persistentes: W = wallet_id, M = condition_node_id (mapeamentos versionados na raiz do repositório).')
    fig.add_annotation(text=nota, xref='paper', yref='paper', x=0, y=-0.085, xanchor='left', yanchor='top',
                       showarrow=False, align='left', font=dict(size=11.5, color=TEXTO_2))
    fig.update_layout(
        title=dict(text='Subgrafos de contexto: caso e controle com atividade semelhante', x=0.01, xanchor='left',
                   y=0.97, font=dict(size=17, color=TEXTO)),
        font=dict(family='Arial', color=TEXTO), plot_bgcolor=FUNDO, paper_bgcolor=FUNDO,
        width=1750, height=940, margin=dict(l=30, r=30, t=90, b=250),
        legend=dict(orientation='h', x=0, y=-0.01, yanchor='top', font=dict(size=12, color=TEXTO_2)),
        hoverlabel=dict(bgcolor=FUNDO, font=dict(family='Arial', color=TEXTO)))
    for a in fig.layout.annotations[:2]:
        a.font = dict(size=14, color=TEXTO)
    return fig


def desenhar(imagens=True):
    from scripts.evidencias.graficos import configurar_navegador, salvar
    dados = json.loads(DADOS.read_text(encoding='utf-8'))
    configurar_navegador()
    salvar(figura(dados), ROOT / 'figuras', 'fig3_subgrafos', imagens)
    print(f'Figura gravada em {ROOT / "figuras"}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='fase', required=True)
    s = sub.add_parser('selecionar')
    s.add_argument('--execucao', type=Path, required=True, help='pasta v3-... de make comparar ESCOPO=5000')
    d = sub.add_parser('desenhar')
    d.add_argument('--sem-imagens', action='store_true')
    args = parser.parse_args(argv)
    if args.fase == 'selecionar':
        selecionar(args.execucao.resolve())
    else:
        desenhar(not args.sem_imagens)


if __name__ == '__main__':
    main()
