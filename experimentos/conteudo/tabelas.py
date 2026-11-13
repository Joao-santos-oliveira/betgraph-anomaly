"""Gera as tabelas T1 a T6 do artigo em CSV, LaTeX (booktabs) e imagem (PNG e PDF).

Lê apenas resultados/artigo e resultados/artigo/extra (gerado por make dados-artigo).
Uso: python -m experimentos.conteudo.tabelas [--sem-imagens] [--destino entrega/tabelas]
"""
import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path

import pandas as pd

from experimentos.conteudo.estilo import (ENTREGA, EXTRA, FONTE, FUNDO_ALTERNADO, FUNDO_CABECALHO, LINHA,
                                          METODOS, ORDEM_METODOS, RESULTADOS, TINTA, TINTA_SUAVE,
                                          contagem, numero, percentual, tex)

DENOMINADOR = 50
CORTES = (100, 541)
INTENSIDADES = (4, 10, 20)


@dataclass
class Tabela:
    nome: str
    titulo: str
    nota: str
    dados: pd.DataFrame
    exibicao: pd.DataFrame
    alinhamento: str
    grupos_colunas: list = field(default_factory=list)  # [(rótulo, quantas colunas)] acima do cabeçalho
    divisores: list = field(default_factory=list)  # linhas antes das quais entra um \midrule
    larga: bool = False


def _json(caminho):
    return json.loads(Path(caminho).read_text(encoding='utf-8'))


def _extra(nome):
    caminho = EXTRA / nome
    if not caminho.exists():
        raise FileNotFoundError(f'{caminho} não existe: rode make dados-artigo RODADA=<pasta da execução>')
    return caminho


# T1 ---------------------------------------------------------------------------------------------

def t1():
    d = _json(_extra('divisao.json'))
    total = d['ligacoes']
    dia = pd.Timestamp(d['dia']).strftime('%d/%m/%Y')
    linhas = [
        ('Grafo do dia', 'Dia', dia, None, 'Negociações registradas em um único dia'),
        ('Grafo do dia', 'Carteiras', d['carteiras'], None,
         f"{numero(d['carteiras_com_contexto'])} com ligação no contexto"),
        ('Grafo do dia', 'Mercados', d['mercados'], None, f"{numero(d['mercados_com_contexto'])} com ligação no contexto"),
        ('Grafo do dia', 'Ligações distintas carteira–mercado', total, 100.0, 'Cobertura total, usada no ranking do dia real'),
        ('Divisão das ligações', 'Contexto', d['contexto'], None, 'Passagem de mensagens e scores de carteira'),
        ('Divisão das ligações', 'Treino', d['treino'], None, 'Positivos; negativos renovados a cada época'),
        ('Divisão das ligações', 'Validação', d['validacao'], None, 'Parada antecipada; negativos 1:1 (seed 43)'),
        ('Divisão das ligações', 'Teste reservado', d['teste'], None, 'Avaliado uma única vez; negativos 1:1 (seed 44)'),
        ('Sorteios e seeds', 'Seed da divisão', d['seed_divisao'], None, 'Mesma divisão para todas as seeds'),
        ('Sorteios e seeds', 'Seeds do modelo', ', '.join(map(str, d['seeds_modelo'])), None, 'Uma GNN treinada por seed'),
        ('Sorteios e seeds', 'Sorteios de alvos', ', '.join(map(str, d['realizacoes'])), None,
         '10 grupos de 5 carteiras e 50 controles por sorteio'),
    ]
    dados = pd.DataFrame(linhas, columns=['bloco', 'item', 'valor', 'percentual', 'detalhe'])
    for i in range(4, 8):
        dados.loc[i, 'percentual'] = 100 * dados.loc[i, 'valor'] / total
    exibir = dados.copy()
    exibir['valor'] = [numero(v) if isinstance(v, int) and i < 8 else str(v) for i, v in enumerate(dados.valor)]
    exibir['percentual'] = [percentual(p) if pd.notna(p) else '' for p in dados.percentual]
    exibir = exibir[['item', 'valor', 'percentual', 'detalhe']]
    exibir.columns = ['Item', 'Valor', '% das ligações', 'Detalhe']
    return Tabela('T1_dados_divisao', 'Dados do dia 16/09/2026 e divisão das ligações',
                  'Contexto, treino, validação e teste somam as 229.072 ligações distintas. O ranking do dia real '
                  'aplica os pesos treinados no contexto à cobertura total. Negativos amostrados uniformemente entre '
                  'pares carteira–mercado não observados.',
                  dados, exibir, 'llrl', divisores=[4, 8])


# T2 ---------------------------------------------------------------------------------------------

def t2():
    avaliacao = _json(RESULTADOS / 'avaliacao_cobertura.json')
    linhas = [dict(seed=int(s), auroc=v['auc'], ap=v['ap'], positivos=v['positivos'], negativos=v['n'] - v['positivos'])
              for s, v in avaliacao.items()]
    dados = pd.DataFrame(linhas)
    media = dados[['auroc', 'ap']].mean()
    dp = dados[['auroc', 'ap']].std(ddof=1)
    dados = pd.concat([dados, pd.DataFrame([dict(seed='média', auroc=media.auroc, ap=media.ap,
                                                 positivos=dados.positivos.iloc[0], negativos=dados.negativos.iloc[0]),
                                            dict(seed='desvio padrão', auroc=dp.auroc, ap=dp.ap)])],
                      ignore_index=True)
    exibir = pd.DataFrame({
        'Seed': [str(s) if not isinstance(s, str) else s.capitalize() for s in dados.seed],
        'AUROC': [numero(v, 4) for v in dados.auroc],
        'AP': [numero(v, 4) for v in dados.ap],
        'Positivos': [numero(v) if pd.notna(v) else '' for v in dados.positivos],
        'Negativos': [numero(v) if pd.notna(v) else '' for v in dados.negativos],
    })
    exibir.loc[3, ['AUROC', 'AP']] = [f'{numero(media.auroc, 4)} ± {numero(dp.auroc, 4)}',
                                      f'{numero(media.ap, 4)} ± {numero(dp.ap, 4)}']
    exibir = exibir.iloc[:4]
    return Tabela('T2_previsao_ligacoes', 'Previsão de ligações no teste reservado',
                  'Teste reservado: 22.907 ligações positivas e 22.907 negativas (1:1), amostradas uniformemente entre '
                  'pares não observados em nenhuma parte da divisão (seed 44), com o grafo de contexto. '
                  'Média ± desvio padrão amostral entre as três seeds. O MLP não entra: não foi avaliado nos mesmos pares.',
                  dados, exibir, 'lrrrr', divisores=[3])


# T3 ---------------------------------------------------------------------------------------------

def _recuperacao():
    misto = pd.read_csv(RESULTADOS / 'resumo.csv').query("fase == 'depois' and papel == 'alvo'")
    misto = misto.assign(desenho='Misto', agregacao='soma dos 3 sorteios por ranking; média entre seeds')
    pareado = pd.read_csv(RESULTADOS / 'pareado' / 'resumo.csv').query("papel == 'alvo' and dose > 0")
    pareado = pareado.rename(columns={'dose': 'mercados'}).assign(
        desenho='Pareado', n=DENOMINADOR, agregacao='média de sorteio × seed (cada ranking com os 50 alvos)')
    linhas = []
    for frame in (misto, pareado):
        assert frame.n.eq(DENOMINADOR).all()
        for _, r in frame.iterrows():
            for corte in CORTES:
                valor = r[f'top{corte}']
                linhas.append(dict(desenho=r.desenho, mercados=int(r.mercados), metodo=r.metodo, corte=corte,
                                   contagem=valor, denominador=DENOMINADOR, percentual=100 * valor / DENOMINADOR,
                                   agregacao=r.agregacao))
    return pd.DataFrame(linhas)


def t3():
    dados = _recuperacao()
    dados['metodo'] = pd.Categorical(dados.metodo, ORDEM_METODOS)
    dados = dados.sort_values(['desenho', 'mercados', 'metodo', 'corte']).reset_index(drop=True)
    linhas = []
    for (desenho, mercados), parte in dados.groupby(['desenho', 'mercados'], observed=True, sort=False):
        linha = {'Desenho': desenho, 'Mercados novos': str(mercados)}
        for metodo in ORDEM_METODOS:
            for corte in CORTES:
                r = parte[(parte.metodo == metodo) & (parte.corte == corte)].iloc[0]
                linha[f'{METODOS[metodo]["rotulo"]} · top {corte}'] = (
                    f'{contagem(r.contagem)}/{DENOMINADOR} ({percentual(r.percentual)})')
        linhas.append(linha)
    exibir = pd.DataFrame(linhas)
    exibir.loc[exibir.Desenho.duplicated(), 'Desenho'] = ''
    grupos = [('', 2)] + [(METODOS[m]['rotulo'], len(CORTES)) for m in ORDEM_METODOS]
    return Tabela('T3_recuperacao_alvos', 'Recuperação dos alvos artificiais por intensidade',
                  'Contagem de alvos no corte / denominador (percentual). Misto: cada sorteio aplica uma intensidade a '
                  '3 ou 4 grupos (15 ou 20 alvos); soma-se os 3 sorteios em cada ranking (50 alvos) e faz-se a média '
                  'entre as seeds. Pareado: os 50 alvos de cada sorteio recebem cada intensidade; média dos 9 rankings '
                  '(3 sorteios × 3 seeds; 3 rankings no relacional). Corte 541 = 2% das 27.039 carteiras.',
                  dados, exibir, 'll' + 'r' * 6, grupos_colunas=grupos, divisores=[3], larga=True)


# T4 ---------------------------------------------------------------------------------------------

def t4():
    resumo = pd.read_csv(RESULTADOS / 'resumo.csv').query("metodo == 'GNN híbrida'")
    linhas = []
    for (mercados, papel), parte in resumo.groupby(['mercados', 'papel']):
        antes = parte[parte.fase == 'antes'].iloc[0]
        depois = parte[parte.fase == 'depois'].iloc[0]
        linha = dict(mercados=int(mercados), papel=papel, denominador=int(antes.n))
        for corte in CORTES:
            linha[f'top{corte}_antes'] = antes[f'top{corte}']
            linha[f'top{corte}_depois'] = depois[f'top{corte}']
            linha[f'top{corte}_variacao'] = depois[f'top{corte}'] - antes[f'top{corte}']
        linhas.append(linha)
    dados = pd.DataFrame(linhas)
    exibir = pd.DataFrame({'Mercados novos': dados.mercados.astype(str), 'Grupo': dados.papel.map(
        {'alvo': 'Alvos', 'controle': 'Controles'})})
    exibir.loc[exibir['Mercados novos'].duplicated(), 'Mercados novos'] = ''
    for corte in CORTES:
        exibir[f'Antes (top {corte})'] = dados[f'top{corte}_antes'].map(contagem) + f'/{DENOMINADOR}'
        exibir[f'Depois (top {corte})'] = dados[f'top{corte}_depois'].map(contagem) + f'/{DENOMINADOR}'
        exibir[f'Variação (top {corte})'] = dados[f'top{corte}_variacao'].map(
            lambda v: ('+' if v > 0 else '') + contagem(v))
    grupos = [('', 2), ('Top 100', 3), ('Top 541', 3)]
    return Tabela('T4_alvos_controles', 'GNN híbrida antes e depois da intervenção: alvos e controles',
                  'Teste misto, média entre as três seeds. "Antes" é a posição das mesmas carteiras no grafo original, '
                  'ou seja, quantas já estavam no corte. Alvos recebem o padrão artificial; cada controle tem o mesmo '
                  'grau inicial do seu alvo e ligações intactas.',
                  dados, exibir, 'll' + 'r' * 6, grupos_colunas=grupos, divisores=[2, 4])


# T5 ---------------------------------------------------------------------------------------------

def t5():
    cobertura = pd.read_csv(RESULTADOS / 'cobertura_total.csv')
    avaliacao = _json(RESULTADOS / 'avaliacao_cobertura.json')
    casos = pd.read_csv(EXTRA / 'casos_dia_real.csv').query("papel == 'caso'") if (EXTRA / 'casos_dia_real.csv').exists() else None
    linhas = []
    for seed in avaliacao:
        ctx, tot = cobertura[f'contexto_{seed}'], cobertura[f'total_{seed}']
        antes = set(cobertura.wallet_id[ctx <= 100])
        depois = set(cobertura.wallet_id[tot <= 100])
        lider_ctx = int(cobertura.wallet_id[ctx == 1].iloc[0])
        lider_tot = int(cobertura.wallet_id[tot == 1].iloc[0])
        assert lider_ctx == avaliacao[seed]['primeiro_contexto'] and lider_tot == avaliacao[seed]['primeiro_total']
        assert len(antes & depois) == avaliacao[seed]['top100_comuns']
        linha = dict(seed=int(seed), permaneceram=len(antes & depois), entraram=len(depois - antes),
                     sairam=len(antes - depois), lider_contexto=lider_ctx, lider_total=lider_tot,
                     # Posições já são postos sem empate: Spearman = Pearson entre elas (sem exigir scipy).
                     mudou_lideranca=lider_ctx != lider_tot, spearman=ctx.corr(tot))
        if casos is not None:
            for _, c in casos.iterrows():
                r = cobertura[cobertura.wallet_id == c.wallet_id].iloc[0]
                linha[f'caso_{c.tipo}'] = int(c.wallet_id)
                linha[f'caso_{c.tipo}_contexto'] = int(r[f'contexto_{seed}'])
                linha[f'caso_{c.tipo}_total'] = int(r[f'total_{seed}'])
        linhas.append(linha)
    dados = pd.DataFrame(linhas)
    exibir = pd.DataFrame({
        'Seed': dados.seed.astype(str), 'Permaneceram': dados.permaneceram.astype(str),
        'Entraram': dados.entraram.astype(str), 'Saíram': dados.sairam.astype(str),
        'Líder (contexto → total)': [f'{a} → {b}' for a, b in zip(dados.lider_contexto, dados.lider_total)],
        'ρ de Spearman': dados.spearman.map(lambda v: numero(v, 3)),
    })
    if casos is not None:
        for tipo in ('permaneceu', 'entrou'):
            wallet = dados[f'caso_{tipo}'].iloc[0]
            exibir[f'Caso que {tipo} ({wallet})'] = [
                f'{numero(a)} → {numero(b)}' for a, b in zip(dados[f'caso_{tipo}_contexto'], dados[f'caso_{tipo}_total'])]
    return Tabela('T5_cobertura_dia_real', 'Ranking do dia real: contexto versus cobertura total (top 100)',
                  'Contexto: ranking com as 137.444 ligações de contexto. Total: mesmos pesos aplicados às 229.072 '
                  'ligações. Permaneceram, entraram e saíram referem-se ao top 100. ρ de Spearman entre as posições '
                  'das 27.039 carteiras. Casos selecionados na seed 42 (T6); posições como contexto → total.',
                  dados, exibir, 'l' + 'r' * (len(exibir.columns) - 1), larga=casos is not None)


# T6 ---------------------------------------------------------------------------------------------

def t6():
    casos = pd.read_csv(_extra('casos_dia_real.csv'))
    partes = pd.read_csv(_extra('contribuicoes_casos.csv'))
    dados = casos.merge(partes, on=['tipo', 'papel', 'wallet_id'], how='left')
    linhas = []
    for _, c in casos.iterrows():
        linha = {
            'Seleção': {'permaneceu': 'Permaneceu', 'entrou': 'Entrou'}[c.tipo] if c.papel == 'caso' else '',
            'Papel': c.papel.capitalize(), 'Carteira': str(c.wallet_id),
            'Posição (contexto → total)': f'{numero(c.posicao_contexto)} → {numero(c.posicao_total)}',
            'Score': numero(c.score, 2), 'Mercados': numero(c.mercados_total), 'Parceiros': numero(c.parceiros),
            'Eventos distintos': numero(c.eventos_distintos),
            'Mercados de evento indeterminado': numero(c.mercados_evento_indeterminado),
        }
        top = partes[(partes.wallet_id == c.wallet_id) & (partes.tipo == c.tipo)].sort_values('ordem')
        for i in range(3):
            if i < len(top):
                p = top.iloc[i]
                linha[f'Contribuição {i + 1}'] = (f'{p.parceiro}: {numero(p.contribuicao, 1)} '
                                                  f'(obs. {p.sobreposicao_observada}; esp. {numero(p.sobreposicao_esperada, 2)})')
            else:
                linha[f'Contribuição {i + 1}'] = '—'
        linhas.append(linha)
    exibir = pd.DataFrame(linhas)
    return Tabela('T6_fichas_casos', 'Fichas dos casos e controles do dia real (seed 42)',
                  'Pelo ranking total da seed 42: o primeiro que permaneceu no top 100 e o primeiro que entrou. Cada '
                  'controle está fora do top 100 com o número de mercados mais próximo (desempate pelo menor ID). '
                  'Score = média das três maiores contribuições; contribuição = surpresa (log10 da cauda de Poisson da '
                  'sobreposição) × (1 + cosseno)/2. obs./esp.: mercados compartilhados observados e esperados pelos graus. '
                  'Parceiros: carteiras com 2 ou mais mercados em comum. Evento identificado quando a pergunta nomeia um '
                  'confronto (“A vs. B” ou placar exato); os demais mercados contam como evento indeterminado.',
                  dados, exibir, 'lllrrrrrrlll', divisores=[2], larga=True)


# Saídas -----------------------------------------------------------------------------------------

def escrever_tex(t, caminho):
    colunas = list(t.exibicao.columns)
    ambiente = 'table*' if t.larga else 'table'
    linhas = [f'% Gerado por experimentos/conteudo/tabelas.py a partir de {t.nome}.csv. Requer \\usepackage{{booktabs}}.',
              f'\\begin{{{ambiente}}}[htbp]', '\\centering', '\\small' if not t.larga else '\\footnotesize',
              f'\\caption{{{tex(t.titulo)}}}', f'\\label{{tab:{t.nome.split("_")[0].lower()}}}']
    if t.larga:
        linhas.append('\\resizebox{\\textwidth}{!}{%')
    linhas += [f'\\begin{{tabular}}{{{t.alinhamento}}}', '\\toprule']
    if t.grupos_colunas:
        celulas, regras, inicio = [], [], 1
        for rotulo, span in t.grupos_colunas:
            if rotulo:
                celulas.append(f'\\multicolumn{{{span}}}{{c}}{{{tex(rotulo)}}}')
                regras.append(f'\\cmidrule(lr){{{inicio}-{inicio + span - 1}}}')
            else:
                celulas += [''] * span
            inicio += span
        linhas.append(' & '.join(celulas) + ' \\\\')
        linhas.append(' '.join(regras))
        nomes = [c.split(' · ')[-1] if ' · ' in c else c for c in colunas]
        nomes = [n.replace(' (top 100)', '').replace(' (top 541)', '') for n in nomes]
    else:
        nomes = colunas
    linhas.append(' & '.join(f'\\textbf{{{tex(n)}}}' for n in nomes) + ' \\\\')
    linhas.append('\\midrule')
    for i, (_, row) in enumerate(t.exibicao.iterrows()):
        if i in t.divisores:
            linhas.append('\\midrule')
        linhas.append(' & '.join(tex(v) for v in row) + ' \\\\')
    linhas += ['\\bottomrule', '\\end{tabular}']
    if t.larga:
        linhas.append('}')
    linhas += [f'\\par\\smallskip\\parbox{{\\linewidth}}{{\\scriptsize\\textit{{Nota.}} {tex(t.nota)}}}',
               f'\\end{{{ambiente}}}', '']
    caminho.write_text('\n'.join(linhas), encoding='utf-8', newline='\n')


def figura(t):
    import plotly.graph_objects as go
    import textwrap

    colunas = list(t.exibicao.columns)
    cab_linhas = [textwrap.wrap(c, 16) for c in colunas]
    valores = [['<br>'.join(textwrap.wrap(str(v), 56)) or ' ' for v in t.exibicao[c]] for c in colunas]
    # Largura de cada coluna: a maior linha do valor ou do cabeçalho já quebrado (negrito ocupa um pouco mais).
    pesos = [max(max(len(p) for v in col for p in v.split('<br>')), max(len(p) for p in cab) * 1.1) + 3
             for col, cab in zip(valores, cab_linhas)]
    # ~8 px por caractere em 12 pt, mais o preenchimento das células: evita que o Plotly quebre de novo.
    largura = int(min(max(sum(pesos) * 9 + 40, 560), 2800))
    cabecalho = ['<b>' + '<br>'.join(cab) + '</b>' for cab in cab_linhas]
    n = len(t.exibicao)
    cores = [[FUNDO_ALTERNADO if i % 2 else 'white' for i in range(n)]] * len(colunas)
    alinhar = ['left' if a == 'l' else 'right' for a in t.alinhamento][:len(colunas)]
    linhas_por_celula = max(1, max(v.count('<br>') + 1 for col in valores for v in col))
    altura_linha = 26 + 15 * (linhas_por_celula - 1)
    nota = '<br>'.join(textwrap.wrap('Nota. ' + t.nota, int(largura / 6.4)))
    altura_nota = 16 * (nota.count('<br>') + 1)
    altura_cab = 30 + 14 * max(c.count('<br>') for c in cabecalho)
    altura = 50 + altura_cab + altura_linha * n + altura_nota + 30
    fig = go.Figure(go.Table(
        columnwidth=pesos,
        header=dict(values=cabecalho, fill_color=FUNDO_CABECALHO, line_color=FUNDO_CABECALHO,
                    font=dict(color='white', size=12, family=FONTE), align=alinhar, height=altura_cab),
        cells=dict(values=valores, fill_color=cores, line_color=LINHA, align=alinhar,
                   font=dict(color=TINTA, size=12, family=FONTE), height=altura_linha)))
    fig.update_layout(
        width=largura, height=altura, margin=dict(l=20, r=20, t=40, b=altura_nota + 24),
        paper_bgcolor='white', font=dict(family=FONTE, color=TINTA),
        title=dict(text=f'<b>{t.nome.split("_")[0]}</b> · {t.titulo}', x=0.012, y=0.985, yanchor='top',
                   font=dict(size=15, color=TINTA)),
        annotations=[dict(text=nota, x=0, y=0, xref='paper', yref='paper', yanchor='top', yshift=-10,
                          xanchor='left', align='left', showarrow=False, font=dict(size=10.5, color=TINTA_SUAVE))])
    return fig


def exportar(figuras, bases):
    """Exporta PNG (3x) e PDF em lote: um único navegador para todas as imagens."""
    import plotly.io as pio
    # Em lote, o tamanho do layout não é respeitado: passa largura e altura de cada figura.
    tamanho = dict(width=[f.layout.width for f in figuras], height=[f.layout.height for f in figuras])
    pio.write_images(figuras, [b.with_suffix('.png') for b in bases], scale=3, **tamanho)
    pio.write_images(figuras, [b.with_suffix('.pdf') for b in bases], **tamanho)


GERADORES = (t1, t2, t3, t4, t5, t6)


def gerar(destino, imagens=True):
    destino.mkdir(parents=True, exist_ok=True)
    indice = ['% Todas as tabelas do artigo. Requer \\usepackage{booktabs} e \\usepackage{graphicx}.']
    figuras, bases = [], []
    for gerador in GERADORES:
        t = gerador()
        t.dados.to_csv(destino / f'{t.nome}.csv', index=False, lineterminator='\n')
        escrever_tex(t, destino / f'{t.nome}.tex')
        if imagens:
            figuras.append(figura(t))
            bases.append(destino / t.nome)
        indice.append(f'\\input{{{t.nome}.tex}}')
        print(f'{t.nome}: {len(t.exibicao)} linhas', flush=True)
    (destino / 'tabelas.tex').write_text('\n'.join(indice) + '\n', encoding='utf-8', newline='\n')
    if imagens:
        exportar(figuras, bases)
        print(f'{len(figuras)} imagens em PNG e PDF', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destino', type=Path, default=ENTREGA / 'tabelas')
    parser.add_argument('--sem-imagens', action='store_true', help='só CSV e LaTeX (sem navegador para o kaleido)')
    args = parser.parse_args()
    gerar(args.destino, imagens=not args.sem_imagens)
