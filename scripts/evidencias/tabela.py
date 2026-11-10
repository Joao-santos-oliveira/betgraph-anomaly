"""Tabela comparativa: recuperação no top 100, AP, decoys e custo.

As métricas vêm da rodada histórica da v3, a única em que os métodos das três
branches foram avaliados sobre as mesmas intervenções (realizações 30101/30201,
coorte de 5.000 carteiras). O arquivo é lido do commit fixado com `git show`,
sem depender do checkout atual. Os custos vêm de execuções na mesma máquina e
na mesma coorte (`custos.csv` produzido por `make comparar ESCOPO=5000`).

Uso:
    python scripts/evidencias/tabela.py --custos-v3 <custos.csv da v3> \
        --custos-v2 <custos.csv da v2> [--saida entrega]
"""
import argparse
import hashlib
import io
import json
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
COMMIT_V3 = 'a67bcb5'
FONTE_V3 = 'resultados/v3/metricas.csv'

PRINCIPAIS = {
    'A': ('Relacional direto', 'main'),
    'E_features': ('Atributos + IsolationForest', 'main'),
    'E_v2': ('Híbrido GNN v2', 'v2'),
    'C': ('GNN v3, score de pares', 'v3'),
}
APENDICE = {
    'C_IF': ('GNN v3 + IsolationForest', 'v3'),
    'D': ('MLP sem mensagens', 'v3'),
    'D_IF': ('MLP + IsolationForest', 'v3'),
    'B': ('Manual', 'v3'),
    'O': ('Canal isolado: lado (outcome)', 'v3'),
    'T': ('Canal isolado: horário', 'v3'),
    'V': ('Canal isolado: valor', 'v3'),
}
# Cenários com alvos (recuperação desejada) e o controle de decoys (indesejada).
CENARIOS = {'S1': 'S1 combinado', 'S3_estrutura': 'S3 estrutura 60%', 'S3_volume': 'S3 volume ×10'}
DECOYS = 'S2'
# Códigos de custo: (branch da execução, nome em custos.csv) -> código da tabela.
CUSTOS = {('v3', 'relacional'): 'A', ('v3', 'floresta'): 'E_features',
          ('v3', 'gnn'): 'C', ('v3', 'mlp'): 'D', ('v2', 'gnn'): 'E_v2'}


def ler_metricas(commit=COMMIT_V3, caminho=FONTE_V3):
    """Lê o CSV de métricas do commit fixado e devolve (tabela, sha256)."""
    bruto = subprocess.check_output(['git', 'show', f'{commit}:{caminho}'], cwd=ROOT)
    return pd.read_csv(io.BytesIO(bruto)), hashlib.sha256(bruto).hexdigest()


def resumir(metricas, codigos):
    """Média, mínimo e máximo por método e cenário, só após a injeção e no universo completo."""
    base = metricas[(metricas.fase == 'injetado') & (metricas.universo == 'completo')
                    & metricas.metodo.isin(codigos)]
    if base.empty:
        raise ValueError('Nenhuma linha injetada/completo para os métodos pedidos')
    grupos = base.groupby(['metodo', 'cenario'])
    resumo = grupos.agg(top100_media=('top100', 'mean'), top100_min=('top100', 'min'),
                        top100_max=('top100', 'max'), ap_media=('AP', 'mean'),
                        ap_min=('AP', 'min'), ap_max=('AP', 'max'),
                        execucoes=('top100', 'size'), denominador=('alvos', 'first'),
                        carteiras=('n', 'first'))
    realizacoes = grupos.realizacao.agg(lambda s: '/'.join(map(str, sorted(s.unique()))))
    return resumo.assign(realizacoes=realizacoes).reset_index()


def ler_custos(caminhos):
    """Combina os custos.csv das execuções em uma linha por código de método."""
    linhas = []
    for branch, caminho in caminhos.items():
        if caminho is None:
            continue
        frame = pd.read_csv(caminho)
        for (metodo, grupo) in frame.groupby('metodo'):
            codigo = CUSTOS.get((branch, metodo))
            if codigo is None:
                continue
            linhas.append({'metodo': codigo, 'branch_execucao': branch,
                           'segundos_media': grupo.segundos.mean(),
                           'segundos_min': grupo.segundos.min(), 'segundos_max': grupo.segundos.max(),
                           'pico_rss_mib_max': grupo.pico_rss_mib.max(),
                           'seeds': '/'.join(map(str, sorted(grupo.seed))),
                           'parametros': grupo.parametros.iloc[0] if 'parametros' in grupo else None,
                           'epocas': grupo.epocas.iloc[0] if 'epocas' in grupo else None,
                           'fonte': str(caminho), 'sha256_fonte': hashlib.sha256(Path(caminho).read_bytes()).hexdigest()})
    return pd.DataFrame(linhas)


def ler_benchmark(caminho):
    """Custos do benchmark por etapa (benchmark/resumo.csv), no mesmo formato de ler_custos.

    Usa o tempo interno do método (segundos_metodo, sem a carga do interpretador), mediano
    entre as repetições; nas redes, a média das medianas das três seeds. O pico de RAM é o
    ru_maxrss do processo da etapa.
    """
    frame = pd.read_csv(caminho)
    linhas = []
    for (branch, etapa), grupo in frame.dropna(subset=['seed']).groupby(['branch', 'etapa']):
        codigo = CUSTOS.get((branch, etapa))
        if codigo is None:
            continue
        linhas.append({'metodo': codigo, 'branch_execucao': branch,
                       'segundos_media': grupo.segundos_metodo_median.mean(),
                       'segundos_min': grupo.segundos_metodo_median.min(),
                       'segundos_max': grupo.segundos_metodo_median.max(),
                       'pico_rss_mib_max': grupo.pico_rss_mib_median.max(),
                       'seeds': '/'.join(str(int(s)) for s in sorted(grupo.seed)),
                       'repeticoes': int(grupo.repeticoes.min()),
                       'parametros': None, 'epocas': None, 'fonte': str(caminho),
                       'sha256_fonte': hashlib.sha256(Path(caminho).read_bytes()).hexdigest()})
    return pd.DataFrame(linhas)


def montar(resumo, custos, nomes):
    """Uma linha por método, com colunas largas por cenário."""
    linhas = []
    for codigo, (nome, branch) in nomes.items():
        linha = {'codigo': codigo, 'metodo': nome, 'branch': branch}
        for cenario, rotulo in {**CENARIOS, DECOYS: 'S2 decoys'}.items():
            r = resumo[(resumo.metodo == codigo) & (resumo.cenario == cenario)]
            if r.empty:
                raise ValueError(f'Sem métricas para {codigo} em {cenario}')
            r = r.iloc[0]
            prefixo = 'decoys_S2' if cenario == DECOYS else f'rec100_{cenario}'
            linha[f'{prefixo}_media'] = r.top100_media
            linha[f'{prefixo}_min'] = r.top100_min
            linha[f'{prefixo}_max'] = r.top100_max
            if cenario != DECOYS:
                linha[f'ap_{cenario}_media'] = r.ap_media
            linha[f'execucoes_{cenario}'] = int(r.execucoes)
        c = custos[custos.metodo == codigo] if not custos.empty else custos
        if not c.empty:
            c = c.iloc[0]
            linha.update(segundos_media=c.segundos_media, segundos_min=c.segundos_min,
                         segundos_max=c.segundos_max, pico_rss_mib=c.pico_rss_mib_max,
                         seeds_custo=c.seeds, parametros=c.parametros, epocas=c.epocas)
        linhas.append(linha)
    return pd.DataFrame(linhas)


def _faixa(media, minimo, maximo, casas=2):
    texto = f'{media:.{casas}f}'.replace('.', ',')
    if minimo != maximo:
        texto += f' ({minimo:g}–{maximo:g})'.replace('.', ',')
    return texto


def _ap(valor):
    return f'{valor:.4f}'.replace('.', ',')


def _custo(linha):
    if pd.isna(linha.get('segundos_media')):
        return '—'
    texto = f"{linha['segundos_media']:.2f} s".replace('.', ',')
    if linha['segundos_min'] != linha['segundos_max']:
        texto += f" ({linha['segundos_min']:.1f}–{linha['segundos_max']:.1f})".replace('.', ',')
    return texto


def markdown(tabela, titulo):
    cab = ['Método', 'Branch'] + [f'Rec@100 {r}' for r in CENARIOS.values()] \
        + [f'AP {r}' for r in CENARIOS.values()] + ['Decoys S2 ↓', 'Tempo por execução', 'Pico de RAM']
    linhas = [f'### {titulo}', '', '| ' + ' | '.join(cab) + ' |',
              '|' + '---|' * 2 + '---:|' * (len(cab) - 2)]
    for _, l in tabela.iterrows():
        celulas = [f"{l.metodo} ({l.codigo})", l.branch]
        celulas += [_faixa(l[f'rec100_{c}_media'], l[f'rec100_{c}_min'], l[f'rec100_{c}_max']) for c in CENARIOS]
        celulas += [_ap(l[f'ap_{c}_media']) for c in CENARIOS]
        celulas.append(_faixa(l.decoys_S2_media, l.decoys_S2_min, l.decoys_S2_max))
        celulas.append(_custo(l))
        pico = l.get('pico_rss_mib')
        celulas.append('—' if pd.isna(pico) else f'{pico:.0f} MiB')
        linhas.append('| ' + ' | '.join(celulas) + ' |')
    return '\n'.join(linhas)


def notas(fontes):
    return '\n'.join([
        '',
        '**Como ler.** Rec@100 é o número de alvos artificiais, de 50, nas 100 primeiras posições '
        'de 5.000 carteiras (prevalência de 1%). Decoys S2 é o número de decoys, de 50, no top 100; '
        'quanto menor, melhor. Os valores são médias sobre as realizações 30101 e 30201 e, nas redes, '
        'sobre as seeds 42, 73 e 2026; entre parênteses, o mínimo e o máximo quando diferem. '
        'AP é a precisão média, na mesma agregação.',
        '',
        '**Custo.** Tempo e pico de RAM de uma execução do método na coorte de 5.000 carteiras, na '
        'mesma máquina, sem a preparação compartilhada dos dados. Nas redes, média entre as três seeds.',
        '',
        f"**Fontes.** Métricas: `{fontes['metricas']['caminho']}` no commit `{fontes['metricas']['commit']}` "
        f"(SHA-256 `{fontes['metricas']['sha256'][:12]}…`). Custos: ver `fontes.json`.",
    ])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--custos-v3', type=Path, help='custos.csv de make comparar ESCOPO=5000 na v3')
    parser.add_argument('--custos-v2', type=Path, help='custos.csv de make comparar ESCOPO=5000 na v2')
    parser.add_argument('--benchmark', type=Path,
                        help='benchmark/resumo.csv de benchmark.py; tem prioridade sobre --custos-v3/--custos-v2')
    parser.add_argument('--saida', type=Path, default=ROOT / 'entrega')
    args = parser.parse_args(argv)

    metricas, sha = ler_metricas()
    custos = (ler_benchmark(args.benchmark) if args.benchmark
              else ler_custos({'v3': args.custos_v3, 'v2': args.custos_v2}))
    resumo = resumir(metricas, list(PRINCIPAIS) + list(APENDICE))
    principal = montar(resumo, custos, PRINCIPAIS)
    apendice = montar(resumo, custos, APENDICE)

    args.saida.mkdir(parents=True, exist_ok=True)
    principal.to_csv(args.saida / 'tabela_comparativa.csv', index=False, lineterminator='\n')
    apendice.to_csv(args.saida / 'tabela_apendice.csv', index=False, lineterminator='\n')
    fontes = {'metricas': {'commit': COMMIT_V3, 'caminho': FONTE_V3, 'sha256': sha,
                           'filtro': 'fase=injetado, universo=completo'},
              'custos': custos.drop(columns=[c for c in ('segundos_media', 'segundos_min', 'segundos_max',
                                                         'pico_rss_mib_max', 'parametros', 'epocas')
                                             if c in custos]).to_dict('records')}
    (args.saida / 'fontes.json').write_text(json.dumps(fontes, indent=2, ensure_ascii=False, default=str) + '\n',
                                            encoding='utf-8')
    texto = ['# Tabela comparativa', '', markdown(principal, 'Métodos do artigo'), notas(fontes), '',
             markdown(apendice, 'Apêndice: variantes, controle e canais isolados da v3'), '']
    (args.saida / 'tabela_comparativa.md').write_text('\n'.join(texto), encoding='utf-8', newline='\n')
    print(f'Tabela gravada em {args.saida}')


if __name__ == '__main__':
    main()
