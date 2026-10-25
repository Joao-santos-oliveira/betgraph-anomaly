"""Avalia alvos designados somente apos os rankings estarem prontos."""
import argparse
from pathlib import Path

from .comum import (load_run, read_json, require, sha, verify_tree, write_json)


def score_ranking(ranking, labels, top_k=(50, 100)):
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score
    require(not ranking.wallet_id.isna().any() and not ranking.wallet_id.duplicated().any(), 'IDs de ranking invalidos')
    require(not labels.wallet_id.isna().any() and not labels.wallet_id.duplicated().any(), 'IDs de rotulos invalidos')
    require(set(ranking.wallet_id) == set(labels.wallet_id), 'Universo do ranking difere dos rotulos')
    require(np.isfinite(ranking.score_anomalia.to_numpy(float)).all(), 'Scores nao finitos')
    ordered = ranking.sort_values(['score_anomalia', 'wallet_id'], ascending=[False, True]).reset_index(drop=True)
    require(ordered.wallet_id.tolist() == ranking.wallet_id.tolist(), 'Ranking fora da ordem esperada')
    require(np.array_equal(ordered.posicao.to_numpy(), np.arange(1, len(ordered) + 1)), 'Posicoes invalidas')
    joined = ordered.merge(labels[['wallet_id', 'alvo', 'doadora', 'grupo', 'elegivel']], on='wallet_id', validate='one_to_one')
    require(set(joined.alvo) == {0, 1}, 'Avaliacao precisa de ambas as classes')
    rows = []
    for universe, subset in [('completo', joined), ('sem_doadoras', joined.loc[~joined.doadora]),
                              ('elegiveis', joined.loc[joined.elegivel])]:
        y = subset.alvo.to_numpy()
        two_classes = len(np.unique(y)) == 2
        row = {'universe': universe, 'n': len(subset), 'positives': int(y.sum()),
               'prevalence': float(y.mean()) if len(y) else None,
               'ap': float(average_precision_score(y, subset.score_anomalia)) if two_classes else None,
               'auc': float(roc_auc_score(y, subset.score_anomalia)) if two_classes else None}
        if universe == 'completo':
            for k in top_k:
                head = joined.head(k)
                recovered = int(head.alvo.sum())
                row[f'recovered_{k}'] = recovered
                row[f'recall_{k}'] = recovered / int(joined.alvo.sum())
                row[f'precision_{k}'] = recovered / len(head)
                row[f'effective_k_{k}'] = len(head)
                cutoff = head.iloc[-1].score_anomalia
                row[f'tied_at_cutoff_{k}'] = int((joined.score_anomalia == cutoff).sum())
        rows.append(row)
    return rows, joined


def evaluate(run):
    import pandas as pd
    run = Path(run).resolve()
    p, injection = load_run(run)
    execution = read_json(run / 'execucao.json')
    require(execution['status'] == 'concluido' and not execution['teste_avaliado'], 'Treino incompleto ou teste avaliado')
    for scenario in ('original', 'injetado'):
        verify_tree(run / scenario / 'resultados', execution['artifacts'][scenario])
    require(sha(run / 'mudancas_features.csv') == execution['feature_changes_sha256'], 'Diagnostico de features alterado')
    output = run / 'avaliacao'
    output.mkdir(exist_ok=False)
    labels = pd.read_csv(run / 'rotulos.csv')
    require(int(labels.alvo.sum()) == p['n_targets'], 'Quantidade de alvos invalida')
    records, positions, groups = [], [], []
    methods = [('features', None, 'features/ranking.csv')]
    methods += [(model, seed, f'{model}_seed_{seed}/ranking/ranking.csv')
                for model in ('gnn', 'mlp') for seed in p['model_seeds']]
    for model, seed, relative in methods:
        pair = {}
        for scenario in ('original', 'injetado'):
            frame = pd.read_csv(run / scenario / 'resultados' / relative)
            scores, joined = score_ranking(frame, labels, p['top_k'])
            for row in scores:
                records.append({'scenario': scenario, 'model': model, 'seed': seed, **row})
            pair[scenario] = joined
        a, b = [pair[s].set_index('wallet_id') for s in ('original', 'injetado')]
        delta = a[['alvo', 'doadora', 'grupo', 'elegivel']].copy()
        delta['posicao_original'], delta['posicao_injetada'] = a.posicao, b.posicao.reindex(a.index)
        delta['mudanca_posicao'] = delta.posicao_original - delta.posicao_injetada
        delta['model'], delta['seed'] = model, seed
        positions.append(delta.reset_index())
        for group, sub in delta.groupby('grupo'):
            groups.append({'model': model, 'seed': seed, 'grupo': group, 'n': len(sub),
                           'delta_medio': float(sub.mudanca_posicao.mean()),
                           'delta_mediano': float(sub.mudanca_posicao.median())})
    metrics = pd.DataFrame(records)
    metrics.to_csv(output / 'metricas.csv', index=False)
    pd.concat(positions, ignore_index=True).to_csv(output / 'posicoes_antes_depois.csv', index=False)
    pd.DataFrame(groups).to_csv(output / 'mudancas_por_grupo.csv', index=False)
    numeric = ['ap', 'auc', 'recovered_50', 'recovered_100', 'recall_50', 'recall_100', 'precision_50', 'precision_100']
    metrics.groupby(['scenario', 'model', 'universe'])[numeric].agg(['count', 'mean', 'std']).to_csv(output / 'resumo_seeds.csv')
    labels.merge(pd.read_csv(run / 'mudancas_features.csv'), on='wallet_id', validate='one_to_one',
                 suffixes=('_rotulo', '_feature')).to_csv(output / 'diagnostico_colaterais.csv', index=False)
    lines = ['# Resultado do piloto sintetico', '',
             'Rotulo = alvo da intervencao artificial; nao representa fraude. Teste de ligacoes nao avaliado.', '',
             f"Seeds de selecao/trocas: {p['selection_seed']}/{p['swap_seed']}. "
             f"Alvos: {p['n_targets']}; trocas: {injection['diagnostic']['accepted']}.", '',
             '## Recuperacao no universo completo', '',
             '| Cenario | Metodo | Seed | AP | AUC | Recuperados@50 | Recuperados@100 |',
             '| --- | --- | --- | --- | --- | --- | --- |']
    for row in metrics.loc[metrics.universe == 'completo'].itertuples():
        seed = '-' if pd.isna(row.seed) else str(int(row.seed))
        lines.append(f'| {row.scenario} | {row.model} | {seed} | {row.ap:.6f} | {row.auc:.6f} | {int(row.recovered_50)} | {int(row.recovered_100)} |')
    lines += ['', '## Limites', '',
              '- Os doadores tambem foram alterados. Consulte os universos complementares e o diagnostico de features.',
              '- Grupo intacta significa endpoints intactos: imputacao e reajuste dos scalers podem alterar features de outros nos; ambas as mudancas sao registradas.',
              '- Preservar graus nao preserva os agregados individuais. O efeito nao e exclusivamente topologico.',
              '- Os negativos podem diferir antes/depois; hashes e intersecoes estao em execucao.json e nos manifestos de treino.',
              '- Mudanca de posicao inclui retreino. Seeds do encoder nao sao replicas independentes de dados.',
              '- Carteiras isoladas e elegiveis estao identificadas. Ausencia de recuperacao tambem e resultado valido.',
              '- Este piloto de desenvolvimento nao substitui realizacoes sinteticas reservadas ou revisao de casos reais.', '']
    (output / 'relatorio.md').write_text('\n'.join(lines), encoding='utf-8')
    # Reconfere arquivos protegidos e entradas apos a avaliacao.
    load_run(run)
    write_json(output / 'integridade.json', {'rotulos_somente_na_avaliacao': True,
                                           'n_rankings_avaliados': len(methods) * 2,
                                           'universos_alinhados': True, 'teste_avaliado': False})
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    evaluate(parser.parse_args().run)
