"""Orquestra as APIs existentes em duas copias context_only."""
import argparse
from pathlib import Path

from .comum import (legacy_imports, load_run, read_json, require, seal_tree, sha,
                    verify_inventory, write_json)


def fit_scenario(folder, p):
    """Nao recebe rotulos, mercados-alvo ou manifesto de intervencao."""
    import joblib
    import numpy as np
    import pandas as pd
    import torch
    legacy_imports()
    from features_contexto import alinhar, features_contexto
    from preparar_gnn import FORWARD
    from ranking_gnn import exportar, salvar_ranking
    from treinar_gnn import treinar

    folder = Path(folder)
    output = folder / 'resultados'
    output.mkdir(exist_ok=False)
    entry = folder / 'entrada'
    split = torch.load(entry / 'divisao.pt', map_location='cpu', weights_only=False)
    users, events, edges = [pd.read_csv(entry / name) for name in ('user.csv', 'event.csv', 'edge.csv')]
    controlled, raw, bundle, report = features_contexto(split, users, events, edges)
    division = output / 'divisao.pt'
    torch.save(controlled, division)
    raw.to_csv(output / 'features_contexto.csv', index=False)
    joblib.dump(bundle, output / 'preprocessador_contexto.joblib')
    write_json(output / 'features_manifest.json', report)
    graph = controlled['context']
    aligned = alinhar(users, 'wallet_id', graph['user'].original_id)[['wallet_id', 'wallet']].copy()
    degree = torch.bincount(graph[FORWARD].edge_index[0], minlength=len(aligned)).numpy()
    aligned['grau_contexto'], aligned['sem_contexto'] = degree, degree == 0
    salvar_ranking(aligned, graph['user'].x.numpy(), output / 'features',
                   {'metodo': 'features + IsolationForest', 'condition': 'context_only',
                    'sha256_divisao': sha(division)}, seed=p['if_seed'])
    reference = None
    rows = []
    for model in ('gnn', 'mlp'):
        for seed in p['model_seeds']:
            run = output / f'{model}_seed_{seed}'
            training = treinar(division, run, seed=seed, model_kind=model,
                               feature_condition='context_only', **p['training'])
            evaluation = {name: torch.load(run / filename, map_location='cpu', weights_only=True)
                          for name, filename in [('val', 'validacao.pt'), ('test', 'teste_reservado.pt')]}
            if reference is None:
                reference = evaluation
            for name in ('val', 'test'):
                for key in ('pairs', 'labels'):
                    require(torch.equal(reference[name][key], evaluation[name][key]), 'Avaliacao mudou entre modelos/seeds')
            require(not training['teste_avaliado'], 'Teste nao pode ser avaliado')
            exportar(division, run / 'melhor_modelo.pt', entry / 'user.csv', run / 'ranking', if_seed=p['if_seed'])
            rows.append({'model': model, 'seed': seed, 'best_epoch': training['melhor_epoca'],
                         'seconds': training['tempo_loop_segundos'],
                         **{f'val_{key}': training['melhor_validacao'][key] for key in ('loss', 'ap', 'auc')}})
    pd.DataFrame(rows).to_csv(output / 'metricas_ligacao.csv', index=False)
    # Dados de teste sao comparados para integridade, nunca passados a um modelo.
    write_json(output / 'integridade.json', {'avaliacao_identica_entre_modelos': True,
                                            'n_treinos': len(rows), 'n_rankings': len(rows) + 1,
                                            'teste_avaliado': False})
    return graph['event'].x.numpy(), reference


def execute(run):
    import numpy as np
    import pandas as pd
    import torch
    legacy_imports()
    from treinar_gnn import codigos
    run = Path(run).resolve()
    p, manifest = load_run(run)
    require(not any((run / scenario / 'resultados').exists() for scenario in ('original', 'injetado')),
            'Resultados ja existem; nao sobrescrever nem retomar parcialmente')
    inv = read_json(run / 'inventario_hashes.json')
    status = {'status': 'executando', 'teste_avaliado': False}
    try:
        original_event, original_eval = fit_scenario(run / 'original', p)
        changed_event, changed_eval = fit_scenario(run / 'injetado', p)
        require(np.array_equal(original_event, changed_event), 'Features de mercado mudaram')
        # Conferir tambem os transformadores, nao apenas as matrizes resultantes.
        import joblib
        a, b = [joblib.load(run / s / 'resultados/preprocessador_contexto.joblib') for s in ('original', 'injetado')]
        require(a['mediana_duracao'] == b['mediana_duracao'], 'Mediana de mercado alterada')
        for attr in ('mean_', 'scale_', 'var_'):
            require(np.array_equal(getattr(a['event_scaler'], attr), getattr(b['event_scaler'], attr)), 'Scaler de mercado alterado')
        require(all(np.array_equal(x, y) for x, y in zip(a['categorias'].categories_, b['categorias'].categories_)), 'Categorias alteradas')
        ne = len(original_event)
        negative_checks = {}
        for name in ('val', 'test'):
            ea, eb = original_eval[name], changed_eval[name]
            na = codigos(ea['pairs'][:, ea['labels'] == 0], ne)
            nb = codigos(eb['pairs'][:, eb['labels'] == 0], ne)
            require(torch.equal(ea['pairs'][:, ea['labels'] == 1], eb['pairs'][:, eb['labels'] == 1]), 'Positivos reservados mudaram')
            negative_checks[name] = {'negative_original': len(na), 'negative_injected': len(nb),
                                     'intersection': len(na & nb), 'equal': na == nb}
        raw_a, raw_b = [pd.read_csv(run / s / 'resultados/features_contexto.csv').set_index('wallet_id') for s in ('original', 'injetado')]
        require(raw_a.index.equals(raw_b.index), 'IDs das features divergentes')
        changes = raw_a.add_suffix('_original').join(raw_b.add_suffix('_injetado'))
        for col in raw_a.columns:
            if col != 'sem_contexto':
                changes[col + '_delta'] = raw_b[col] - raw_a[col]
        changes['features_brutas_alteradas'] = (raw_a != raw_b).any(axis=1)
        splits = [torch.load(run / s / 'resultados/divisao.pt', map_location='cpu', weights_only=False)
                  for s in ('original', 'injetado')]
        x_a, x_b = [s['context']['user'].x.numpy() for s in splits]
        changes['features_transformadas_alteradas'] = (x_a != x_b).any(axis=1)
        changes.to_csv(run / 'mudancas_features.csv')
        verify_inventory(inv)
        status.update(status='concluido', markets_equal=True, negative_comparison=negative_checks,
                      n_treinos=4 * len(p['model_seeds']), n_rankings=4 * len(p['model_seeds']) + 2)
        status['artifacts'] = {s: seal_tree(run / s / 'resultados') for s in ('original', 'injetado')}
        status['feature_changes_sha256'] = sha(run / 'mudancas_features.csv')
        write_json(run / 'integridade.json', {'geracao': manifest['diagnostic']['checks'], 'originais_intactos': True,
                                            'treino_concluido': True, 'markets_equal': True, 'teste_avaliado': False})
    except Exception as exc:
        status.update(status='falhou', reason=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        verify_inventory(inv)
        write_json(run / 'execucao.json', status)
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    execute(parser.parse_args().run)
