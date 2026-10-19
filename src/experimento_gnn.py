"""Executa protocolo fixo: GNN/MLP, duas condicoes, tres seeds e rankings."""
import argparse
from datetime import datetime, timezone
from itertools import combinations
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from features_contexto import alinhar, features_contexto
from preparar_features import ROOT
from preparar_gnn import FORWARD
from ranking_gnn import exportar, salvar_ranking, sha256
from treinar_gnn import treinar, validar_divisao


def comparar_rankings(left, right):
    """Correlacao de postos medios; top-k usa desempate deterministico por ID."""
    if set(left.wallet_id) != set(right.wallet_id):
        raise ValueError('Rankings com universos diferentes')
    a = left.set_index('wallet_id').score_anomalia.sort_index()
    b = right.set_index('wallet_id').score_anomalia.reindex(a.index)
    correlation = (a.rank(method='average').corr(b.rank(method='average'))
                   if a.nunique() > 1 and b.nunique() > 1 else np.nan)
    result = {'spearman': float(correlation) if np.isfinite(correlation) else None}
    for fraction in [.005, .01, .02]:
        k = math.ceil(len(a) * fraction)
        x, y = set(left.head(k).wallet_id), set(right.head(k).wallet_id)
        result[f'k_{fraction}'] = k
        result[f'intersecao_{fraction}'] = len(x & y)
        result[f'jaccard_{fraction}'] = len(x & y) / len(x | y)
    return result


def executar(directory, source, output, seeds=(42, 43, 44), epochs=100):
    if output.exists():
        raise FileExistsError(output)
    split = torch.load(source, map_location='cpu', weights_only=False)
    validar_divisao(split)
    users = pd.read_csv(directory / 'user.csv')
    events = pd.read_csv(directory / 'event.csv')
    edges = pd.read_csv(directory / 'edge.csv')
    controlled, raw, bundle, feature_report = features_contexto(split, users, events, edges)
    output.mkdir(parents=True)
    protocol = {'source': str(source.resolve()), 'sha256_source': sha256(source),
                'sha256_inputs': {name: sha256(directory / name) for name in ['user.csv', 'event.csv', 'edge.csv']},
                'sha256_code': {p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')},
                'model_seeds': list(seeds), 'eval_seed': 43, 'negative_seed': 42, 'if_seed': 42,
                'models': ['gnn', 'mlp'], 'conditions': ['full_day', 'context_only'],
                'epochs': epochs, 'patience': 15, 'lr': .001, 'hidden': 64, 'embedding': 32,
                'dropout': .1, 'criterio': 'menor BCE de validacao; min_delta=0.0001; inclui epoca zero',
                'split': 'mesmos pares do arquivo de origem; nenhuma nova divisao',
                'context_only': feature_report, 'teste_avaliado': False,
                'nota': 'capacidade difere: GNN tem projecoes de vizinhos; quantidade registrada por modelo'}
    (output / 'protocolo.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    joblib.dump(bundle, output / 'preprocessador_contexto.joblib')
    raw.to_csv(output / 'features_contexto.csv', index=False)
    rows, rankings, reference_eval, reference_test = [], {}, None, None
    degree_report = None
    for condition, current in [('full_day', split), ('context_only', controlled)]:
        folder = output / condition
        folder.mkdir()
        division = folder / 'divisao.pt'
        torch.save(current, division)
        graph = current['context']
        aligned = alinhar(users, 'wallet_id', graph['user'].original_id)[['wallet_id', 'wallet']].copy()
        degree = torch.bincount(graph[FORWARD].edge_index[0], minlength=len(aligned)).numpy()
        aligned['grau_contexto'], aligned['sem_contexto'] = degree, degree == 0
        rankings[condition + '/features'] = salvar_ranking(
            aligned, graph['user'].x.numpy(), folder / 'baseline_features',
            {'metodo': 'features + IsolationForest', 'condition': condition, 'sha256_divisao': sha256(division)})
        for kind in ['gnn', 'mlp']:
            for seed in seeds:
                run = folder / f'{kind}_seed_{seed}'
                print(f'\nCONDICAO={condition} MODELO={kind} SEED={seed}', flush=True)
                report = treinar(division, run, seed=seed, epochs=epochs,
                                 model_kind=kind, eval_seed=43, negative_seed=42, feature_condition=condition)
                evaluation = torch.load(run / 'validacao.pt', weights_only=True)
                test = torch.load(run / 'teste_reservado.pt', weights_only=True)
                if reference_eval is None:
                    reference_eval, reference_test = evaluation, test
                    endpoints = graph[FORWARD].edge_index
                    du = torch.bincount(endpoints[0], minlength=graph['user'].num_nodes)
                    de = torch.bincount(endpoints[1], minlength=graph['event'].num_nodes)
                    pairs, labels = evaluation['pairs'], evaluation['labels'].numpy()
                    scores = (du[pairs[0]] * de[pairs[1]]).numpy()
                    degree_report = {'metodo': 'produto dos graus de contexto',
                                     'ap': float(average_precision_score(labels, scores)),
                                     'auc': float(roc_auc_score(labels, scores)),
                                     'nota': 'sem BCE: score nao calibrado como logit', 'teste_avaliado': False}
                for key in ['pairs', 'labels']:
                    if not torch.equal(reference_eval[key], evaluation[key]) or not torch.equal(reference_test[key], test[key]):
                        raise RuntimeError('Exemplos de avaliacao mudaram entre execucoes')
                metrics = report['melhor_validacao']
                rows.append({'condition': condition, 'model': kind, 'seed': seed,
                             'initial_loss': report['validacao_inicial']['loss'],
                             'val_loss': metrics['loss'], 'val_ap': metrics['ap'], 'val_auc': metrics['auc'],
                             'best_epoch': report['melhor_epoca'], 'epochs': report['epochs_executadas'],
                             'parameters': report['n_parametros'], 'seconds': report['tempo_loop_segundos'],
                             'ap_both_context': metrics['ambos_com_vizinhos'].get('ap'),
                             'ap_missing_context': metrics['algum_sem_vizinhos'].get('ap')})
                rankings[f'{condition}/{kind}_{seed}'] = exportar(
                    division, run / 'melhor_modelo.pt', directory / 'user.csv', run / 'ranking')
                pd.DataFrame(rows).to_csv(output / 'resultados.csv', index=False)
    comparison = []
    for (name_a, a), (name_b, b) in combinations(rankings.items(), 2):
        if name_a.split('/')[0] != name_b.split('/')[0]:
            continue
        comparison.append({'a': name_a, 'b': name_b, **comparar_rankings(a, b)})
    pd.DataFrame(comparison).to_csv(output / 'comparacoes_rankings.csv', index=False)
    results = pd.DataFrame(rows)
    results.groupby(['condition', 'model'])[['val_loss', 'val_ap', 'val_auc', 'seconds']].agg(['mean', 'std']).to_csv(output / 'resumo_seeds.csv')
    (output / 'controle_grau.json').write_text(json.dumps(degree_report, indent=2), encoding='utf-8')
    integrity = {'execucoes': len(rows), 'rankings': len(rankings), 'avaliacao_identica': True,
                 'teste_avaliado': False, 'n_carteiras': len(users),
                 'sha256_pares_validacao': report['sha256_pares_validacao']}
    (output / 'integridade.json').write_text(json.dumps(integrity, indent=2), encoding='utf-8')
    print(results.to_string(index=False), flush=True)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'data/processed/dev')
    parser.add_argument('--split', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    executar(args.input, args.split or args.input / 'gnn_seed_42/divisao.pt',
             args.output or args.input / f'experimento_{stamp}', seeds=tuple(args.seeds), epochs=args.epochs)
