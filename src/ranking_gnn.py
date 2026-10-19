"""Exporta embeddings do checkpoint no contexto fixo e prioriza com IsolationForest."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from baseline import ajustar_ranking
from features_contexto import alinhar
from modelo_gnn import MODELOS
from preparar_gnn import FORWARD
from treinar_gnn import validar_divisao


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def salvar_ranking(users, x, output, provenance, seed=42):
    """Empates por wallet_id; score maior prioriza, sem atribuir rotulo."""
    if output.exists():
        raise FileExistsError(output)
    if x.ndim != 2 or len(x) != len(users) or not np.isfinite(x).all():
        raise ValueError('Representacoes invalidas ou desalinhadas')
    if users.wallet_id.duplicated().any() or users.wallet_id.isna().any():
        raise ValueError('IDs invalidos')
    detector, ranking, times = ajustar_ranking(users, x, seed=seed)
    output.mkdir(parents=True)
    ranking.to_csv(output / 'ranking.csv', index=False)
    ranking.loc[ranking.selecionada_top].to_csv(output / 'top.csv', index=False)
    joblib.dump(detector, output / 'isolation_forest.joblib')
    manifest = {**provenance, 'if_seed': seed, 'if_parametros': detector.get_params(),
                'n_carteiras': len(users), 'dimensoes': x.shape[1], 'tempos': times,
                'top_fraction': .01, 'desempate': 'wallet_id crescente; ceil(n * fracao)',
                'score': '-score_samples; nao e probabilidade de fraude',
                'ajuste': 'IsolationForest ajustado e aplicado nos mesmos nos; exploracao transdutiva',
                'sem_contexto_no_top': int(ranking.loc[ranking.selecionada_top, 'sem_contexto'].sum()),
                'teste_avaliado': False}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return ranking


def exportar(source, checkpoint, users_path, output, if_seed=42):
    """Confere origem e IDs antes de salvar embeddings e ranking de carteiras."""
    if output.exists():
        raise FileExistsError(output)
    split = torch.load(source, map_location='cpu', weights_only=False)
    validar_divisao(split)
    graph = split['context']
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if saved.get('sha256_divisao') != sha256(source):
        raise ValueError('Checkpoint pertence a outra divisao')
    model = MODELOS[saved['model_kind']](**saved['model_config'])
    model.load_state_dict(saved['model_state'])
    model.eval()
    with torch.no_grad():
        z = model(graph.x_dict, graph.edge_index_dict)
    if any(not torch.isfinite(value).all() for value in z.values()):
        raise ValueError('Embeddings nao finitos')
    users = alinhar(pd.read_csv(users_path), 'wallet_id', graph['user'].original_id)
    degree = torch.bincount(graph[FORWARD].edge_index[0], minlength=len(users)).numpy()
    users = users[['wallet_id', 'wallet']].copy()
    users['grau_contexto'] = degree
    users['sem_contexto'] = degree == 0
    provenance = {'metodo': saved['model_kind'] + ' + IsolationForest',
                  'checkpoint': str(checkpoint.resolve()), 'sha256_checkpoint': sha256(checkpoint),
                  'sha256_divisao': sha256(source), 'sha256_users': sha256(users_path),
                  'encoder_seed': saved['seed'], 'melhor_epoca': saved['epoch'],
                  'embeddings': 'contexto fixo; sem reajuste sobre grafo completo'}
    ranking = salvar_ranking(users, z['user'].numpy(), output, provenance, if_seed)
    np.savez_compressed(output / 'embeddings.npz', user=z['user'].numpy(), event=z['event'].numpy(),
                        user_original_id=graph['user'].original_id.numpy(),
                        event_original_id=graph['event'].original_id.numpy())
    mappings = {kind: {str(original): idx for idx, original in enumerate(graph[kind].original_id.tolist())}
                for kind in ['user', 'event']}
    (output / 'indices_originais.json').write_text(json.dumps(mappings), encoding='utf-8')
    return ranking


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--users', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--if-seed', type=int, default=42)
    args = parser.parse_args()
    exportar(args.input, args.checkpoint, args.users, args.output, args.if_seed)
