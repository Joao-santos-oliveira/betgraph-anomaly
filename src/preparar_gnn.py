"""Separa pares unicos para propagacao, treino e avaliacao da GNN."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch_geometric.data import HeteroData

from preparar_features import ROOT

FORWARD = ('user', 'bets_on', 'event')
REVERSE = ('event', 'rev_bets_on', 'user')


def separar(data, seed=42, val_fraction=.1, test_fraction=.1, target_fraction=.25):
    """Retorna grafo de contexto fixo e pares-alvo disjuntos por endpoint."""
    if not (0 < val_fraction < 1 and 0 < test_fraction < 1
            and val_fraction + test_fraction < 1 and 0 < target_fraction < 1):
        raise ValueError('Fracoes invalidas')
    data.validate(raise_on_error=True)
    # Uma ligacao por carteira-mercado, independente do outcome.
    pairs = torch.unique(data[FORWARD].edge_index.cpu(), dim=1, sorted=True)
    n = pairs.shape[1]
    nv, nt = int(n * val_fraction), int(n * test_fraction)
    ntrain = n - nv - nt
    nlabels = int(ntrain * target_fraction)
    if min(nv, nt, nlabels, ntrain - nlabels) < 1:
        raise ValueError('Poucos pares para a divisao solicitada')
    generator = torch.Generator().manual_seed(seed)
    shuffled = pairs[:, torch.randperm(n, generator=generator)]
    test = shuffled[:, :nt]
    val = shuffled[:, nt:nt + nv]
    train = shuffled[:, nt + nv:nt + nv + nlabels]
    context = shuffled[:, nt + nv + nlabels:]
    graph = HeteroData()
    for kind in ['user', 'event']:
        graph[kind].x = data[kind].x.cpu().clone()
        if not torch.isfinite(graph[kind].x).all():
            raise ValueError('Features nao finitas')
        graph[kind].original_id = data[kind].original_id.cpu().clone()
    graph[FORWARD].edge_index = context.clone()
    graph[REVERSE].edge_index = context.flip(0).clone()
    graph.validate(raise_on_error=True)
    # all_positive permite excluir TODO par observado da futura amostragem negativa.
    return {'context': graph, 'train_positive': train, 'val_positive': val,
            'test_positive': test, 'all_positive': pairs}


def executar(directory, output, seed=42):
    """Salva a divisao e diagnosticos sem modificar hetero.pt."""
    source = directory / 'hetero.pt'
    data = torch.load(source, map_location='cpu', weights_only=False)
    split = separar(data, seed)
    graph = split['context']
    isolated = {}
    for axis, kind in enumerate(['user', 'event']):
        degree = torch.bincount(graph[FORWARD].edge_index[axis], minlength=graph[kind].num_nodes)
        isolated[kind] = int((degree == 0).sum())
    report = {
        'seed': seed, 'sha256_hetero': hashlib.sha256(source.read_bytes()).hexdigest(),
        'fracao_validacao': .1, 'fracao_teste': .1, 'fracao_alvos_dentro_treino': .25,
        'arestas_originais': data[FORWARD].num_edges,
        'pares_unicos': split['all_positive'].shape[1],
        'pares_contexto': graph[FORWARD].num_edges,
        'alvos_treino': split['train_positive'].shape[1],
        'alvos_validacao': split['val_positive'].shape[1],
        'alvos_teste': split['test_positive'].shape[1],
        'nos_sem_vizinhos_no_contexto': isolated,
        'protocolo': 'contexto fixo para treino, validacao e teste; features do recorte inteiro',
        'limitacao': 'reconstrucao transdutiva, nao previsao temporal; sem edge_attr e sem outcomes',
    }
    output.mkdir(parents=True, exist_ok=True)
    torch.save(split, output / 'divisao.pt')
    (output / 'divisao_manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'data/processed/dev')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    executar(args.input, args.output or args.input / f'gnn_seed_{args.seed}', args.seed)


if __name__ == '__main__':
    main()
