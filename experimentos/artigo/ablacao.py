"""Avaliação pareada dos três scores sobre os mesmos candidatos e embeddings."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experimentos.gnn.detectores import hibrido, ranking
from experimentos.artigo.preparar import FORWARD, protocol
from experimentos.relacional import scores_relacionais


def classificar(folder):
    graph = torch.load(folder / 'rede.pt', weights_only=False)['context']
    users = graph['user'].original_id.numpy()
    events = graph['event'].original_id.numpy()
    pairs = graph[FORWARD].edge_index.numpy()
    dims = (len(users), len(events))
    scores = scores_relacionais(pairs, *dims)['surpresa_coparticipacao']
    ordenado = ranking(users, scores)
    pd.testing.assert_frame_equal(ordenado, pd.read_csv(folder / 'relacional.csv'))
    ranks = {('Relacional', 0): ordenado.set_index('wallet_id').posicao}
    for seed in protocol['seeds_modelo']:
        treino = folder / f'gnn_{seed}'
        if not treino.exists():
            treino = folder / f'v2_{seed}'  # Rodadas anteriores à adoção do nome principal.
        with np.load(treino / 'embeddings.npz') as emb:
            np.testing.assert_array_equal(emb['user_original_id'], users)
            np.testing.assert_array_equal(emb['event_original_id'], events)
            z = emb['user']
        hybrid = ranking(users, hibrido(pairs, *dims, z))
        pd.testing.assert_frame_equal(hybrid, pd.read_csv(treino / 'ranking.csv'))
        ranks['GNN híbrida', seed] = hybrid.set_index('wallet_id').posicao
        sim = ranking(users, hibrido(pairs, *dims, z, apenas_similaridade=True))
        ranks['Similaridade da GNN', seed] = sim.set_index('wallet_id').posicao
    return ranks


def avaliar(rodada, destino):
    destino.mkdir(parents=True, exist_ok=True)
    original = classificar(rodada / 'original')
    real = pd.DataFrame(original).sort_index()
    real.columns = [f'{method}_{seed}' for method, seed in real.columns]
    real.to_csv(destino / 'posicoes_dia_real.csv')
    real.nsmallest(100, 'GNN híbrida_42').to_csv(destino / 'top100_dia_real.csv')
    counts = []
    for realization in protocol['realizacoes']:
        folder = rodada / str(realization)
        labels = pd.read_csv(folder / 'rotulos.csv').set_index('wallet_id')
        assert labels.index.is_unique
        assert labels.groupby('papel').size().to_dict() == {'alvo': 50, 'controle': 50}
        after = classificar(folder)
        for (method, seed), rank in after.items():
            for phase, series in [('antes', original[method, seed]), ('depois', rank)]:
                frame = labels.assign(posicao=series.reindex(labels.index))
                assert frame.posicao.notna().all()
                for (dose, role), group in frame.groupby(['dose', 'papel']):
                    counts.append(dict(realizacao=realization, metodo=method, seed=seed,
                                       fase=phase, mercados=dose, papel=role, n=len(group),
                                       top100=int(group.posicao.le(100).sum()),
                                       top541=int(group.posicao.le(541).sum())))
    rows = pd.DataFrame(counts)
    rows.to_csv(destino / 'por_realizacao_seed.csv', index=False)
    keys = ['metodo', 'fase', 'mercados', 'papel']
    measures = ['n', 'top100', 'top541']
    per_seed = rows.groupby(keys + ['seed'])[measures].sum()
    summary = per_seed.groupby(keys)[measures].mean().reset_index()
    assert summary.n.eq(50).all()
    summary.to_csv(destino / 'resumo.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rodada', type=Path)
    parser.add_argument('destino', type=Path)
    args = parser.parse_args()
    avaliar(args.rodada, args.destino)
