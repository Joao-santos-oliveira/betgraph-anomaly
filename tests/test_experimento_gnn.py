"""Controles de informacao, independencia de vizinhos e protocolo ponta a ponta."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from experimento_gnn import executar, comparar_rankings
from features_contexto import features_contexto
from modelo_gnn import MLPSemMensagens
from preparar_gnn import FORWARD, REVERSE
from ranking_gnn import exportar


def fixture():
    graph = HeteroData()
    graph['user'].x = torch.zeros(4, 5)
    graph['user'].original_id = torch.tensor([10, 30, 50, 90])
    graph['event'].x = torch.zeros(4, 3)
    graph['event'].original_id = torch.tensor([11, 31, 51, 91])
    context = torch.tensor([[0, 0, 1, 2], [0, 1, 1, 2]])
    graph[FORWARD].edge_index = context
    graph[REVERSE].edge_index = context.flip(0)
    split = {'context': graph, 'train_positive': torch.tensor([[1, 3], [2, 0]]),
             'val_positive': torch.tensor([[3], [1]]), 'test_positive': torch.tensor([[3], [2]])}
    split['all_positive'] = torch.cat([context, split['train_positive'], split['val_positive'], split['test_positive']], dim=1)
    users = pd.DataFrame({'wallet_id': [90, 50, 30, 10], 'wallet': ['d', 'c', 'b', 'a']})
    events = pd.DataFrame({'condition_node_id': [11, 31, 51, 91],
                          'duracao_mercado_segundos': [10., 20., np.nan, 100000.],
                          'duracao_mercado_valida': [True, True, False, True],
                          'categoria_principal': ['A', 'A', 'B', 'UNSEEN']})
    pairs = split['all_positive']
    edges = pd.DataFrame({'wallet_id': graph['user'].original_id[pairs[0]],
                          'condition_node_id': graph['event'].original_id[pairs[1]],
                          'volume_usd': np.arange(1, 9, dtype=float), 'n_trades': np.ones(8),
                          'preco_medio': np.full(8, .5)})
    # Dois outcomes do mesmo par precisam contribuir juntos para a agregacao.
    edges = pd.concat([edges, edges.iloc[[0]]], ignore_index=True)
    return split, users, events, edges


class ExperimentoTest(unittest.TestCase):
    def test_features_nao_usam_arestas_ocultas(self):
        split, users, events, edges = fixture()
        a, raw, bundle, report = features_contexto(split, users, events, edges)
        altered = edges.copy()
        altered.loc[4:7, ['volume_usd', 'n_trades', 'preco_medio']] = 999999.
        b, _, _, _ = features_contexto(split, users, events, altered)
        for kind in ['user', 'event']:
            self.assertTrue(torch.equal(a['context'][kind].x, b['context'][kind].x))
        self.assertEqual(raw.wallet_id.tolist(), [10, 30, 50, 90])
        self.assertEqual(raw.iloc[0].volume_total_usd, 4.)
        self.assertEqual(raw.iloc[0].n_trades_total, 3.)
        self.assertEqual(raw.iloc[0].n_mercados_distintos, 2)
        self.assertTrue(raw.iloc[-1].sem_contexto)
        self.assertEqual(raw.iloc[-1].volume_total_usd, 0.)
        self.assertEqual(bundle['mediana_duracao'], 15.)
        self.assertEqual(bundle['user_scaler'].n_samples_seen_, 3)
        self.assertEqual(report['carteiras_sem_contexto'], 1)
        self.assertEqual(split['context']['user'].x.shape[1], 5)
        # Metadados de um mercado isolado nao alteram o ajuste dos ativos.
        events.loc[3, 'duracao_mercado_segundos'] = 1.
        c, _, changed, _ = features_contexto(split, users, events, edges)
        np.testing.assert_array_equal(changed['event_scaler'].mean_, bundle['event_scaler'].mean_)
        self.assertTrue(torch.equal(a['context']['event'].x[:3], c['context']['event'].x[:3]))

    def test_mlp_independente_das_arestas(self):
        split, *_ = fixture()
        graph = split['context']
        model = MLPSemMensagens(5, 3)
        model.eval()
        a, b = model(graph.x_dict, graph.edge_index_dict), model(graph.x_dict, {})
        for kind in a:
            self.assertTrue(torch.equal(a[kind], b[kind]))

    def test_protocolo_rankings_e_seeds(self):
        split, users, events, edges = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, frame in [('user', users), ('event', events), ('edge', edges)]:
                frame.to_csv(root / f'{name}.csv', index=False)
            source, output = root / 'divisao.pt', root / 'experiment'
            torch.save(split, source)
            with contextlib.redirect_stdout(io.StringIO()):
                results = executar(root, source, output, seeds=(42, 43), epochs=2)
            self.assertEqual(len(results), 8)
            integrity = json.loads((output / 'integridade.json').read_text())
            self.assertTrue(integrity['avaliacao_identica'])
            self.assertFalse(integrity['teste_avaliado'])
            folder = output / 'context_only/gnn_seed_42'
            with np.load(folder / 'ranking/embeddings.npz') as saved:
                np.testing.assert_array_equal(saved['user_original_id'], [10, 30, 50, 90])
                self.assertEqual(saved['user'].shape, (4, 32))
            ranked = pd.read_csv(folder / 'ranking/ranking.csv')
            self.assertEqual(ranked.set_index('wallet_id').loc[90, 'wallet'], 'd')
            self.assertEqual(set(ranked.wallet_id), {10, 30, 50, 90})
            comparison = comparar_rankings(ranked, ranked)
            self.assertEqual(comparison['jaccard_0.01'], 1.)
            with self.assertRaises(ValueError):
                exportar(source, folder / 'melhor_modelo.pt', root / 'user.csv', root / 'wrong')
            with self.assertRaises(FileExistsError):
                executar(root, source, output, epochs=2)


if __name__ == '__main__':
    unittest.main()
