"""Verificacoes de integracao da preparacao em uma fixture pequena."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from preparar_features import FILES, preparar
from construir_hetero import construir


class PreparacaoTest(unittest.TestCase):
    def test_imputacao_indices_reversas_e_reuso(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            users = pd.DataFrame({'wallet': ['a', 'b'], 'wallet_id': [3, 99],
                'volume_total_usd': [10., 40.], 'n_trades_total': [1, 4],
                'n_mercados_distintos': [1, 2], 'ticket_medio_geral': [10., 10.],
                'preco_medio_ponderado': [.2, .8]})
            events = pd.DataFrame({'condition_id': ['x', 'y', 'z'],
                'condition_node_id': [7, 42, 100], 'categoria_principal': ['A', 'B', 'A'],
                'duracao_mercado_segundos': [10., np.nan, 30.],
                'duracao_mercado_valida': [True, False, True]})
            edges = pd.DataFrame({'wallet': ['a', 'b', 'b'], 'wallet_id': [3, 99, 99],
                'condition_id': ['x', 'y', 'z'], 'condition_node_id': [7, 42, 100],
                'token_outcome': ['Yes'] * 3, 'token_outcome_id': [5] * 3,
                'volume_usd': [10., 20., 20.], 'preco_medio': [.2, .7, .9],
                'primeiro_trade': ['2026-09-16 01:00:00+00:00'] * 3})
            for key, df in [('user', users), ('event', events), ('edge', edges)]:
                df.to_csv(root / FILES[key], index=False)
            for key, mapping in [('wallets', {'a': 3, 'b': 99}),
                                 ('conditions', {'x': 7, 'y': 42, 'z': 100}),
                                 ('outcomes', {'Yes': 5})]:
                (root / f'mapeamento_{key}.json').write_text(json.dumps(mapping))
            first, second = root / 'first', root / 'second'
            with contextlib.redirect_stdout(io.StringIO()):
                preparar(root, first, sample_size=0)
                preparar(root, second, sample_size=0, preprocessador=first / 'preprocessador.joblib')
                construir(first, root)
            bundle = joblib.load(first / 'preprocessador.joblib')
            self.assertEqual(bundle['mediana_duracao'], 20.)
            with np.load(first / 'features.npz') as one, np.load(second / 'features.npz') as two:
                for key in one.files:
                    np.testing.assert_array_equal(one[key], two[key])
                np.testing.assert_array_equal(one['event'][:, 1], [0, 1, 0])
                recovered = np.expm1(bundle['scalers']['event'].inverse_transform(one['event'][:, :1]))
                np.testing.assert_allclose(recovered.ravel(), [10, 20, 30], rtol=1e-6)
            graph = torch.load(first / 'hetero.pt', weights_only=False)
            self.assertEqual(graph['event'].num_nodes, 3)
            np.testing.assert_array_equal(graph['user'].original_id, [3, 99])
            relation = ('user', 'bets_on', 'event')
            np.testing.assert_array_equal(graph[relation].edge_index, [[0, 1, 1], [0, 1, 2]])
            np.testing.assert_array_equal(graph['event', 'rev_bets_on', 'user'].edge_index,
                                          graph[relation].edge_index.flip(0))
            self.assertEqual(json.loads((first / 'indices_originais.json').read_text())['user'], {'3': 0, '99': 1})
            # Um endpoint sem no correspondente deve falhar antes de salvar features.
            edges.loc[0, 'wallet_id'] = 1000
            edges.to_csv(root / FILES['edge'], index=False)
            with self.assertRaisesRegex(ValueError, 'orfas'):
                preparar(root, root / 'invalid', sample_size=0)


if __name__ == '__main__':
    unittest.main()
