"""Verifica negativos, integridade das particoes e reproducibilidade do treino."""
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
from preparar_gnn import FORWARD, REVERSE, separar
from treinar_gnn import amostrar_negativos, codigos, treinar, validar_divisao


class TreinoTest(unittest.TestCase):
    def test_negativos(self):
        forbidden = {0, 1, 5, 9}
        a = amostrar_negativos(4, 5, 10, forbidden, np.random.default_rng(42))
        b = amostrar_negativos(4, 5, 10, forbidden, np.random.default_rng(42))
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(len(codigos(a, 5)), 10)
        self.assertFalse(codigos(a, 5) & forbidden)
        self.assertTrue((a[0] < 4).all() and (a[1] < 5).all())
        with self.assertRaises(ValueError):
            amostrar_negativos(4, 5, 17, forbidden, np.random.default_rng(42))

    def test_treino_reproduzivel_e_checkpoint(self):
        torch.manual_seed(8)
        graph = HeteroData()
        for kind, count, width in [('user', 10, 5), ('event', 12, 7)]:
            graph[kind].x = torch.randn(count, width)
            graph[kind].original_id = torch.arange(count)
        graph[FORWARD].edge_index = torch.tensor([(u, e) for u in range(10) for e in range(6)]).t()
        graph[REVERSE].edge_index = graph[FORWARD].edge_index.flip(0)
        split = separar(graph)
        valid = validar_divisao(split)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'divisao.pt'
            torch.save(split, source)
            with contextlib.redirect_stdout(io.StringIO()):
                first = treinar(source, root / 'one', epochs=4, patience=3, threads=1)
                second = treinar(source, root / 'two', epochs=4, patience=3, threads=1)
            self.assertEqual(first['melhor_validacao'], second['melhor_validacao'])
            one = pd.read_csv(root / 'one/historico.csv').drop(columns='seconds')
            two = pd.read_csv(root / 'two/historico.csv').drop(columns='seconds')
            pd.testing.assert_frame_equal(one, two)
            self.assertTrue((one.gradient_norm > 0).all())
            evaluation = torch.load(root / 'one/validacao.pt', weights_only=False)
            neg = evaluation['pairs'][:, evaluation['labels'] == 0]
            self.assertFalse(codigos(neg, 12) & valid)
            reserved = torch.load(root / 'one/teste_reservado.pt', weights_only=False)
            test_neg = reserved['pairs'][:, reserved['labels'] == 0]
            self.assertFalse(codigos(test_neg, 12) & (valid | codigos(neg, 12)))
            report = json.loads((root / 'one/treino_manifest.json').read_text())
            self.assertFalse(report['teste_avaliado'])
            self.assertTrue((root / 'one/melhor_modelo.pt').exists())
            with self.assertRaises(FileExistsError):
                treinar(source, root / 'one', epochs=1)
        split['val_positive'] = split['train_positive'].clone()
        with self.assertRaises(ValueError):
            validar_divisao(split)


if __name__ == '__main__':
    unittest.main()
