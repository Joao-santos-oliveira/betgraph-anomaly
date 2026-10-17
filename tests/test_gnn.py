"""Testa separacao de pares e gradientes do encoder/decoder."""
from pathlib import Path
import sys
import unittest

import torch
from torch_geometric.data import HeteroData

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from preparar_gnn import FORWARD, REVERSE, separar
from modelo_gnn import GNNBipartida


def pares_set(pairs):
    return set(map(tuple, pairs.t().tolist()))


class GNNTest(unittest.TestCase):
    def test_divisao_e_gradientes(self):
        torch.manual_seed(42)
        data = HeteroData()
        for kind, count, features in [('user', 10, 5), ('event', 12, 7)]:
            data[kind].x = torch.randn(count, features)
            data[kind].original_id = torch.arange(count) * 10
        pairs = torch.tensor([(u, e) for u in range(10) for e in range(6)]).t()
        # Simula outcomes distintos e garante que nao atravessam as particoes.
        data[FORWARD].edge_index = torch.cat([pairs, pairs[:, :10]], dim=1)
        data[REVERSE].edge_index = data[FORWARD].edge_index.flip(0)
        original = data[FORWARD].edge_index.clone()
        split, repeated = separar(data), separar(data)
        parts = [split['context'][FORWARD].edge_index] + [split[k] for k in ['train_positive', 'val_positive', 'test_positive']]
        sets = [pares_set(p) for p in parts]
        self.assertEqual(set.union(*sets), pares_set(pairs))
        for i in range(len(sets)):
            for j in range(i):
                self.assertFalse(sets[i] & sets[j])
        self.assertTrue(torch.equal(original, data[FORWARD].edge_index))
        self.assertTrue(torch.equal(split['train_positive'], repeated['train_positive']))
        graph = split['context']
        self.assertTrue(torch.equal(graph[REVERSE].edge_index, graph[FORWARD].edge_index.flip(0)))
        self.assertNotIn('edge_attr', graph[FORWARD])
        model = GNNBipartida(5, 7)
        embeddings = model(graph.x_dict, graph.edge_index_dict)
        self.assertEqual(embeddings['user'].shape, (10, 32))
        self.assertEqual(embeddings['event'].shape, (12, 32))
        positives = split['train_positive']
        # Mercados 6..11 nao aparecem em nenhum par observado da fixture.
        negatives = positives.clone()
        negatives[1] += 6
        labels = torch.cat([torch.ones(positives.shape[1]), torch.zeros(negatives.shape[1])])
        logits = model.decode(embeddings, torch.cat([positives, negatives], dim=1))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        model.eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(model(graph.x_dict, graph.edge_index_dict)['user'],
                                        model(graph.x_dict, graph.edge_index_dict)['user']))


if __name__ == '__main__':
    unittest.main()
