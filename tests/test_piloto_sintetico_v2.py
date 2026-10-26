"""Regressao da impossibilidade individual e integracao da elegibilidade v2."""
import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experimentos.piloto_sintetico.elegibilidade_v2 import selecionar_elegiveis
from experimentos.piloto_sintetico.comum import read_json, validate_protocol
from experimentos.piloto_sintetico.gerar import inject
from test_piloto_sintetico import fixture


class EligibilityV2Test(unittest.TestCase):
    def test_exclui_caso_12694_e_aceita_limite_exato(self):
        markets = set(range(10))
        neighbors = {0: set(range(10, 26)), 1: set(range(10, 23)), 2: set(range(10, 15))}
        degrees = [16, 13, 5]
        ids = [12694, 100, 200]
        # Dois mercados reservados: capacidade 8; exigencias 10 e 8 respectivamente.
        reserved = {(0, 0), (0, 1), (1, 0), (1, 1), (2, 100)}
        eligible, report = selecionar_elegiveis(neighbors, degrees, markets, reserved, ids, .6, 5)
        self.assertEqual(eligible, [1, 2])
        self.assertEqual(report['eligible_before_capacity_check'], 3)
        self.assertEqual(report['excluded_by_reserved_capacity'][0]['wallet_id'], 12694)
        self.assertEqual(report['excluded_by_reserved_capacity'][0]['required'], 10)
        self.assertEqual(report['excluded_by_reserved_capacity'][0]['allowed'], 8)
        # Uma conexao fora do grupo de mercados nao reduz a capacidade.
        self.assertIn(2, eligible)

    def test_preserva_filtros_de_grau_e_concentracao(self):
        neighbors = {0: {0, 1, 2}, 1: {3}, 2: {3, 4, 5, 6, 7, 8}, 3: {3, 4}}
        eligible, report = selecionar_elegiveis(neighbors, [3, 1, 6, 2], {0, 1}, set(),
                                               [10, 20, 30, 40], .6, 2)
        self.assertEqual(eligible, [3])
        self.assertEqual(report['excluded_by_reserved_capacity'], [])

    def test_v2_muda_apenas_versao_do_protocolo(self):
        base=ROOT / 'experimentos/piloto_sintetico'
        a,b=[read_json(base/f'protocolo_{v}.json') for v in ('v1','v2')]
        self.assertEqual(validate_protocol(b)['version'],'v2')
        self.assertEqual({k:v for k,v in a.items() if k!='version'},
                         {k:v for k,v in b.items() if k!='version'})
        with self.assertRaisesRegex(ValueError, 'Versao'):
            validate_protocol({**b, 'version':'desconhecida'})

    def test_integracao_v2_reproduzivel_com_invariantes(self):
        import pandas as pd
        import torch
        split, users, _, edges, p = fixture()
        p['version']='v2'
        before=copy.deepcopy(split)
        a,ae,swaps,labels,diagnostic=inject(split,users,edges,p)
        b,be,again,labels2,other=inject(split,users,edges,p)
        self.assertEqual(diagnostic,other)
        self.assertTrue(all(diagnostic['checks'].values()))
        self.assertEqual(int(labels.alvo.sum()),p['n_targets'])
        self.assertTrue(all(r['after']>=r['required'] for r in diagnostic['concentration']))
        pd.testing.assert_frame_equal(ae,be)
        pd.testing.assert_frame_equal(swaps,again)
        pd.testing.assert_frame_equal(labels,labels2)
        for key in ('train_positive','val_positive','test_positive','all_positive'):
            self.assertTrue(torch.equal(split[key],before[key]))


if __name__ == '__main__':
    unittest.main()
