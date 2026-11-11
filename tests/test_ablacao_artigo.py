import unittest

import numpy as np

from experimentos.gnn_v2.detectores import hibrido


class ScoreSimilaridadeTest(unittest.TestCase):
    def test_mesmos_candidatos_tres_parceiros_e_denominador(self):
        # 0 e 1 compartilham dois mercados. A carteira 2 aparece em apenas um.
        pairs = np.array([[0, 0, 1, 1, 2], [0, 1, 0, 1, 0]])
        z = np.array([[1., 0.], [0., 1.], [1., 0.]])
        score = hibrido(pairs, 3, 2, z, apenas_similaridade=True)
        np.testing.assert_allclose(score, [1/6, 1/6, 0])

    def test_retem_os_tres_maiores(self):
        pairs = np.array([(u, m) for u in range(5) for m in (0, 1)]).T
        z = np.array([[1., 0.], [1., 0.], [0., 1.], [-1., 0.], [1., 0.]])
        scores = hibrido(pairs, 5, 2, z, apenas_similaridade=True)
        self.assertAlmostEqual(scores[0], (1 + 1 + .5)/3)
