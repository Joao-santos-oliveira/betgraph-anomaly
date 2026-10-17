"""Verifica orientacao do ranking e reproducibilidade em dados controlados."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from baseline import ajustar_ranking


class BaselineTest(unittest.TestCase):
    def test_outlier_repetibilidade_e_desempate(self):
        # Grupo identico e um ponto claramente distante, sem interpretar fraude.
        x = np.zeros((100, 5), dtype=np.float32)
        x[-1] = 100
        users = pd.DataFrame({'wallet_id': np.arange(100), 'wallet': [str(i) for i in range(100)]})
        _, first, _ = ajustar_ranking(users, x)
        _, second, _ = ajustar_ranking(users, x)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first.iloc[0].wallet_id, 99)
        self.assertEqual(first.selecionada_top.sum(), 1)
        self.assertEqual(first.iloc[1:].wallet_id.tolist(), list(range(99)))
        self.assertTrue(first.score_anomalia.is_monotonic_decreasing)
        with self.assertRaises(ValueError):
            ajustar_ranking(users, x, top_fraction=0)


if __name__ == '__main__':
    unittest.main()
