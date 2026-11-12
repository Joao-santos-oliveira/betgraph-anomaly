import unittest
import pandas as pd

from experimentos.artigo.pareado import comparar_posicoes


class MudancaPareadaTest(unittest.TestCase):
    def test_compara_o_mesmo_id_independente_da_ordem_das_linhas(self):
        rows = []
        for dose in [0, 4, 10, 20]:
            for user in range(50):
                # Um alvo melhora, um piora, os demais mantêm sua posição.
                direction = (user == 0) - (user == 1)
                rows.append(dict(realizacao=51011, metodo='Relacional', seed=0,
                                 papel='alvo', dose=dose, wallet_id=user,
                                 posicao=100 + 2*user - direction*dose))
        result = comparar_posicoes(pd.DataFrame(rows).sample(frac=1, random_state=7))
        self.assertEqual(result.subiram.tolist(), [1, 1, 1])
        self.assertEqual(result.desceram.tolist(), [1, 1, 1])
        self.assertEqual(result.iguais.tolist(), [48, 48, 48])
