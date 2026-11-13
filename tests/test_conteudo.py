"""Confere as tabelas do artigo contra os números do roteiro e a regra de eventos da T6."""
import unittest

import pandas as pd

from experimentos.conteudo import estilo, tabelas


class FormatacaoTest(unittest.TestCase):
    def test_numero_brasileiro(self):
        self.assertEqual(estilo.numero(229072), '229.072')
        self.assertEqual(estilo.numero(0.96111, 4), '0,9611')
        self.assertEqual(estilo.contagem(36.666666), '36,67')
        self.assertEqual(estilo.contagem(50.0), '50')

    def test_latex_escapa_simbolos(self):
        self.assertEqual(estilo.tex('2% & 3_4'), r'2\% \& 3\_4')
        self.assertEqual(estilo.tex('1 → 2'), r'1 $\rightarrow$ 2')


class DenominadoresTest(unittest.TestCase):
    """Valores fixados no roteiro do artigo."""

    @classmethod
    def setUpClass(cls):
        cls.t3 = tabelas.t3().dados

    def valor(self, desenho, mercados, metodo, corte):
        r = self.t3.query('desenho == @desenho and mercados == @mercados and metodo == @metodo and corte == @corte')
        self.assertEqual(len(r), 1)
        return r.iloc[0]

    def test_denominador_sempre_50(self):
        self.assertTrue(self.t3.denominador.eq(50).all())

    def test_misto_intensidade_20(self):
        # 15/15 + 15/15 + 20/20 = 50/50 em cada ranking
        for metodo in ('GNN híbrida', 'Relacional'):
            self.assertEqual(self.valor('Misto', 20, metodo, 100).contagem, 50)
        por_sorteio = pd.read_csv(estilo.RESULTADOS / 'por_realizacao_seed.csv').query(
            "fase == 'depois' and mercados == 20 and papel == 'alvo' and metodo == 'GNN híbrida'")
        self.assertEqual(sorted(por_sorteio.groupby('realizacao').n.first()), [15, 15, 20])

    def test_pareado_intensidade_20(self):
        # 39/50, 34/50 e 37/50: média de 36,67/50 por ranking
        r = self.valor('Pareado', 20, 'GNN híbrida', 100)
        self.assertAlmostEqual(r.contagem, (39 + 34 + 37) / 3)
        sorteios = pd.read_csv(estilo.RESULTADOS / 'pareado' / 'por_sorteio_seed.csv').query(
            "dose == 20 and papel == 'alvo' and metodo == 'GNN híbrida'")
        self.assertEqual(sorted(sorteios.groupby('realizacao').top100.mean()), [34, 37, 39])

    def test_similaridade_isolada_nao_recupera(self):
        self.assertTrue(self.t3.query("metodo == 'Similaridade da GNN'").contagem.eq(0).all())


class EventoTest(unittest.TestCase):
    def test_confrontos_identificados(self):
        from experimentos.artigo.extrair import evento
        chave = 'cr flamengo | red bull bragantino'
        self.assertEqual(evento('Will CR Flamengo vs. Red Bull Bragantino end in a draw?'), chave)
        self.assertEqual(evento('Red Bull Bragantino vs. CR Flamengo: O/U 2.5'), chave)
        self.assertEqual(evento('Exact Score: RC Deportivo A Coruña 2 - 1 Sevilla FC?'),
                         'rc deportivo a coruña | sevilla fc')
        self.assertEqual(evento('LoL: Team WE vs JD Gaming (BO5) - LPL Regional Finals Playoffs'), 'jd gaming | team we')
        self.assertEqual(evento('Will there be a run scored in the first inning?: Miami Marlins vs. Arizona Diamondbacks'),
                         'arizona diamondbacks | miami marlins')

    def test_indeterminados(self):
        from experimentos.artigo.extrair import evento
        for pergunta in ('Will Sparta Rotterdam win on 2026-09-19?', 'Will XRP dip to $0.80 by December 31, 2026?',
                         'Oneil Cruz: Total Bases O/U 5.5', None):
            self.assertIsNone(evento(pergunta))


if __name__ == '__main__':
    unittest.main()
