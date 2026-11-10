"""Confere a tabela comparativa contra os valores publicados da rodada v3."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evidencias import tabela


class TabelaComparativaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metricas, cls.sha = tabela.ler_metricas()
        cls.resumo = tabela.resumir(cls.metricas, list(tabela.PRINCIPAIS) + list(tabela.APENDICE))

    def valor(self, metodo, cenario, coluna='top100_media'):
        linha = self.resumo[(self.resumo.metodo == metodo) & (self.resumo.cenario == cenario)]
        self.assertEqual(len(linha), 1)
        return linha.iloc[0][coluna]

    def test_valores_publicados(self):
        self.assertAlmostEqual(self.valor('A', 'S3_estrutura'), 32.0)
        self.assertAlmostEqual(self.valor('E_features', 'S3_volume'), 9.0)
        self.assertAlmostEqual(self.valor('E_v2', 'S3_estrutura'), 31.8333, places=3)
        self.assertAlmostEqual(self.valor('A', 'S2'), 29.5)
        self.assertAlmostEqual(self.valor('C', 'S1'), 0.0)
        self.assertAlmostEqual(self.valor('A', 'S3_estrutura', 'ap_media'), 0.2319, places=4)

    def test_denominador_e_realizacoes(self):
        self.assertTrue((self.resumo.denominador == 50).all())
        self.assertTrue((self.resumo.carteiras == 5000).all())
        self.assertTrue((self.resumo.realizacoes == '30101/30201').all())
        # Baselines: uma execução por realização; redes: três seeds por realização.
        self.assertEqual(self.valor('A', 'S1', 'execucoes'), 2)
        self.assertEqual(self.valor('C', 'S1', 'execucoes'), 6)

    def test_custos_mapeados_por_branch(self):
        with tempfile.TemporaryDirectory() as pasta:
            v3, v2 = Path(pasta, 'v3.csv'), Path(pasta, 'v2.csv')
            pd.DataFrame({'metodo': ['relacional', 'gnn', 'gnn'], 'seed': [42, 42, 73],
                          'segundos': [0.5, 10.0, 20.0], 'pico_rss_mib': [400, 700, 710]}).to_csv(v3, index=False)
            pd.DataFrame({'metodo': ['gnn'], 'seed': [42], 'segundos': [5.0],
                          'pico_rss_mib': [600]}).to_csv(v2, index=False)
            custos = tabela.ler_custos({'v3': v3, 'v2': v2}).set_index('metodo')
            self.assertAlmostEqual(custos.loc['C', 'segundos_media'], 15.0)
            self.assertEqual(custos.loc['C', 'pico_rss_mib_max'], 710)
            self.assertAlmostEqual(custos.loc['E_v2', 'segundos_media'], 5.0)
            self.assertNotIn('D', custos.index)

    def test_custos_do_benchmark(self):
        with tempfile.TemporaryDirectory() as pasta:
            resumo = Path(pasta, 'resumo.csv')
            pd.DataFrame({
                'branch': ['v3', 'v3', 'v3', 'v2', 'v3'], 'etapa': ['relacional', 'gnn', 'gnn', 'gnn', 'coorte_5000'],
                'seed': [42, 42, 73, 42, None], 'segundos_metodo_median': [0.3, 30.0, 34.0, 25.0, None],
                'pico_rss_mib_median': [470, 720, 730, 650, 640], 'repeticoes': [3, 3, 3, 3, 3]}).to_csv(resumo, index=False)
            custos = tabela.ler_benchmark(resumo).set_index('metodo')
            self.assertAlmostEqual(custos.loc['C', 'segundos_media'], 32.0)
            self.assertEqual(custos.loc['C', 'pico_rss_mib_max'], 730)
            self.assertEqual(custos.loc['C', 'seeds'], '42/73')
            self.assertAlmostEqual(custos.loc['E_v2', 'segundos_media'], 25.0)
            self.assertEqual(set(custos.index), {'A', 'C', 'E_v2'})  # etapas sem seed não viram método

    def test_saida_completa(self):
        with tempfile.TemporaryDirectory() as pasta:
            tabela.main(['--saida', pasta])
            principal = pd.read_csv(Path(pasta, 'tabela_comparativa.csv'))
            self.assertEqual(principal.codigo.tolist(), ['A', 'E_features', 'E_v2', 'C'])
            self.assertIn('Decoys S2', Path(pasta, 'tabela_comparativa.md').read_text(encoding='utf-8'))


class GraficosTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from scripts.evidencias import graficos
        except ImportError as exc:  # plotly é dependência opcional (requirements-figuras.txt)
            raise unittest.SkipTest(f'plotly indisponível: {exc}')
        cls.graficos = graficos
        cls.pontos = graficos.dados(tabela.ler_metricas()[0])

    def test_pontos_por_execucao(self):
        contagem = self.pontos.groupby(['cenario', 'metodo']).size()
        self.assertEqual(contagem[('S1', 'A')], 2)       # uma execução por realização
        self.assertEqual(contagem[('S1', 'C')], 6)       # duas realizações × três seeds
        self.assertEqual(set(self.pontos.cenario), {'S1', 'S2', 'S3_estrutura', 'S3_volume'})
        self.assertTrue((self.pontos.alvos == 50).all())

    def test_pontos_dentro_da_barra(self):
        desvio = (self.pontos.x - self.pontos.metodo.map(
            {m: i for i, m in enumerate(tabela.PRINCIPAIS)})).abs()
        self.assertTrue((desvio < self.graficos.MEIA_BARRA).all())
        self.assertLess(self.pontos.top100.max(), self.graficos.Y_MEDIA)

    def test_figuras_com_denominador(self):
        fig1 = self.graficos.figura_recuperacao(self.pontos)
        fig2 = self.graficos.figura_decoys(self.pontos)
        nota1, nota2 = fig1.layout.annotations[-1].text, fig2.layout.annotations[-1].text
        self.assertIn('50 alvos artificiais', nota1)
        self.assertIn('50 decoys', nota2)
        for nota in (nota1, nota2):
            self.assertIn('30101', nota)
            self.assertIn('30201', nota)
        medias = [a.text for a in fig1.layout.annotations if a.text.startswith('média')]
        self.assertIn('média<br><b>32,0</b>', medias)


class BenchmarkTest(unittest.TestCase):
    def test_nomes_das_etapas(self):
        from scripts.evidencias.benchmark import nome_da_etapa
        self.assertEqual(nome_da_etapa(['/x/scripts/modelos_amostra.py', 'gnn', '--seed', 73, '--epocas', 60]),
                         ('gnn', 73))
        self.assertEqual(nome_da_etapa(['/x/scripts/modelos_amostra.py', 'preparar', '--escopo', '5000']),
                         ('preparar', None))
        self.assertEqual(nome_da_etapa(['src/filtrar_recorrentes.py', 'a.csv']), ('filtrar_recorrentes', None))
        self.assertEqual(nome_da_etapa(['/x/scripts/coorte.py', '/repo']), ('coorte_5000', None))

    def test_resumo_por_etapa(self):
        from scripts.evidencias.benchmark import resumir
        etapas = pd.DataFrame({'repeticao': [1, 2, 3, 1], 'branch': ['v3'] * 4,
                               'etapa': ['gnn', 'gnn', 'gnn', 'coorte_5000'], 'seed': [42, 42, 42, None],
                               'metodo': ['C', 'C', 'C', None], 'segundos_processo': [10., 30., 20., 5.],
                               'pico_rss_mib': [700., 710., 705., 600.], 'segundos_metodo': [8., 28., 18., None]})
        resumo = resumir(etapas).set_index('etapa')
        self.assertEqual(resumo.loc['gnn', 'segundos_processo_median'], 20.)
        self.assertEqual(resumo.loc['gnn', 'segundos_metodo_median'], 18.)
        self.assertEqual(resumo.loc['gnn', 'repeticoes'], 3)
        self.assertTrue(pd.isna(resumo.loc['coorte_5000', 'seed']))


class SubgrafosTest(unittest.TestCase):
    """Regra de pareamento e corte da vizinhança, em dados controlados e na seleção gravada."""

    def setUp(self):
        from scripts.evidencias import subgrafos
        self.s = subgrafos

    def test_pareamento_respeita_caliper_e_escolhe_o_mais_proximo(self):
        atividade = pd.DataFrame({'grau': [10, 10, 11, 200, 1], 'volume_usd': [1000., 1100., 5000., 1000., 1000.]},
                                 index=[1, 2, 3, 4, 5])
        controle, info = self.s.parear(1, atividade, {2, 3, 4, 5})
        self.assertEqual(controle, 2)
        self.assertLessEqual(max(abs(info['delta_grau_z']), abs(info['delta_volume_z'])), self.s.CALIPER)
        with self.assertRaises(ValueError):
            self.s.parear(1, atividade, {4}, caliper=0.01)

    def test_corte_da_vizinhanca(self):
        # Focal 0 opera 15 mercados; parceiro 1 compartilha 0..4; carteiras 2..9 tocam vários mercados.
        linhas = [(0, m, 100. + m) for m in range(15)] + [(1, m, 50.) for m in range(5)]
        linhas += [(w, m, 10.) for w in range(2, 10) for m in range(w, w + 3)]
        ligacoes = pd.DataFrame(linhas, columns=['wallet_id', 'condition_node_id', 'volume_usd'])
        viz = self.s.vizinhanca(0, 1, ligacoes)
        self.assertEqual(len(viz['mercados']), self.s.MAX_MERCADOS)
        self.assertEqual(set(viz['mercados'][:5]), set(range(5)))   # compartilhados primeiro
        self.assertEqual(viz['carteiras'][:2], [0, 1])
        self.assertLessEqual(len(viz['carteiras']), 2 + self.s.MAX_OUTRAS)
        self.assertEqual(viz['omitidos']['mercados_da_focal'], 5)
        desenhadas = set(viz['ligacoes'].wallet_id)
        self.assertTrue(desenhadas <= set(viz['carteiras']))
        self.assertTrue(set(viz['ligacoes'].condition_node_id) <= set(viz['mercados']))

    def test_selecao_gravada_cumpre_a_regra(self):
        if not self.s.SELECAO.exists():
            self.skipTest('Seleção ainda não gerada (subgrafos.py selecionar)')
        sel = json.loads(self.s.SELECAO.read_text(encoding='utf-8'))
        caso, controle = sel['resumo']['caso'], sel['resumo']['controle']
        self.assertEqual(caso['posicao_focal'], 1)
        self.assertGreater(controle['posicao_focal'], self.s.FORA_DO_TOPO)
        par = sel['pareamento']
        self.assertLessEqual(max(abs(par['delta_grau_z']), abs(par['delta_volume_z'])), par['caliper'])
        for painel in (caso, controle):
            self.assertLessEqual(len(painel['mercados']), self.s.MAX_MERCADOS)
            self.assertLessEqual(len(painel['carteiras']), 2 + self.s.MAX_OUTRAS)


if __name__ == '__main__':
    unittest.main()
