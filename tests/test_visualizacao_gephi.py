"""Invariantes do roteiro das figuras Gephi (contagens, pesos, graus, regra de modalidade/evento)."""
import json
import unittest
from pathlib import Path

import pandas as pd

from visualizacao_gephi import comum
from visualizacao_gephi.layout import anel

DADOS = comum.ARESTAS.exists() and comum.MERCADOS.exists()


class RegraModalidade(unittest.TestCase):
    def test_tags(self):
        self.assertEqual(comum.modalidade('[Sports Esports Games counter strike 2]', 'Sports'), 'CS2')
        self.assertEqual(comum.modalidade('[Sports Soccer Games UEL]', 'Sports'), 'Futebol')
        self.assertEqual(comum.modalidade('[Sports Tennis Games]', 'Sports'), 'Tênis')
        self.assertEqual(comum.modalidade('[Politics Iran]', 'Politics'), 'Política')
        self.assertEqual(comum.modalidade('', 'nan'), 'indeterminado')

    def test_evento(self):
        df = pd.DataFrame({'question': ['AC Milan vs. Benfica: O/U 2.5', 'Will AC Milan win on 2026-09-16?',
                                        'Raphinha: Anytime Goalscorer'],
                           'tags': ['[Sports Soccer Games UEL]'] * 3})
        ev = comum.eventos(df)
        self.assertEqual(ev.evento[0], 'AC Milan × Benfica')
        self.assertEqual(ev.evento[1], 'AC Milan × Benfica')
        self.assertEqual(ev.regra_evento[1], 'nome_equipe')
        self.assertEqual(ev.evento[2], 'indeterminado')

    def test_evento_preserva_hifen_no_nome(self):
        df = pd.DataFrame({'question': ['Counter-Strike: QUAZAR vs ex-RUSTEC (BO3) - Group Stage',
                                        'Counter-Strike: Teletubisie vs ex-Zero Tenacity - Map 1 Winner'],
                           'tags': ['[Sports Esports Games counter strike 2]'] * 2})
        self.assertEqual(comum.eventos(df).evento.tolist(),
                         ['QUAZAR × ex-RUSTEC', 'Teletubisie × ex-Zero Tenacity'])


class Layout(unittest.TestCase):
    def test_anel_cobre_todos_e_respeita_grupos(self):
        ordem = list(range(10))
        grupos = pd.Series(['a'] * 4 + ['b'] * 6, index=ordem)
        pos, arcos = anel(ordem, grupos, lacuna_grupo=5)
        self.assertEqual(len(pos), 10)
        self.assertEqual([g for g, *_ in arcos], ['a', 'b'])


@unittest.skipUnless(DADOS, 'data/raw ausente (diretório ignorado pelo Git)')
class Roteiro(unittest.TestCase):
    def test_global(self):
        p = comum.pares()
        gw, gm = comum.graus()
        self.assertEqual((len(gw), len(gm), len(gw) + len(gm), len(p)), (27039, 16693, 43732, 229072))
        self.assertFalse(p.duplicated(['wallet_id', 'condition_node_id']).any())

    def test_nucleo(self):
        from visualizacao_gephi import recortes
        _, pesos, meta, _ = recortes.dados_nucleo()
        self.assertEqual(len(meta), 144)
        self.assertEqual(len(meta) * 5 + 0, 720)
        self.assertEqual(len(pesos), 10)
        self.assertEqual((min(pesos.values()), max(pesos.values())), (153, 160))
        self.assertNotIn(144, pesos.values())

    def test_casos(self):
        from visualizacao_gephi import recortes
        casos = recortes.dados_casos()
        self.assertEqual([c['foco'] for c in casos], [17762, 1507, 16179, 53])
        self.assertEqual([len(c['mercados']) for c in casos], [171, 171, 41, 41])
        for c in casos:
            # só arestas do foco e dos parceiros para mercados do foco
            self.assertTrue(set(c['arestas'].condition_node_id) <= set(c['mercados']))
            self.assertTrue(set(c['arestas'].wallet_id) <= {c['foco'], *c['parceiros']})


FONTE = comum.RECORTES / 'gephi_intervencao' / 'fonte'


@unittest.skipUnless((FONTE / 'arestas_grupo0_por_dose.csv').exists(), 'cópias do teste pareado não exportadas')
class Intervencao(unittest.TestCase):
    def test_graus_preservados_e_aninhamento(self):
        ar = pd.read_csv(FONTE / 'arestas_grupo0_por_dose.csv')
        man = json.loads((FONTE / 'manifesto_intervencao.json').read_text(encoding='utf-8'))
        esperado = {7896: 34, 36905: 27, 10780: 24, 7219: 31, 9621: 42}
        novos = man['mercados_novos_ordem']
        for d in (0, 4, 10, 20):
            sub = ar[ar.dose == d]
            self.assertEqual(sub.groupby('wallet_id').size().to_dict(), esperado)
            comuns = {m for m, g in sub.groupby('condition_node_id') if g.wallet_id.nunique() == 5}
            self.assertEqual(comuns, set(novos[:d]))


@unittest.skipUnless((comum.FIGURAS / 'gephi_casos.pdf').exists(), 'figuras ainda não geradas')
class Entregas(unittest.TestCase):
    def test_arquivos_e_largura(self):
        for nome in ('gephi_global', 'gephi_nucleo', 'gephi_casos', 'gephi_intervencao'):
            # O SVG global excede o limite de distribuição; PDF e PNG preservam a figura.
            for ext in (('pdf', 'png') if nome == 'gephi_global' else ('pdf', 'svg', 'png')):
                self.assertTrue((comum.FIGURAS / f'{nome}.{ext}').exists(), f'{nome}.{ext}')
            info = json.loads((comum.RECORTES / nome / 'figura.json').read_text(encoding='utf-8'))
            self.assertAlmostEqual(info['largura_cm'], 17.5, places=1)
        for nome in ('gephi_nucleo', 'gephi_casos', 'gephi_intervencao'):
            info = json.loads((comum.RECORTES / nome / 'figura.json').read_text(encoding='utf-8'))
            self.assertTrue(7.0 <= info['altura_cm'] <= 10.0, (nome, info['altura_cm']))

    def test_svg_sem_setas(self):
        for nome in ('gephi_nucleo', 'gephi_casos', 'gephi_intervencao'):
            txt = (comum.FIGURAS / f'{nome}.svg').read_text(encoding='utf-8')
            self.assertNotIn('marker-end', txt)


if __name__ == '__main__':
    unittest.main()
