"""Integridade da intervencao e compatibilidade com os algoritmos intactos."""
import contextlib
import copy
import io
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experimentos.piloto_sintetico.comum import (inventory, read_json, sha, validate_protocol,
                                                verify_inventory, write_json)
from experimentos.piloto_sintetico.gerar import Inviavel, generate, inject


def protocol():
    return read_json(ROOT / 'experimentos/piloto_sintetico/protocolo_v1.json')


def fixture():
    import numpy as np
    import pandas as pd
    import torch
    from torch_geometric.data import HeteroData
    from experimentos.piloto_sintetico.comum import legacy_imports
    legacy_imports()
    from preparar_gnn import FORWARD, REVERSE
    graph = HeteroData()
    graph['user'].x = torch.zeros(13, 5)
    graph['user'].original_id = torch.tensor([100 + 7 * u for u in range(13)])
    graph['event'].x = torch.zeros(8, 3)
    graph['event'].original_id = torch.tensor([200 + 3 * e for e in range(8)])
    context = torch.tensor([(u, e) for u in range(12) for e in (u % 8, (u + 1) % 8)]).t()
    graph[FORWARD].edge_index = context
    graph[REVERSE].edge_index = context.flip(0)
    split = {'context': graph, 'train_positive': torch.tensor([[12, 12, 0], [0, 1, 5]]),
             'val_positive': torch.tensor([[12, 1], [2, 6]]), 'test_positive': torch.tensor([[12, 2], [3, 7]])}
    split['all_positive'] = torch.cat([context] + [split[k] for k in ('train_positive', 'val_positive', 'test_positive')], dim=1)
    users = pd.DataFrame({'wallet_id': graph['user'].original_id.numpy(), 'wallet': [f'w{u}' for u in range(13)]})
    events = pd.DataFrame({'condition_node_id': graph['event'].original_id.numpy(),
                           'duracao_mercado_segundos': np.arange(1, 9, dtype=float) * 100,
                           'duracao_mercado_valida': [True] * 8, 'categoria_principal': ['A', 'B'] * 4})
    pairs = split['all_positive']
    edges = pd.DataFrame({'wallet_id': graph['user'].original_id[pairs[0]].numpy(),
                          'wallet': [f'w{u}' for u in pairs[0].tolist()],
                          'condition_node_id': graph['event'].original_id[pairs[1]].numpy(),
                          'volume_usd': np.arange(1, pairs.shape[1] + 1, dtype=float),
                          'n_trades': np.ones(pairs.shape[1]), 'preco_medio': np.full(pairs.shape[1], .5),
                          'token_outcome_id': pairs[1].numpy() * 2,
                          'primeiro_trade': ['2026-09-16T01:00:00Z'] * pairs.shape[1]})
    duplicate = edges.iloc[[0]].copy()
    duplicate['token_outcome_id'] += 1
    edges = pd.concat([edges, duplicate], ignore_index=True)
    p = protocol()
    p.update(n_targets=2, n_markets=2, min_user_degree=2, min_market_degree=2, concentration=.5, max_attempts=1000)
    p['model_seeds'] = [42, 73]
    p['training']['epochs'] = 2
    return split, users.sample(frac=1, random_state=1), events, edges, p


class ProtocolTest(unittest.TestCase):
    def test_schema_e_seeds(self):
        p = protocol()
        self.assertEqual(validate_protocol(p)['n_targets'], 50)
        for change in ({'condition': 'full_day'}, {'selection_seed': 42}, {'max_attempts': 0}, {'model_seeds': [42, 42]}):
            with self.assertRaises(ValueError):
                validate_protocol({**p, **change})

    def test_hash_detecta_alteracao(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'input.txt'
            path.write_text('original')
            saved = inventory([path])
            verify_inventory(saved)
            path.write_text('alterado')
            with self.assertRaisesRegex(ValueError, 'Arquivo alterado'):
                verify_inventory(saved)

    def test_entrada_ausente_e_destino_existente(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / 'run'
            with self.assertRaisesRegex(ValueError, 'Entradas ausentes'):
                generate(ROOT / 'experimentos/piloto_sintetico/protocolo_v1.json', root / 'input', root / 'split.pt', out)
            self.assertEqual(read_json(out / 'manifesto_injecao.json')['status'], 'bloqueado')
            with self.assertRaises(FileExistsError):
                generate(ROOT / 'experimentos/piloto_sintetico/protocolo_v1.json', root / 'input', root / 'split.pt', out)


class InjectionTest(unittest.TestCase):
    def test_trocas_reproduziveis_blocos_e_particoes(self):
        import pandas as pd
        import torch
        split, users, _, edges, p = fixture()
        before_split, before_edges = copy.deepcopy(split), edges.copy(deep=True)
        a, ae, alog, labels, diagnostic = inject(split, users, edges, p)
        b, be, blog, blabels, second = inject(split, users, edges, p)
        self.assertTrue(diagnostic['checks']['degrees_equal'])
        self.assertEqual(diagnostic, second)
        self.assertEqual(int(labels.alvo.sum()), 2)
        self.assertFalse(labels.loc[labels.sem_contexto, ['alvo', 'doadora']].any().any())
        self.assertTrue(all(x['after'] >= x['required'] for x in diagnostic['concentration']))
        pd.testing.assert_frame_equal(ae, be)
        pd.testing.assert_frame_equal(alog, blog)
        pd.testing.assert_frame_equal(labels, blabels)
        pd.testing.assert_frame_equal(edges, before_edges)
        for k in ('train_positive', 'val_positive', 'test_positive', 'all_positive'):
            self.assertTrue(torch.equal(split[k], before_split[k]))
        from preparar_gnn import FORWARD
        self.assertTrue(torch.equal(a['context'][FORWARD].edge_index, b['context'][FORWARD].edge_index))
        self.assertTrue(torch.equal(split['context'][FORWARD].edge_index, before_split['context'][FORWARD].edge_index))
        untouched = [c for c in edges.columns if c not in ('wallet', 'wallet_id')]
        pd.testing.assert_frame_equal(edges[untouched], ae[untouched])
        reserved = edges.wallet_id.eq(184)
        pd.testing.assert_frame_equal(edges.loc[reserved], ae.loc[reserved])
        # Log permite reconstruir exatamente o CSV, inclusive blocos com dois outcomes.
        reconstructed = edges.copy(deep=True)
        names = users.set_index('wallet_id').wallet
        for row in alog.itertuples():
            reconstructed.loc[row.rows_target_to_donor, ['wallet_id', 'wallet']] = [row.donor_wallet_id, names[row.donor_wallet_id]]
            reconstructed.loc[row.rows_donor_to_target, ['wallet_id', 'wallet']] = [row.target_wallet_id, names[row.target_wallet_id]]
        pd.testing.assert_frame_equal(reconstructed, ae)

    def test_inviabilidade_nao_relaxa_cenario(self):
        split, users, _, edges, p = fixture()
        for changed in ({'n_targets': 100}, {'min_market_degree': 100}, {'max_attempts': 1, 'concentration': 1.0}):
            with self.assertRaises(Inviavel):
                inject(split, users, edges, {**p, **changed})

    def test_rejeita_csv_desalinhado(self):
        split, users, _, edges, p = fixture()
        bad = edges.copy()
        bad.loc[0, 'wallet'] = 'outra_carteira'
        with self.assertRaisesRegex(ValueError, 'Identidade'):
            inject(split, users, bad, p)
        # Eliminar a unica linha de um par reservado quebra a cobertura.
        with self.assertRaisesRegex(ValueError, 'cobrir exatamente'):
            inject(split, users, edges.iloc[:-2], p)

    def test_features_nao_usam_ocultas_e_rotulos(self):
        import numpy as np
        import torch
        split, users, events, edges, p = fixture()
        changed, altered, _, labels, _ = inject(split, users, edges, p)
        from features_contexto import features_contexto
        from preparar_gnn import FORWARD
        a, _, _, _ = features_contexto(changed, users, events, altered)
        shifted = altered.copy()
        shifted.loc[shifted.wallet_id.eq(184), ['volume_usd', 'n_trades', 'preco_medio']] = 999999.
        b, _, _, _ = features_contexto(changed, users, events, shifted)
        for kind in ('user', 'event'):
            self.assertTrue(torch.equal(a['context'][kind].x, b['context'][kind].x))
            self.assertTrue(np.isfinite(a['context'][kind].x.numpy()).all())
        self.assertEqual(a['context']['user'].x.shape[1], 6)
        self.assertNotIn('alvo', a['context']['user'])
        # As features dos mercados permanecem iguais antes/depois.
        original, _, _, _ = features_contexto(split, users, events, edges)
        self.assertTrue(torch.equal(original['context']['event'].x, a['context']['event'].x))
        self.assertEqual(int(labels.alvo.sum()), p['n_targets'])

    def test_avaliacao_scores_empates_universo_e_delta(self):
        import numpy as np
        import pandas as pd
        from experimentos.piloto_sintetico.avaliar import score_ranking
        labels = pd.DataFrame({'wallet_id': np.arange(120), 'alvo': [1] * 50 + [0] * 70,
                               'doadora': [False] * 120, 'grupo': ['alvo'] * 50 + ['intacta'] * 70,
                               'elegivel': [True] * 120})
        ranking = pd.DataFrame({'wallet_id': np.arange(120), 'posicao': np.arange(1, 121),
                                'score_anomalia': [1.] * 50 + [0.] * 70})
        rows, _ = score_ranking(ranking, labels)
        self.assertEqual(rows[0]['recovered_50'], 50)
        self.assertEqual(rows[0]['precision_100'], .5)
        self.assertEqual(rows[0]['ap'], 1.)
        self.assertEqual(rows[0]['auc'], 1.)
        self.assertEqual(rows[0]['tied_at_cutoff_50'], 50)
        with self.assertRaises(ValueError):
            score_ranking(ranking.iloc[:-1], labels)
        with self.assertRaises(ValueError):
            score_ranking(ranking.iloc[::-1], labels)

    def test_ponta_a_ponta_sem_intervencao_e_tres_metodos(self):
        import numpy as np
        import pandas as pd
        import torch
        from experimentos.piloto_sintetico.executar import execute, fit_scenario
        from experimentos.piloto_sintetico.avaliar import evaluate
        from experimentos.piloto_sintetico.comum import seal_tree
        split, users, events, edges, p = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entry = root / 'input'
            entry.mkdir()
            for name, frame in [('user', users), ('event', events), ('edge', edges)]:
                frame.to_csv(entry / f'{name}.csv', index=False)
            source = entry / 'divisao.pt'
            torch.save(split, source)
            config = root / 'protocol.json'
            write_json(config, p)
            run = root / 'run'
            before = sha(source)
            with contextlib.redirect_stdout(io.StringIO()):
                generate(config, entry, source, run)
                status = execute(run)
                metrics = evaluate(run)
            self.assertEqual(status['n_treinos'], 8)
            self.assertEqual(status['n_rankings'], 10)
            self.assertEqual(len(metrics), 30)  # 10 rankings x tres universos.
            self.assertEqual(sha(source), before)
            self.assertFalse(read_json(run / 'integridade.json')['teste_avaliado'])
            position = pd.read_csv(run / 'avaliacao/posicoes_antes_depois.csv')
            np.testing.assert_array_equal(position.mudanca_posicao, position.posicao_original - position.posicao_injetada)
            with self.assertRaises(ValueError):
                execute(run)
            with self.assertRaises(FileExistsError):
                evaluate(run)
            # Duas entradas identicas sem intervencao produzem os mesmos resultados.
            import shutil
            pp = copy.deepcopy(p)
            pp['model_seeds'] = [42]
            refs = []
            for scenario in ('a', 'b'):
                folder = root / scenario
                shutil.copytree(run / 'original/entrada', folder / 'entrada')
                with contextlib.redirect_stdout(io.StringIO()):
                    _, reference = fit_scenario(folder, pp)
                refs.append(reference)
            for model, relative in [('features', 'features/ranking.csv'), ('gnn', 'gnn_seed_42/ranking/ranking.csv'),
                                    ('mlp', 'mlp_seed_42/ranking/ranking.csv')]:
                left, right = [pd.read_csv(root / x / 'resultados' / relative) for x in ('a', 'b')]
                pd.testing.assert_frame_equal(left, right)
            for name in ('val', 'test'):
                for key in ('pairs', 'labels'):
                    self.assertTrue(torch.equal(refs[0][name][key], refs[1][name][key]))
            # Alteracao de uma entrada congelada e detectada antes de reexecutar.
            with (run / 'original/entrada/user.csv').open('a') as stream:
                stream.write('\n')
            with self.assertRaisesRegex(ValueError, 'Artefato alterado'):
                execute(run)


if __name__ == '__main__':
    unittest.main()
