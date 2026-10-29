"""Gerador v2 copiado de c0b7390; unica extensao: filtrar mercados por catalogo binario."""
import copy,math
from experimentos.piloto_sintetico.comum import legacy_imports,require,validate_protocol,PARTS
from experimentos.piloto_sintetico.gerar import Inviavel

def inject(split, users, edges, protocol, allowed_markets=None):
    """Retorna copias, log de trocas, rotulos e verificacoes calculadas."""
    import numpy as np
    import pandas as pd
    import torch
    legacy_imports()
    from preparar_gnn import FORWARD, REVERSE
    from treinar_gnn import validar_divisao
    from features_contexto import alinhar

    p = validate_protocol(protocol)
    validar_divisao(split)
    graph = split['context']
    require(set(graph[FORWARD].keys()) == {'edge_index'} and set(graph[REVERSE].keys()) == {'edge_index'},
            'Contexto deve conter apenas indices; atributos por outcome devem permanecer no CSV')
    users = alinhar(users, 'wallet_id', graph['user'].original_id)
    require('wallet' in users and not users.wallet.isna().any() and not users.wallet.duplicated().any(), 'Carteiras invalidas')
    uids = graph['user'].original_id.tolist()
    eids = graph['event'].original_id.tolist()
    require(len(set(eids)) == len(eids), 'IDs de mercados duplicados')
    ui, ei = {v: i for i, v in enumerate(uids)}, {v: i for i, v in enumerate(eids)}
    require(edges.wallet_id.isin(uids).all() and edges.condition_node_id.isin(eids).all(), 'CSV com endpoints desconhecidos')
    require('wallet' in edges and edges.wallet.eq(edges.wallet_id.map(dict(zip(uids, users.wallet)))).all(), 'Identidade de carteira inconsistente')
    values = edges[['volume_usd', 'n_trades', 'preco_medio']].to_numpy(float)
    require(np.isfinite(values).all() and (values >= 0).all(), 'Atributos invalidos')
    original = set(map(tuple, graph[FORWARD].edge_index.t().tolist()))
    all_original = set(map(tuple, split['all_positive'].t().tolist()))
    csv_pairs = set(zip(edges.wallet_id.map(ui), edges.condition_node_id.map(ei)))
    require(csv_pairs == all_original, 'CSV deve cobrir exatamente os pares da divisao')
    reserved = all_original - original
    neighbors = {i: set() for i in range(len(uids))}
    for u, e in original:
        neighbors[u].add(e)
    du = np.array([len(neighbors[i]) for i in range(len(uids))])
    de = np.bincount([e for _, e in original], minlength=len(eids))
    rng = np.random.default_rng(p['selection_seed'])
    candidates = sorted([i for i in np.flatnonzero(de >= p['min_market_degree']).tolist() if allowed_markets is None or eids[i] in allowed_markets], key=lambda i: eids[i])
    diagnostic = {'market_candidates': len(candidates), 'eligible_users': 0}
    if len(candidates) < p['n_markets']:
        raise Inviavel('Mercados elegiveis insuficientes', diagnostic)
    markets = set(map(int, rng.choice(candidates, p['n_markets'], replace=False)))
    eligible = sorted([u for u in neighbors if du[u] >= p['min_user_degree']
                       and math.ceil(p['concentration'] * int(du[u])) <= len(markets)
                       and len(neighbors[u] & markets) < math.ceil(p['concentration'] * int(du[u]))], key=lambda i: uids[i])
    if p['version'] == 'v2':
        from experimentos.piloto_sintetico.elegibilidade_v2 import selecionar_elegiveis
        eligible, capacity = selecionar_elegiveis(neighbors, du, markets, reserved, uids,
                                                  p['concentration'], p['min_user_degree'])
        for row in capacity['excluded_by_reserved_capacity']:
            row['reserved_target_market_ids'] = sorted(eids[e] for e in row.pop('reserved_target_market_indices'))
        diagnostic['eligibility_v2'] = capacity
    diagnostic.update(eligible_users=len(eligible), target_markets=sorted(eids[e] for e in markets))
    if len(eligible) < p['n_targets']:
        raise Inviavel('Carteiras elegiveis insuficientes', diagnostic)
    targets = set(map(int, rng.choice(eligible, p['n_targets'], replace=False)))
    target_order = sorted(targets, key=lambda i: uids[i])
    before = {u: len(neighbors[u] & markets) for u in targets}
    current = set(original)
    # Indices de linhas acompanham cada bloco ao longo das trocas.
    blocks = {}
    for row, (u, e) in enumerate(zip(edges.wallet_id.map(ui), edges.condition_node_id.map(ei))):
        blocks.setdefault((int(u), int(e)), []).append(row)
    altered = edges.copy(deep=True).reset_index(drop=True)
    log = []
    rng = np.random.default_rng(p['swap_seed'])
    attempts = 0
    # Lista com posicoes estaveis evita reconstruir todos os doadores por proposta.
    donors = sorted([(u, e) for u, e in current if u not in targets and e in markets],
                    key=lambda pair: (uids[pair[0]], eids[pair[1]]))
    for u in target_order:
        required = math.ceil(p['concentration'] * int(du[u]))
        while len(neighbors[u] & markets) < required:
            if attempts >= p['max_attempts'] or not donors:
                diagnostic.update(attempts=attempts, accepted=len(log), partial_swaps=log,
                                  target_wallets=[uids[x] for x in target_order])
                raise Inviavel('Nao foi possivel atingir a concentracao sem relaxar as regras', diagnostic)
            attempts += 1
            outside = sorted(neighbors[u] - markets, key=lambda i: eids[i])
            a = int(rng.choice(outside))
            idx = int(rng.integers(len(donors)))
            v, b = donors[idx]
            if (u, b) in current or (v, a) in current or (u, b) in reserved or (v, a) in reserved:
                continue
            left, right = blocks.pop((u, a)), blocks.pop((v, b))
            blocks[(v, a)], blocks[(u, b)] = left, right
            altered.loc[left, ['wallet_id', 'wallet']] = [uids[v], users.iloc[v].wallet]
            altered.loc[right, ['wallet_id', 'wallet']] = [uids[u], users.iloc[u].wallet]
            current.difference_update({(u, a), (v, b)})
            current.update({(u, b), (v, a)})
            neighbors[u].remove(a)
            neighbors[u].add(b)
            neighbors[v].remove(b)
            neighbors[v].add(a)
            donors.pop(idx)  # Toda troca consome uma conexao doadora no conjunto-alvo.
            log.append({'target_wallet_id': uids[u], 'donor_wallet_id': uids[v],
                        'outside_market_id': eids[a], 'target_market_id': eids[b],
                        'rows_target_to_donor': left, 'rows_donor_to_target': right})
    result = copy.deepcopy(split)
    pairs = torch.tensor(sorted(current), dtype=torch.long).t().contiguous()
    result['context'][FORWARD].edge_index = pairs
    result['context'][REVERSE].edge_index = pairs.flip(0).clone()
    result['all_positive'] = torch.unique(torch.cat([pairs] + [result[k] for k in PARTS], dim=1), dim=1)
    validar_divisao(result)
    require(len(current) == len(original) and not (current & reserved), 'Conectividade invalida')
    for axis, count in ((0, len(uids)), (1, len(eids))):
        require(torch.equal(torch.bincount(pairs[axis], minlength=count),
                            torch.bincount(graph[FORWARD].edge_index[axis], minlength=count)), 'Graus alterados')
    for k in PARTS:
        require(torch.equal(result[k], split[k]), f'Particao alterada: {k}')
    for kind in ('user', 'event'):
        require(torch.equal(result['context'][kind].original_id, graph[kind].original_id), 'IDs alterados')
    attrs = [c for c in edges.columns if c not in ('wallet_id', 'wallet')]
    pd.testing.assert_frame_equal(edges[attrs].reset_index(drop=True), altered[attrs])
    require(altered.wallet.eq(altered.wallet_id.map(dict(zip(uids, users.wallet)))).all(), 'Carteiras desalinhadas')
    require(set(zip(altered.wallet_id.map(ui), altered.condition_node_id.map(ei))) == current | reserved, 'CSV e grafo divergentes')
    donors_changed = {ui[row['donor_wallet_id']] for row in log}
    affected_markets = sorted({r[k] for r in log for k in ('outside_market_id', 'target_market_id')})
    labels = pd.DataFrame({'wallet_id': uids, 'alvo': [int(u in targets) for u in range(len(uids))],
                           'doadora': [u in donors_changed for u in range(len(uids))],
                           'elegivel': [u in set(eligible) for u in range(len(uids))],
                           'grau_contexto': du, 'sem_contexto': du == 0})
    labels['grupo'] = ['alvo' if u in targets else 'doadora' if u in donors_changed else
                       'isolada' if du[u] == 0 else 'intacta' for u in range(len(uids))]
    concentration = [{'wallet_id': uids[u], 'degree': int(du[u]), 'before': before[u],
                      'after': len(neighbors[u] & markets),
                      'required': math.ceil(p['concentration'] * int(du[u]))} for u in target_order]
    require(all(row['after'] >= row['required'] for row in concentration), 'Intensidade incompleta')
    diagnostic.update(attempts=attempts, accepted=len(log), concentration=concentration,
                      affected_markets=affected_markets, n_donors=len(donors_changed),
                      n_users=len(uids), n_events=len(eids), n_pairs=len(current), n_rows=len(edges),
                      removed_original_pairs=len(original - current),
                      checks={'degrees_equal': True, 'reserved_disjoint': True, 'parts_equal': True,
                              'attributes_equal_by_row': True, 'csv_graph_equal': True, 'ids_equal': True,
                              'intensity_complete': True, 'reverse_valid': True})
    return result, altered, pd.DataFrame(log), labels, diagnostic
