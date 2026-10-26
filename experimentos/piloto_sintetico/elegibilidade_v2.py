"""Elegibilidade v2: capacidade individual descontando positivos reservados."""
import math


def selecionar_elegiveis(neighbors, degrees, markets, reserved, original_ids, concentration, min_degree):
    """Conserva a ordem por ID e registra as exclusoes anteriores ao sorteio."""
    blocked = {}
    for user, market in reserved:
        if market in markets:
            blocked.setdefault(user, set()).add(market)
    eligible, excluded = [], []
    previous_count = 0
    for user in sorted(neighbors, key=lambda u: original_ids[u]):
        degree = int(degrees[user])
        required = math.ceil(concentration * degree)
        before = len(neighbors[user] & markets)
        if degree < min_degree or required > len(markets) or before >= required:
            continue
        previous_count += 1
        forbidden = blocked.get(user, set())
        allowed = len(markets) - len(forbidden)
        if required > allowed:
            excluded.append({'wallet_id': int(original_ids[user]), 'degree': degree,
                             'required': required, 'allowed': allowed,
                             'reserved_target_market_indices': sorted(forbidden)})
        else:
            eligible.append(user)
    return eligible, {'eligible_before_capacity_check': previous_count,
                      'excluded_by_reserved_capacity': excluded,
                      'capacity_rule': 'ceil(concentration * degree) <= n_markets - reserved_target_markets',
                      'note': 'Condicao necessaria; nao garante viabilidade conjunta das trocas'}
