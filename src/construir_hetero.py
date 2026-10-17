"""Monta HeteroData com indices consecutivos e relacao reversa."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected

ROOT = Path(__file__).resolve().parents[1]


def construir(directory, mappings):
    """Valida IDs persistentes, monta o grafo e salva indices e diagnostico."""
    u, e, a = [pd.read_csv(directory / f'{k}.csv') for k in ['user', 'event', 'edge']]
    with np.load(directory / 'features.npz') as archive:
        features = {k: archive[k].copy() for k in archive.files}
    for name, df, source, target in [
        ('wallets', u, 'wallet', 'wallet_id'),
        ('conditions', e, 'condition_id', 'condition_node_id'),
        ('wallets', a, 'wallet', 'wallet_id'),
        ('conditions', a, 'condition_id', 'condition_node_id'),
        ('outcomes', a, 'token_outcome', 'token_outcome_id'),
    ]:
        mapping = json.loads((mappings / f'mapeamento_{name}.json').read_text())
        if not df[source].map(mapping).eq(df[target]).all():
            raise ValueError(f'IDs inconsistentes com mapeamento_{name}.json')
    indices = {}
    data = HeteroData()
    for key, df, col in [('user', u, 'wallet_id'), ('event', e, 'condition_node_id')]:
        if df[col].isna().any() or df[col].duplicated().any():
            raise ValueError(f'IDs invalidos: {col}')
        indices[key] = {int(original): index for index, original in enumerate(df[col])}
        data[key].x = torch.from_numpy(features[key])
        data[key].original_id = torch.tensor(df[col].to_numpy(), dtype=torch.long)
        if data[key].x.shape[0] != len(df):
            raise ValueError('Contagem de features divergente')
    src = a.wallet_id.map(indices['user'])
    dst = a.condition_node_id.map(indices['event'])
    if src.isna().any() or dst.isna().any():
        raise ValueError('Aresta orfa no subgrafo')
    relation = ('user', 'bets_on', 'event')
    data[relation].edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    data[relation].edge_attr = torch.from_numpy(features['edge'])
    data[relation].outcome_id = torch.tensor(a.token_outcome_id.to_numpy(), dtype=torch.long)
    if data[relation].edge_attr.shape[0] != len(a):
        raise ValueError('Contagem de atributos de aresta divergente')
    for values in features.values():
        if not np.isfinite(values).all():
            raise ValueError('NaN ou infinito nas features')
    data = ToUndirected()(data)
    data.validate(raise_on_error=True)
    reverse = ('event', 'rev_bets_on', 'user')
    assert torch.equal(data[reverse].edge_index, data[relation].edge_index.flip(0))
    assert torch.equal(data[reverse].edge_attr, data[relation].edge_attr)
    report = {'users': len(u), 'events': len(e), 'arestas_diretas': len(a),
              'arestas_reversas': data[reverse].num_edges,
              'grau_medio_user_com_multiarestas': len(a) / len(u),
              'grau_medio_event_com_multiarestas': len(a) / len(e),
              'pares_user_event_distintos': len(a[['wallet_id', 'condition_node_id']].drop_duplicates())}
    torch.save(data, directory / 'hetero.pt')
    (directory / 'indices_originais.json').write_text(json.dumps(indices, indent=2), encoding='utf-8')
    (directory / 'grafo_manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(data)
    print(json.dumps(report, indent=2))
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'data/processed/dev')
    parser.add_argument('--mappings', type=Path, default=ROOT)
    args = parser.parse_args()
    construir(args.input, args.mappings)


if __name__ == '__main__':
    main()
