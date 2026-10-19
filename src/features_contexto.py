"""Recalcula features usando apenas arestas de contexto, preservando todos os nos."""
import copy

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from preparar_features import USER_LOG, log_positivo
from preparar_gnn import FORWARD
from treinar_gnn import validar_divisao


def alinhar(frame, column, ids):
    """Reordena pela identidade original; nunca assume ordem incidental do CSV."""
    if frame[column].isna().any() or frame[column].duplicated().any():
        raise ValueError('Identificadores ausentes ou duplicados')
    if set(frame[column]) != set(ids.tolist()):
        raise ValueError('IDs do CSV e do grafo diferem')
    return frame.set_index(column).loc[ids.tolist()].reset_index()


def features_contexto(split, users, events, edges):
    """Agrega todos os outcomes dos pares permitidos; ajusta somente nos ativos."""
    validar_divisao(split)
    result = copy.deepcopy(split)
    graph = result['context']
    users = alinhar(users, 'wallet_id', graph['user'].original_id)
    events = alinhar(events, 'condition_node_id', graph['event'].original_id)
    pairs = graph[FORWARD].edge_index.numpy()
    allowed = pd.MultiIndex.from_arrays([users.wallet_id.to_numpy()[pairs[0]],
                                        events.condition_node_id.to_numpy()[pairs[1]]])
    keys = pd.MultiIndex.from_frame(edges[['wallet_id', 'condition_node_id']])
    selected = edges.loc[keys.isin(allowed)].copy()
    if len(pd.MultiIndex.from_frame(selected[['wallet_id', 'condition_node_id']]).unique()) != len(allowed):
        raise ValueError('CSV nao cobre todos os pares de contexto')
    values = selected[['volume_usd', 'n_trades', 'preco_medio']].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Agregados de contexto invalidos')
    selected['preco_x_volume'] = selected.preco_medio * selected.volume_usd
    aggregate = selected.groupby('wallet_id').agg(
        volume_total_usd=('volume_usd', 'sum'), n_trades_total=('n_trades', 'sum'),
        n_mercados_distintos=('condition_node_id', 'nunique'),
        preco_x_volume=('preco_x_volume', 'sum'), preco_simples=('preco_medio', 'mean'))
    active = users.wallet_id.isin(aggregate.index).to_numpy()
    aggregate = aggregate.reindex(users.wallet_id)
    volume = aggregate.volume_total_usd.fillna(0)
    trades = aggregate.n_trades_total.fillna(0)
    price = (aggregate.preco_x_volume / volume.replace(0, np.nan)).fillna(aggregate.preco_simples)
    median_price = float(price.loc[active].median())
    if not np.isfinite(median_price):
        raise ValueError('Sem preco observado para imputacao')
    raw = pd.DataFrame({'wallet_id': users.wallet_id.to_numpy(),
                        'volume_total_usd': volume.to_numpy(),
                        'n_trades_total': trades.to_numpy(),
                        'n_mercados_distintos': aggregate.n_mercados_distintos.fillna(0).to_numpy(),
                        'ticket_medio_geral': (volume / trades.replace(0, np.nan)).fillna(0).to_numpy(),
                        'preco_medio_ponderado': price.fillna(median_price).to_numpy(),
                        'sem_contexto': ~active})
    numeric = np.column_stack([log_positivo(raw, USER_LOG), raw.preco_medio_ponderado])
    user_scaler = StandardScaler().fit(numeric[active])
    ux = np.column_stack([user_scaler.transform(numeric), ~active]).astype(np.float32)
    event_active = np.bincount(pairs[1], minlength=len(events)) > 0
    missing = (events.duracao_mercado_segundos.isna() | ~events.duracao_mercado_valida |
               (events.duracao_mercado_segundos < 0))
    duration_median = float(events.loc[event_active & ~missing, 'duracao_mercado_segundos'].median())
    if not np.isfinite(duration_median):
        raise ValueError('Sem duracao valida no contexto')
    duration = np.log1p(events.duracao_mercado_segundos.mask(missing, duration_median).to_numpy()).reshape(-1, 1)
    event_scaler = StandardScaler().fit(duration[event_active])
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float32)
    encoder.fit(events.loc[event_active, ['categoria_principal']])
    ex = np.column_stack([event_scaler.transform(duration), missing,
                          encoder.transform(events[['categoria_principal']])]).astype(np.float32)
    if not np.isfinite(ux).all() or not np.isfinite(ex).all():
        raise ValueError('Features nao finitas')
    graph['user'].x, graph['event'].x = torch.from_numpy(ux), torch.from_numpy(ex)
    bundle = {'user_scaler': user_scaler, 'event_scaler': event_scaler, 'categorias': encoder,
              'mediana_preco': median_price, 'mediana_duracao': duration_median,
              'user_features': ['log1p_' + name for name in USER_LOG] + ['preco_medio_ponderado', 'sem_contexto']}
    report = {'n_arestas_agregadas_contexto': len(selected), 'n_pares_contexto': len(allowed),
              'carteiras_sem_contexto': int((~active).sum()), 'mercados_sem_contexto': int((~event_active).sum()),
              'ajuste': 'scalers, medianas e categorias ajustados apenas nos nos ativos no contexto',
              'sem_contexto': 'contagens, volume e ticket zero; preco mediano dos ativos; flag binaria adicional',
              'preco': 'ponderado por volume; media simples dos precos das arestas quando volume zero',
              'limite': 'metadados dos mercados permanecem transdutivos; nao ha separacao temporal'}
    return result, raw, bundle, report
