"""Prepara features reproduziveis sem construir ou treinar a GNN."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
FILES = {'user': 'betgraph_wallets_16set_nos.csv',
         'event': 'betgraph_nos_evento_16set_final.csv',
         'edge': 'betgraph_arestas_16set_final.csv'}
USER_LOG = ['volume_total_usd', 'n_trades_total', 'n_mercados_distintos', 'ticket_medio_geral']


def log_positivo(frame, columns):
    """Aplica log1p somente em medidas finitas e nao negativas."""
    values = frame[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f'Valores invalidos em {columns}')
    return np.log1p(values)


def preparar(raw, output, sample_size=5000, seed=42, preprocessador=None):
    """Seleciona carteiras e salva matrizes, metadados e transformadores."""
    frames = {k: pd.read_csv(raw / v) for k, v in FILES.items()}
    u, e, a = (frames[k] for k in ['user', 'event', 'edge'])
    counts = {k: len(v) for k, v in frames.items()}
    for df, col in [(u, 'wallet_id'), (e, 'condition_node_id')]:
        if df[col].isna().any() or df[col].duplicated().any():
            raise ValueError(f'ID ausente ou duplicado: {col}')
    if not a.wallet_id.isin(u.wallet_id).all() or not a.condition_node_id.isin(e.condition_node_id).all():
        raise ValueError('Arestas orfas na entrada')
    if sample_size < 0:
        raise ValueError('sample-size deve ser >= 0; 0 seleciona todas as carteiras')
    u = u.sort_values('wallet_id')
    if sample_size and sample_size < len(u):
        u = u.sample(n=sample_size, random_state=seed)
        a = a[a.wallet_id.isin(u.wallet_id)].copy()
        e = e[e.condition_node_id.isin(a.condition_node_id)].copy()
    u = u.sort_values('wallet_id').reset_index(drop=True)
    e = e.sort_values('condition_node_id').reset_index(drop=True)
    a = a.sort_values(['wallet_id', 'condition_node_id', 'token_outcome_id']).reset_index(drop=True)
    if min(len(u), len(e), len(a)) == 0:
        raise ValueError('Conjunto vazio')
    missing = e.duracao_mercado_segundos.isna() | ~e.duracao_mercado_valida | (e.duracao_mercado_segundos < 0)
    if preprocessador:
        bundle = joblib.load(preprocessador)
    else:
        median = e.loc[~missing, 'duracao_mercado_segundos'].median()
        if not np.isfinite(median):
            raise ValueError('Sem duracoes validas para calcular mediana')
        bundle = {'mediana_duracao': float(median), 'scalers': {},
                  'categorias': OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float32)}
        bundle['categorias'].fit(e[['categoria_principal']])
    duration = e.duracao_mercado_segundos.mask(missing, bundle['mediana_duracao'])
    time = pd.to_datetime(a.primeiro_trade, utc=True, errors='raise')
    if time.isna().any() or time.dt.normalize().nunique() != 1:
        raise ValueError('Esperado um unico dia de trades validos')
    # Hora UTC dentro do dia observado, nao idade nem tempo entre trades.
    time_fraction = (time - time.dt.normalize()).dt.total_seconds().to_numpy() / 86400
    numeric = {
        'user': np.column_stack([log_positivo(u, USER_LOG), u.preco_medio_ponderado]),
        'event': np.log1p(duration.to_numpy(dtype=float)).reshape(-1, 1),
        'edge': np.column_stack([log_positivo(a, ['volume_usd']), a.preco_medio, time_fraction]),
    }
    arrays = {}
    for key, values in numeric.items():
        if not np.isfinite(values).all():
            raise ValueError(f'Feature nao finita: {key}')
        if not preprocessador:
            bundle['scalers'][key] = StandardScaler().fit(values)
        arrays[key] = bundle['scalers'][key].transform(values).astype(np.float32)
    arrays['event'] = np.column_stack([arrays['event'], missing.astype(np.float32),
                                      bundle['categorias'].transform(e[['categoria_principal']])]).astype(np.float32)
    feature_names = {
        'user': ['log1p_' + c for c in USER_LOG] + ['preco_medio_ponderado'],
        'event': ['log1p_duracao_mercado_segundos', 'duracao_ausente'] + bundle['categorias'].get_feature_names_out().tolist(),
        'edge': ['log1p_volume_usd', 'preco_medio', 'primeiro_trade_fracao_dia_utc'],
    }
    if not all(np.isfinite(v).all() for v in arrays.values()):
        raise ValueError('Matriz final contem NaN ou infinito')
    output.mkdir(parents=True, exist_ok=True)
    for key, df in [('user', u), ('event', e), ('edge', a)]:
        df.to_csv(output / f'{key}.csv', index=False)
    np.savez_compressed(output / 'features.npz', **arrays)
    bundle['feature_names'] = feature_names
    joblib.dump(bundle, output / 'preprocessador.joblib')
    report = {'seed': seed, 'sample_size': sample_size, 'entrada': counts,
              'selecionados': {'user': len(u), 'event': len(e), 'edge': len(a)},
              'duracoes_ausentes_entrada': int(frames['event'].duracao_mercado_segundos.isna().sum()),
              'duracoes_imputadas_selecao': int(missing.sum()),
              'mediana_duracao': bundle['mediana_duracao'], 'features': feature_names,
              'ajuste': str(preprocessador) if preprocessador else 'transdutivo no subgrafo selecionado'}
    (output / 'features_manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=ROOT / 'data/raw')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/processed/dev')
    parser.add_argument('--sample-size', type=int, default=5000, help='0 para conjunto completo')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--preprocessador', type=Path, help='Reutiliza scaler, mediana e categorias ja ajustados')
    args = parser.parse_args()
    preparar(args.raw, args.output, args.sample_size, args.seed, args.preprocessador)


if __name__ == '__main__':
    main()
