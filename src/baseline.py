"""Baseline nao supervisionado sobre features de carteira, sem usar o grafo."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest

from preparar_features import ROOT, USER_LOG, log_positivo


def carregar_entrada(directory):
    """Confere schema e alinhamento entre carteiras e features salvas."""
    users = pd.read_csv(directory / 'user.csv')
    with np.load(directory / 'features.npz') as archive:
        x = archive['user'].copy()
    prep = joblib.load(directory / 'preprocessador.joblib')
    names = ['log1p_' + c for c in USER_LOG] + ['preco_medio_ponderado']
    if prep['feature_names']['user'] != names:
        raise ValueError('Schema de features incompativel')
    if len(users) < 2 or x.shape != (len(users), len(names)):
        raise ValueError('Formato de features invalido ou menos de duas carteiras')
    for col in ['wallet_id', 'wallet']:
        if users[col].isna().any() or users[col].duplicated().any():
            raise ValueError(f'Identificador ausente ou duplicado: {col}')
    expected = prep['scalers']['user'].transform(np.column_stack([
        log_positivo(users, USER_LOG), users.preco_medio_ponderado])).astype(np.float32)
    if not np.isfinite(x).all() or not np.allclose(x, expected, rtol=1e-6, atol=1e-6):
        raise ValueError('Features invalidas ou desalinhadas com user.csv')
    return users, x, names


def ajustar_ranking(users, x, seed=42, trees=100, top_fraction=0.01):
    """Ajusta a floresta e ordena scores; top nao e rotulo de fraude."""
    if not 0 < top_fraction <= 1 or trees < 1:
        raise ValueError('top-fraction deve estar em (0, 1] e trees deve ser positivo')
    model = IsolationForest(n_estimators=trees, max_samples='auto',
                            contamination='auto', random_state=seed, n_jobs=1)
    start = perf_counter()
    model.fit(x)
    fit_seconds = perf_counter() - start
    start = perf_counter()
    # No sklearn, score_samples menor indica maior anomalia. Invertemos o sinal.
    scores = -model.score_samples(x)
    scoring_seconds = perf_counter() - start
    if not np.isfinite(scores).all():
        raise ValueError('Score nao finito')
    ranking = users.copy()
    ranking['score_anomalia'] = scores
    ranking = ranking.sort_values(['score_anomalia', 'wallet_id'], ascending=[False, True]).reset_index(drop=True)
    ranking.insert(0, 'posicao', np.arange(1, len(ranking) + 1))
    k = math.ceil(len(ranking) * top_fraction)
    ranking['selecionada_top'] = ranking.posicao <= k
    times = {'ajuste_segundos': fit_seconds, 'scoring_segundos': scoring_seconds}
    return model, ranking, times


def executar(directory, output, seed=42, trees=100, top_fraction=0.01):
    """Salva ranking, top, modelo e manifesto reproduzivel."""
    users, x, names = carregar_entrada(directory)
    model, ranking, times = ajustar_ranking(users, x, seed, trees, top_fraction)
    output.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(output / 'ranking.csv', index=False)
    ranking.loc[ranking.selecionada_top].to_csv(output / 'top.csv', index=False)
    joblib.dump(model, output / 'modelo.joblib')
    hashes = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
              for name in ['user.csv', 'features.npz', 'preprocessador.joblib', 'features_manifest.json']}
    manifest = {
        'metodo': 'IsolationForest', 'sklearn': sklearn.__version__,
        'parametros': model.get_params(), 'max_samples_efetivo': int(model.max_samples_),
        'entrada': str(directory.resolve()), 'sha256_entrada': hashes,
        'features': names, 'n_carteiras': len(ranking),
        'top_fraction': top_fraction, 'n_top': int(ranking.selecionada_top.sum()),
        'score': '-score_samples; maior significa mais anomalo; nao e probabilidade',
        'desempate': 'wallet_id crescente; top usa ceil(n * fracao)',
        'protocolo': 'ajuste e scoring nas mesmas carteiras; exploracao transdutiva sem rotulos',
        'tempos': times,
        'quantis_score': {str(q): float(ranking.score_anomalia.quantile(q)) for q in [0, .25, .5, .75, .99, 1]},
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'n_carteiras': len(ranking), 'n_top': manifest['n_top'], **times}, indent=2))
    return ranking


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'data/processed/dev')
    parser.add_argument('--output', type=Path, help='Padrao: INPUT/baseline_seed_SEED')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--trees', type=int, default=100)
    parser.add_argument('--top-fraction', type=float, default=.01)
    args = parser.parse_args()
    output = args.output or args.input / f'baseline_seed_{args.seed}'
    executar(args.input, output, args.seed, args.trees, args.top_fraction)


if __name__ == '__main__':
    main()
