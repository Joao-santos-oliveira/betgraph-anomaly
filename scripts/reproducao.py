"""Prepara ou verifica a execução curta do Betgraph Anomaly (Linux/WSL)."""
import argparse
import hashlib
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

import pandas as pd
import torch
from baseline import ajustar_ranking
from construir_hetero import construir
from experimentos.investigacao_anomalias.relacional import scores_relacionais
from features_contexto import features_contexto
from preparar_features import FILES, preparar
from preparar_gnn import FORWARD, separar

INPUT = ROOT / 'data/reproducao/amostra'
REFERENCE = ROOT / 'resultados/reproducao/referencia.json'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inputs():
    return {name: sha(INPUT / name) for name in FILES.values()}


def sample():
    if INPUT.exists():
        if inputs() != json.loads(REFERENCE.read_text())['entradas_sha256']:
            raise ValueError(f'Confira os arquivos existentes em {INPUT}')
        print('Amostra existente conferida.')
        return
    INPUT.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=INPUT.parent) as temp:
        folder = Path(temp)
        preparar(ROOT / 'data/raw', folder, sample_size=256, seed=42)
        ready = folder / 'amostra'
        ready.mkdir()
        for kind, name in FILES.items():
            (folder / f'{kind}.csv').rename(ready / name)
        ready.rename(INPUT)


def run():
    torch.set_num_threads(2)
    folder = Path(mkdtemp(prefix='execucao-', dir=INPUT.parent))
    preparar(INPUT, folder, sample_size=0, seed=42)
    graph = construir(folder, ROOT)
    split = separar(graph, seed=42)
    frames = [pd.read_csv(folder / f'{kind}.csv') for kind in ('user', 'event', 'edge')]
    base, _, _, info = features_contexto(split, *frames)
    context = base['context']
    ids = pd.DataFrame({'wallet_id': context['user'].original_id.numpy()})
    stats = scores_relacionais(context[FORWARD].edge_index.numpy(), len(ids), context['event'].num_nodes)
    relational = ids.assign(score_anomalia=stats['surpresa_coparticipacao']).sort_values(
        ['score_anomalia', 'wallet_id'], ascending=[False, True])
    _, forest, _ = ajustar_ranking(ids, context['user'].x.numpy(), seed=42)
    result = {'entradas_sha256': inputs(), 'carteiras': len(ids), 'mercados': len(frames[1]),
              'linhas': len(frames[2]), 'pares': split['all_positive'].shape[1],
              'pares_contexto': info['n_pares_contexto'], 'ordem_sha256': {}}
    for method, ranking in [('relacional', relational), ('floresta', forest)]:
        ranking[['wallet_id','score_anomalia']].to_csv(folder / f'{method}.csv', index=False)
        ordered = '\n'.join(str(i) for i in ranking.wallet_id)
        result['ordem_sha256'][method] = hashlib.sha256(ordered.encode('ascii')).hexdigest()
    return folder, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('acao', choices=['amostra', 'verificar'])
    action = parser.parse_args().acao
    if action == 'amostra':
        sample()
        return
    expected = json.loads(REFERENCE.read_text())
    if not INPUT.is_dir():
        parser.exit(2, 'Extraia o pacote de reprodução ou execute make amostra com os CSVs tratados.\n')
    if inputs() != expected['entradas_sha256']:
        parser.exit(2, 'Os três CSVs da amostra devem corresponder aos hashes da referência.\n')
    folder, result = run()
    different = [k for k in expected if result.get(k) != expected[k]]
    receipt = {'resultado': result, 'divergencias': different, 'python': platform.python_version(),
               'plataforma': platform.platform(), 'versoes': {n:version(n) for n in
                ('torch','torch-geometric','scikit-learn','pandas','numpy','scipy')}}
    path = folder / 'verificacao.json'
    path.write_text(json.dumps(receipt, indent=2) + '\n')
    print('Relatório:', path)
    if different:
        parser.exit(1, f'Divergências: {different}\n')
    print('Contagens e ordem integral dos dois rankings conferidas.')


if __name__ == '__main__':
    main()
