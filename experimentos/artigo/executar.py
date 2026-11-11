"""Reproduz a análise principal em uma pasta exclusiva a partir dos CSVs brutos."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import shutil
from tempfile import mkdtemp

from experimentos.artigo.filtrar import ROOT, filtrar
from experimentos.artigo.preparar import FORWARD, preparar, protocol
from experimentos.artigo.ablacao import avaliar
from experimentos.gnn_v2.detectores import hibrido, ranking
from experimentos.gnn_v2.treino import treinar
from construir_hetero import construir
from preparar_features import preparar as preparar_csv
import torch


def treinar_cenario(task):
    folder, seed = task
    split = torch.load(folder / 'rede.pt', weights_only=False)
    graph = split['context']
    dest = folder / f'v2_{seed}'
    z, report = treinar(split, dest, 'gnn', seed, contrast=True)
    assert report['teste_avaliado'] is False
    scores = hibrido(graph[FORWARD].edge_index.numpy(), graph['user'].num_nodes,
                     graph['event'].num_nodes, z)
    ranking(graph['user'].original_id.numpy(), scores).to_csv(dest / 'ranking.csv', index=False)


def executar(raw, destino, processos):
    destino.mkdir(parents=True, exist_ok=True)
    run = Path(mkdtemp(dir=destino.resolve(), prefix='artigo-'))
    print(f'Execução em {run}', flush=True)
    for mapping in ROOT.glob('mapeamento_*.json'):
        shutil.copy2(mapping, run / mapping.name)
    (run / 'src').symlink_to(ROOT / 'src', target_is_directory=True)
    manifest = json.loads((ROOT / 'resultados/entradas.json').read_text())
    filtrar(run, raw.resolve(), manifest)
    source = run / 'dia'
    preparar_csv(run / 'data/raw', source, sample_size=0, seed=42)
    construir(source, run)
    preparar(source, run)
    folders = ['original', *map(str, protocol['realizacoes'])]
    tasks = [(run / folder, seed) for folder in folders for seed in protocol['seeds_modelo']]
    with ProcessPoolExecutor(max_workers=processos, mp_context=multiprocessing.get_context('spawn')) as pool:
        list(pool.map(treinar_cenario, tasks))
    avaliar(run, run / 'resultados')
    print(run / 'resultados', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--destino', type=Path, required=True)
    parser.add_argument('--processos', type=int, choices=[1, 2, 3], default=1)
    args = parser.parse_args()
    executar(args.raw, args.destino, args.processos)
