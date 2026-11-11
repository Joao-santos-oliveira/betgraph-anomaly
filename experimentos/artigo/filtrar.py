"""Reconstrói uma execução isolada a partir dos dois CSVs publicados."""
import os
import shutil
import subprocess
import sys
from experimentos.comum import ROOT, sha

ENV = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')


def conferir(folder, hashes):
    for name, expected in hashes.items():
        if not (folder/name).is_file() or sha(folder/name) != expected:
            raise ValueError(f'Arquivo ausente ou diferente da edição documentada: {folder/name}')


def python(repo, *args):
    subprocess.run([sys.executable, '-B', *map(str,args)], cwd=repo, env=ENV, check=True)


def filtrar(repo, raw, manifest):
    destino = repo/'data/raw'
    destino.mkdir(parents=True)
    for name in manifest['brutos']:
        shutil.copyfile(raw/name,destino/name)
    conferir(destino, manifest['brutos'])
    edge='data/raw/betgraph_arestas_16set'
    event='data/raw/betgraph_nos_evento_16set'
    python(repo,'src/processar_graficos_arestas.py',edge+'.csv')
    python(repo,'src/processar_graficos_nos.py',event+'.csv')
    python(repo,'src/filtrar_recorrentes.py',event+'_processado.csv',edge+'_processado.csv')
    python(repo,'src/processar_graficos_wallets.py',edge+'_final.csv')
    conferir(destino,manifest['tratados'])
