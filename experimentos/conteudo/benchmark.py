"""Benchmark por etapa do pipeline do dia real (make benchmark). Só depende da máquina aqui.

Cada etapa roda em um processo próprio; os.wait4 devolve o tempo de parede e o pico de
memória residente (ru_maxrss) daquele processo. O pipeline inteiro se repete N vezes, cada
vez numa pasta nova, apagada ao fim. Grava em benchmark/:
  etapas.csv      uma linha por etapa e repetição
  resumo.csv      mediana, mínimo e máximo por etapa
  hardware.json   CPU, memória, sistema, versões e threads
  protocolo.json  repetições, seeds, threads e commit

Só roda em Linux (os.wait4); no Windows, use o WSL.
Uso: python -m experimentos.conteudo.benchmark --raw data/raw [--repeticoes 3]
"""
import argparse
from datetime import datetime, timezone
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from tempfile import mkdtemp

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SEED_SCORES = 42
THREADS = {'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'}


def etapas(seeds):
    lista = [('filtrar', None, 'Conferir e filtrar CSVs brutos'), ('atributos', None, 'Preparar atributos'),
             ('grafo', None, 'Construir grafo heterogêneo'), ('divisao', None, 'Dividir ligações e intervenções')]
    lista += [('treino', s, f'Treinar GNN híbrida (seed {s})') for s in seeds]
    lista += [('relacional', None, 'Score relacional direto'),
              ('hibrido', SEED_SCORES, f'Score GNN híbrida (seed {SEED_SCORES})'),
              ('similaridade', SEED_SCORES, f'Score similaridade isolada (seed {SEED_SCORES})'),
              ('cobertura', None, 'Cobertura total e teste reservado')]
    return lista


def executar_etapa(nome, run, seed, raw):
    """Roda uma etapa dentro do processo filho."""
    import numpy as np
    import torch
    from experimentos.artigo.executar import treinar_cenario
    from experimentos.artigo.filtrar import filtrar
    from experimentos.artigo.preparar import FORWARD, preparar
    from experimentos.artigo.cobertura import cobertura
    from experimentos.gnn.detectores import hibrido
    from experimentos.relacional import scores_relacionais
    from construir_hetero import construir
    from preparar_features import preparar as preparar_csv

    if nome == 'filtrar':
        filtrar(run, raw, json.loads((ROOT / 'resultados/entradas.json').read_text()))
    elif nome == 'atributos':
        preparar_csv(run / 'data/raw', run / 'dia', sample_size=0, seed=42)
    elif nome == 'grafo':
        construir(run / 'dia', run)
    elif nome == 'divisao':
        preparar(run / 'dia', run)
    elif nome == 'treino':
        treinar_cenario((run / 'original', seed))
    elif nome == 'cobertura':
        cobertura(run, run / 'resultados')
    else:
        graph = torch.load(run / 'original' / 'rede.pt', weights_only=False)['context']
        pairs = graph[FORWARD].edge_index.numpy()
        dims = (graph['user'].num_nodes, graph['event'].num_nodes)
        if nome == 'relacional':
            scores_relacionais(pairs, *dims)
        else:
            with np.load(run / 'original' / f'gnn_{seed}' / 'embeddings.npz') as emb:
                z = emb['user']
            hibrido(pairs, *dims, z, apenas_similaridade=nome == 'similaridade')


def medir(nome, run, seed, raw):
    comando = [sys.executable, '-B', '-m', 'experimentos.conteudo.benchmark', '--etapa', nome, '--rodada', str(run),
               '--raw', str(raw)] + (['--seed', str(seed)] if seed is not None else [])
    inicio = time.perf_counter()
    processo = subprocess.Popen(comando, cwd=ROOT, env={**os.environ, **THREADS}, stdout=subprocess.DEVNULL)
    _, status, uso = os.wait4(processo.pid, 0)
    segundos = time.perf_counter() - inicio
    codigo = os.waitstatus_to_exitcode(status)
    if codigo:
        raise subprocess.CalledProcessError(codigo, comando)
    return dict(segundos=segundos, rss_mib=uso.ru_maxrss / 1024, cpu_usuario_s=uso.ru_utime, cpu_sistema_s=uso.ru_stime)


def _saida(comando):
    try:
        return subprocess.check_output(comando, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def hardware(destino):
    info = {'data_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'sistema': platform.platform(),
            'kernel': platform.release(), 'maquina': platform.machine(), 'python': sys.version.split()[0],
            'nucleos_logicos': os.cpu_count(), 'threads_por_processo': THREADS}
    cpu = _saida(['lscpu'])
    if cpu:
        campos = dict(l.split(':', 1) for l in cpu.splitlines() if ':' in l)
        info['cpu'] = {k: campos[k].strip() for k in ('Model name', 'Thread(s) per core', 'Core(s) per socket',
                                                      'Socket(s)', 'CPU max MHz') if k in campos}
    try:
        mem = dict(l.split(':', 1) for l in Path('/proc/meminfo').read_text().splitlines())
        info['memoria_total_mib'] = int(mem['MemTotal'].split()[0]) / 1024
        info['swap_total_mib'] = int(mem['SwapTotal'].split()[0]) / 1024
    except (OSError, KeyError):
        pass
    info['wsl'] = 'microsoft' in platform.release().lower()
    if info['wsl']:
        host = _saida(['powershell.exe', '-NoProfile', '-Command', '(Get-CimInstance Win32_Processor).Name; '
                       '[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1MB)'])
        if host:
            linhas = [l.strip() for l in host.splitlines() if l.strip()]
            info['host_windows'] = {'cpu': linhas[0] if linhas else None,
                                    'memoria_fisica_mib': float(linhas[1]) if len(linhas) > 1 else None}
    info['disco_destino'] = _saida(['df', '-T', '-h', str(destino)])
    versoes = {}
    for pacote in ('numpy', 'pandas', 'scipy', 'sklearn', 'torch', 'torch_geometric'):
        try:
            versoes[pacote] = __import__(pacote).__version__
        except ImportError:
            versoes[pacote] = None
    info['versoes'] = versoes
    return info


def benchmark(raw, destino_execucoes, repeticoes, saida):
    from experimentos.artigo.preparar import protocol
    seeds = protocol['seeds_modelo']
    destino_execucoes.mkdir(parents=True, exist_ok=True)
    saida.mkdir(parents=True, exist_ok=True)
    registros = []
    for r in range(1, repeticoes + 1):
        run = Path(mkdtemp(dir=destino_execucoes.resolve(), prefix='benchmark-'))
        for mapping in ROOT.glob('mapeamento_*.json'):
            shutil.copy2(mapping, run / mapping.name)
        (run / 'src').symlink_to(ROOT / 'src', target_is_directory=True)
        inicio = time.perf_counter()
        try:
            for nome, seed, rotulo in etapas(seeds):
                m = medir(nome, run, seed, raw.resolve())
                registros.append(dict(repeticao=r, etapa=nome if seed is None else f'{nome}_{seed}', rotulo=rotulo,
                                      seed=seed, **m))
                print(f'[r{r}] {rotulo}: {m["segundos"]:.1f} s, {m["rss_mib"]:.0f} MiB', flush=True)
            registros.append(dict(repeticao=r, etapa='total', rotulo='Pipeline completo',
                                  segundos=time.perf_counter() - inicio))
        finally:
            shutil.rmtree(run, ignore_errors=True)
    etapas_df = pd.DataFrame(registros)
    etapas_df.to_csv(saida / 'etapas.csv', index=False, lineterminator='\n')
    ordem = list(dict.fromkeys(etapas_df.etapa))
    resumo = etapas_df.groupby(['etapa', 'rotulo'], sort=False).agg(
        repeticoes=('repeticao', 'nunique'), segundos_mediana=('segundos', 'median'), segundos_min=('segundos', 'min'),
        segundos_max=('segundos', 'max'), rss_mib_mediana=('rss_mib', 'median'), rss_mib_min=('rss_mib', 'min'),
        rss_mib_max=('rss_mib', 'max')).reset_index()
    resumo['etapa'] = pd.Categorical(resumo.etapa, ordem)
    resumo.sort_values('etapa').to_csv(saida / 'resumo.csv', index=False, lineterminator='\n')
    (saida / 'hardware.json').write_text(json.dumps(hardware(destino_execucoes), indent=2, ensure_ascii=False) + '\n',
                                         encoding='utf-8')
    commit = _saida(['git', '-C', str(ROOT), 'rev-parse', '--short', 'HEAD'])
    alterado = bool(_saida(['git', '-C', str(ROOT), 'status', '--porcelain', '--untracked-files=no']))
    (saida / 'protocolo.json').write_text(json.dumps(dict(
        commit=commit, alteracoes_nao_commitadas=alterado, repeticoes=repeticoes, seeds_modelo=seeds,
        seed_scores=SEED_SCORES, threads=THREADS, medicao='os.wait4 por etapa: tempo de parede e ru_maxrss',
        pasta_por_repeticao='nova a cada repetição, apagada ao fim'), indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8')
    print(resumo.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=ROOT / 'data' / 'raw')
    parser.add_argument('--destino', type=Path, default=ROOT / 'data' / 'benchmark')
    parser.add_argument('--saida', type=Path, default=ROOT / 'benchmark')
    parser.add_argument('--repeticoes', type=int, default=3)
    parser.add_argument('--etapa')
    parser.add_argument('--rodada', type=Path)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    if args.etapa:
        executar_etapa(args.etapa, args.rodada, args.seed, args.raw)
    else:
        benchmark(args.raw, args.destino, args.repeticoes, args.saida)
