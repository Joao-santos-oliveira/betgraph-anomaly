"""Benchmark por etapa: método principal e comparadores na mesma máquina e no mesmo recorte.

Executa `make comparar ESCOPO=5000` de cada branch (v3: relacional, IsolationForest,
GNN v3 e MLP; v2: híbrido GNN v2) a partir dos dois CSVs brutos, sem alterar as
branches. O script importa o `scripts/pipeline.py` de cada uma, a partir de worktrees
temporárias dos commits fixados, e troca só a função que dispara cada etapa por uma
versão medida: cada etapa roda em um processo próprio, e `os.wait4` devolve o tempo
de parede e o pico de memória residente (ru_maxrss) daquele processo.

Repete o pipeline inteiro N vezes, alternando as branches, e grava em `benchmark/`:
  etapas.csv          uma linha por etapa e repetição
  resumo.csv          mediana, mínimo e máximo por etapa
  hardware.json       CPU, memória, sistema, versões e threads
  rankings_sha256.json  hashes dos rankings de cada repetição (conferência de determinismo)
  protocolo.json      commits, argumentos e repetições

Só roda em Linux (os.wait4 e o pipeline exigem Linux; no Windows, use o WSL).

Uso:
    python scripts/evidencias/benchmark.py [--repeticoes 3] [--epocas-v3 60] [--epocas-v2 100]
"""
import argparse
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
COMMITS = {'v3': 'a67bcb5', 'v2': '5176508'}
EPOCAS_PADRAO = {'v3': 60, 'v2': 100}  # os padrões do Makefile de cada branch
METODOS = {('v3', 'relacional'): 'A', ('v3', 'floresta'): 'E_features', ('v3', 'gnn'): 'C',
           ('v3', 'mlp'): 'D', ('v2', 'gnn'): 'E_v2', ('v2', 'relacional'): 'A', ('v2', 'floresta'): 'E_features',
           ('v2', 'mlp'): 'MLP_v2'}


def nome_da_etapa(args):
    """Rótulo legível para os argumentos que o pipeline passa ao interpretador."""
    textos = [str(a) for a in args]
    primeiro = Path(textos[0]).name if textos else ''
    if primeiro == 'modelos_amostra.py':
        metodo = textos[1]
        seed = textos[textos.index('--seed') + 1] if '--seed' in textos else ''
        return metodo, (int(seed) if seed else None)
    rotulos = {'processar_graficos_arestas.py': 'filtrar_arestas', 'processar_graficos_nos.py': 'filtrar_nos',
               'filtrar_recorrentes.py': 'filtrar_recorrentes', 'processar_graficos_wallets.py': 'agregar_carteiras',
               'coorte.py': 'coorte_5000'}
    return rotulos.get(primeiro, primeiro or '?'), None


def carregar_pipeline(worktree):
    spec = importlib.util.spec_from_file_location(f'pipeline_{worktree.name}', worktree / 'scripts' / 'pipeline.py')
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def executar(branch, worktree, raw, destino, epocas, repeticao, registros):
    """Roda o pipeline da branch uma vez, medindo cada etapa."""
    pipeline = carregar_pipeline(worktree)
    original = pipeline.python

    def medido(repo, *args):
        etapa, seed = nome_da_etapa(args)
        inicio = time.perf_counter()
        processo = subprocess.Popen([sys.executable, '-B', *map(str, args)], cwd=repo, env=pipeline.ENV)
        _, status, uso = os.wait4(processo.pid, 0)
        processo.returncode = os.waitstatus_to_exitcode(status)
        segundos = time.perf_counter() - inicio
        if processo.returncode:
            raise subprocess.CalledProcessError(processo.returncode, args)
        linha = {'repeticao': repeticao, 'branch': branch, 'etapa': etapa, 'seed': seed,
                 'metodo': METODOS.get((branch, etapa)), 'segundos_processo': segundos,
                 'pico_rss_mib': uso.ru_maxrss / 1024, 'cpu_usuario_s': uso.ru_utime, 'cpu_sistema_s': uso.ru_stime}
        custos = Path(repo).parent / 'resultados' / f'{etapa}_{seed}' / 'custos.json'
        if seed is not None and custos.exists():
            interno = json.loads(custos.read_text())
            linha.update(segundos_metodo=interno.get('segundos'), epocas=interno.get('epocas'),
                         parametros=interno.get('parametros'))
        registros.append(linha)
        print(f'  [{branch} r{repeticao}] {etapa}{"" if seed is None else f" seed {seed}"}: '
              f'{segundos:.1f} s, {linha["pico_rss_mib"]:.0f} MiB', flush=True)

    pipeline.python = medido
    argv = ['pipeline.py', branch, '--raw', str(raw), '--destino', str(destino), '--modelos', 'todos',
            '--escopo', '5000', '--cenario', 'original', '--injecao', '40101', '--epocas', str(epocas),
            '--detector', 'principal']
    antigo, sys.argv = sys.argv, argv
    inicio = time.perf_counter()
    antes = set(destino.glob(f'{branch}-*'))
    try:
        pipeline.main()
    finally:
        sys.argv = antigo
        pipeline.python = original
    pasta = (set(destino.glob(f'{branch}-*')) - antes).pop()
    registros.append({'repeticao': repeticao, 'branch': branch, 'etapa': 'pipeline_total', 'seed': None,
                      'metodo': None, 'segundos_processo': time.perf_counter() - inicio})
    hashes = {str(p.relative_to(pasta / 'resultados')): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted((pasta / 'resultados').glob('*/ranking.csv'))}
    return pasta, hashes


def _saida(comando):
    try:
        return subprocess.check_output(comando, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def hardware(destino):
    info = {'data_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'sistema': platform.platform(), 'kernel': platform.release(), 'maquina': platform.machine(),
            'python': sys.version.split()[0], 'nucleos_logicos': os.cpu_count()}
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
    if info['wsl']:  # CPU e memória física do Windows hospedeiro
        host = _saida(['powershell.exe', '-NoProfile', '-Command',
                       '(Get-CimInstance Win32_Processor).Name; '
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


def resumir(etapas):
    chave = ['branch', 'etapa', 'seed', 'metodo']
    base = etapas.assign(seed=etapas.seed.fillna(-1), metodo=etapas.metodo.fillna(''))
    agregados = {'segundos_processo': ['median', 'min', 'max'], 'repeticao': 'nunique'}
    for coluna in ('pico_rss_mib', 'segundos_metodo'):
        if coluna in base:
            agregados[coluna] = ['median', 'min', 'max']
    resumo = base.groupby(chave, sort=False).agg(agregados)
    resumo.columns = ['_'.join(c) if c[1] != 'nunique' else 'repeticoes' for c in resumo.columns]
    resumo = resumo.reset_index()
    resumo['seed'] = resumo.seed.replace(-1, pd.NA)
    return resumo


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--repeticoes', type=int, default=3)
    parser.add_argument('--epocas-v3', type=int, default=EPOCAS_PADRAO['v3'])
    parser.add_argument('--epocas-v2', type=int, default=EPOCAS_PADRAO['v2'])
    parser.add_argument('--raw', type=Path, default=ROOT / 'data' / 'raw')
    parser.add_argument('--saida', type=Path, default=ROOT / 'benchmark')
    parser.add_argument('--trabalho', type=Path, default=ROOT / 'data' / 'benchmark',
                        help='worktrees e execuções; precisa estar no mesmo sistema de arquivos do repositório')
    parser.add_argument('--manter-execucoes', action='store_true')
    args = parser.parse_args(argv)
    if not hasattr(os, 'wait4'):
        parser.error('os.wait4 indisponível: rode no Linux ou no WSL')

    raw = args.raw.resolve()
    trabalho = args.trabalho.resolve()
    destino = trabalho / 'execucoes'
    destino.mkdir(parents=True, exist_ok=True)
    epocas = {'v3': args.epocas_v3, 'v2': args.epocas_v2}
    worktrees = {}
    registros, hashes = [], {}
    try:
        for branch, commit in COMMITS.items():
            caminho = trabalho / f'worktree_{branch}'
            if caminho.exists():
                subprocess.run(['git', 'worktree', 'remove', '--force', str(caminho)], cwd=ROOT, check=False)
            subprocess.run(['git', 'worktree', 'add', '--detach', '--quiet', str(caminho), commit], cwd=ROOT, check=True)
            worktrees[branch] = caminho
        for r in range(1, args.repeticoes + 1):
            ordem = list(COMMITS) if r % 2 else list(reversed(COMMITS))  # alterna para espalhar deriva térmica
            for branch in ordem:
                print(f'Repetição {r}/{args.repeticoes}: {branch} ({COMMITS[branch]}), {epocas[branch]} épocas', flush=True)
                pasta, h = executar(branch, worktrees[branch], raw, destino, epocas[branch], r, registros)
                hashes[f'{branch}_r{r}'] = h
                if not args.manter_execucoes:
                    shutil.rmtree(pasta)
    finally:
        for caminho in worktrees.values():
            subprocess.run(['git', 'worktree', 'remove', '--force', str(caminho)], cwd=ROOT, check=False)

    etapas = pd.DataFrame(registros)
    args.saida.mkdir(parents=True, exist_ok=True)
    etapas.to_csv(args.saida / 'etapas.csv', index=False, lineterminator='\n')
    resumir(etapas).to_csv(args.saida / 'resumo.csv', index=False, lineterminator='\n')
    deterministico = {b: len({json.dumps(v, sort_keys=True) for k, v in hashes.items() if k.startswith(b)}) == 1
                      for b in COMMITS}
    (args.saida / 'rankings_sha256.json').write_text(
        json.dumps({'identicos_entre_repeticoes': deterministico, 'por_repeticao': hashes}, indent=2) + '\n')
    (args.saida / 'hardware.json').write_text(json.dumps(hardware(destino), indent=2, ensure_ascii=False) + '\n',
                                              encoding='utf-8')
    protocolo = {'commits': COMMITS, 'checkout_main': _saida(['git', '-C', str(ROOT), 'rev-parse', 'HEAD']),
                 'repeticoes': args.repeticoes, 'epocas': epocas, 'escopo': '5000', 'cenario': 'original',
                 'seeds_redes': [42, 73, 2026], 'seed_floresta': 42, 'seed_divisao': 9103,
                 'threads': {k: v for k, v in carregar_pipeline(ROOT).ENV.items() if k.endswith('NUM_THREADS')},
                 'medicao': 'os.wait4 por processo: tempo de parede e ru_maxrss; segundos_metodo vem de custos.json '
                            '(tempo interno do método, sem a carga do interpretador)',
                 'sistema_de_arquivos': _saida(['df', '-T', str(trabalho)])}
    (args.saida / 'protocolo.json').write_text(json.dumps(protocolo, indent=2, ensure_ascii=False) + '\n',
                                               encoding='utf-8')
    print(f'Benchmark gravado em {args.saida}; rankings idênticos entre repetições: {deterministico}')


if __name__ == '__main__':
    main()
