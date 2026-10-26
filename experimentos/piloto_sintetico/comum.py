"""Proveniencia, validacao e protecao de arquivos, sem depender de ML."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
PARTS = ('train_positive', 'val_positive', 'test_positive')


def legacy_imports():
    # Os scripts originais usam imports absolutos a partir de src/.
    source = str(ROOT / 'src')
    if source not in sys.path:
        sys.path.insert(0, source)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def validate_protocol(p):
    required = {'version', 'condition', 'n_targets', 'n_markets', 'min_user_degree',
                'min_market_degree', 'concentration', 'selection_seed', 'swap_seed',
                'max_attempts', 'model_seeds', 'if_seed', 'training', 'top_k',
                'reserved_selection_swap_seeds'}
    require(set(p) == required, 'Schema do protocolo invalido')
    require(p['version'] in ('v1', 'v2'), 'Versao de protocolo desconhecida')
    require(p['condition'] == 'context_only', 'Somente context_only e permitido')
    for key in ('n_targets', 'n_markets', 'min_user_degree', 'min_market_degree', 'max_attempts'):
        require(type(p[key]) is int and p[key] > 0, f'Parametro invalido: {key}')
    require(0 < p['concentration'] <= 1, 'Concentracao invalida')
    seeds = p['model_seeds']
    require(bool(seeds) and len(seeds) == len(set(seeds)), 'Seeds vazias ou duplicadas')
    all_seeds = seeds + [p[k] for k in ('selection_seed', 'swap_seed', 'if_seed')]
    require(all(type(s) is int and s >= 0 for s in all_seeds), 'Seed invalida')
    require(p['selection_seed'] != p['swap_seed'], 'Separar seeds de selecao e troca')
    t = p['training']
    require(set(t) == {'epochs', 'patience', 'lr', 'hidden', 'embedding', 'dropout',
                       'threads', 'min_delta', 'eval_seed', 'negative_seed'}, 'Schema de treino invalido')
    for k in ('epochs', 'patience', 'hidden', 'embedding', 'threads'):
        require(type(t[k]) is int and t[k] > 0, f'Treino invalido: {k}')
    require(t['lr'] > 0 and t['min_delta'] >= 0 and 0 <= t['dropout'] < 1, 'Parametros de treino invalidos')
    require(all(type(t[k]) is int and t[k] >= 0 for k in ('eval_seed', 'negative_seed')), 'Seed de treino invalida')
    require(not ({p['selection_seed'], p['swap_seed']} & set(seeds + [p['if_seed'], t['eval_seed'], t['negative_seed']])), 'Seeds da intervencao devem ser independentes')
    require(p['top_k'] == [50, 100], 'Cortes devem ser top-50/top-100')
    return p


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True, encoding='utf-8').strip()


def inventory(inputs):
    tracked = git('ls-files', 'src', 'tests', 'requirements.txt', 'mapeamento_*.json').splitlines()
    protected = {str((ROOT / name).resolve()): sha(ROOT / name) for name in tracked}
    code = list((ROOT / 'experimentos/piloto_sintetico').glob('*.py'))
    code += list((ROOT / 'experimentos/piloto_sintetico').glob('protocolo_*.json'))
    code += list((ROOT / 'tests').glob('test_piloto_sintetico*.py'))
    return {'commit': git('rev-parse', 'HEAD'), 'git_status': git('status', '--short'),
            'protected': protected, 'inputs': {str(Path(p).resolve()): sha(p) for p in inputs},
            'new_code': {str(p.resolve()): sha(p) for p in code if p.exists()},
            'python': platform.python_version(), 'executable': sys.executable,
            'versions': {name: version(name) for name in ('numpy', 'pandas', 'torch', 'torch-geometric', 'scikit-learn', 'joblib')}}


def version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def verify_inventory(saved):
    for group in ('protected', 'inputs', 'new_code'):
        for path, expected in saved[group].items():
            require(Path(path).is_file() and sha(path) == expected, f'Arquivo alterado: {path}')


def seal_tree(folder):
    return {str(p.relative_to(folder)): sha(p) for p in sorted(Path(folder).rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts}


def verify_tree(folder, seal):
    for name, digest in seal.items():
        path = Path(folder) / name
        require(path.is_file() and sha(path) == digest, f'Artefato alterado: {path}')


def load_run(run):
    run = Path(run).resolve()
    manifest = read_json(run / 'manifesto_injecao.json')
    require(manifest['status'] == 'viavel', 'Intervencao inviavel: treino proibido')
    p = validate_protocol(read_json(run / 'protocolo.json'))
    require(sha(run / 'protocolo.json') == manifest['protocol_sha256'], 'Protocolo alterado')
    verify_inventory(read_json(run / 'inventario_hashes.json'))
    verify_tree(run, manifest['artifacts'])
    return p, manifest
