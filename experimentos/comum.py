"""Proveniencia, validacao e protecao de arquivos, sem depender de ML."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


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


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
