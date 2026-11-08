"""Experimentos desta versão e dependências históricas locais."""
from pathlib import Path

# A referência histórica é instalada por scripts/preparar_referencia.py.
_referencia = Path(__file__).resolve().parents[1] / "data" / "referencias" / "c0b7390" / "experimentos"
if _referencia.is_dir():
    __path__.append(str(_referencia))
