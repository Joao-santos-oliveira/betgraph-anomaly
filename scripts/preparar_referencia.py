"""Disponibiliza o comparador publicado no commit v2 para uso pela v3."""
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "c0b7390d9992665d81b8429d731e3b4c9f106f05"
DEST = ROOT / "data" / "referencias" / "c0b7390" / "experimentos" / "gnn_v2"
FILES = ("__init__.py", "treino.py", "objetivos.py", "features.py", "detectores.py")


def main():
    DEST.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=DEST.parent) as directory:
        temp = Path(directory) / "gnn_v2"
        temp.mkdir()
        for name in FILES:
            original = subprocess.check_output(
                ["git", "show", f"{COMMIT}:experimentos/gnn_v2/{name}"], cwd=ROOT
            )
            if DEST.exists():
                if not (DEST/name).is_file() or (DEST/name).read_bytes() != original:
                    raise SystemExit(f"Referência local divergente: {DEST/name}")
            (temp/name).write_bytes(original)
        if not DEST.exists():
            temp.rename(DEST)
    print("Comparador histórico disponível em", DEST)


if __name__ == "__main__":
    main()
