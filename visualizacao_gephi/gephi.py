"""Compila e executa RenderGephi.java contra os jars do Gephi 0.10.1 instalado (sem abrir a interface)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

GEPHI = Path(r'C:\Program Files\Gephi-0.10.1')
JDK = Path(r'C:\Program Files\Java\openJdk-24\bin')
AQUI = Path(__file__).resolve().parent / 'java'
BUILD = AQUI / 'build'


def classpath() -> str:
    """Todos os jars do Gephi instalado (módulos, bibliotecas externas e plataforma NetBeans)."""
    def ok(p: Path) -> bool:
        n = p.name
        if 'jre-x64' in p.parts or 'extra' in p.parts:
            return False
        # módulos de interface gráfica e o núcleo NetBeans quebram o modo headless (HeadlessException ao salvar)
        if n.startswith('org-gephi-desktop-') or n.endswith('-ui.jar') or n in ('core.jar', 'core-base.jar', 'boot.jar'):
            return False
        return True
    jars = sorted(str(p).replace('\\', '/') for p in GEPHI.rglob('*.jar') if ok(p))
    return ';'.join(jars)


def _argfile(nome: str, linhas: list[str]) -> Path:
    BUILD.mkdir(exist_ok=True)
    f = BUILD / nome
    f.write_text('\n'.join(f'"{l}"' if (' ' in l or ';' in l) else l for l in linhas), encoding='utf-8')
    return f


def compilar() -> None:
    fonte = AQUI / 'RenderGephi.java'
    alvo = BUILD / 'RenderGephi.class'
    if alvo.exists() and alvo.stat().st_mtime > fonte.stat().st_mtime:
        return
    a = _argfile('compilar.args', ['-encoding', 'UTF-8', '--release', '11', '-nowarn', '-cp', classpath(),
                                   '-d', str(BUILD).replace('\\', '/'), str(fonte).replace('\\', '/')])
    subprocess.run([str(JDK / 'javac.exe'), f'@{a}'], check=True)


def renderizar(gexf: Path, prefixo: Path, memoria: str = '6g', **opcoes) -> None:
    compilar()
    prefs = BUILD / 'prefs'
    prefs.mkdir(exist_ok=True)
    cp = '.;' + classpath()  # cwd=BUILD: evita o acento do caminho do projeto dentro do argfile
    a = _argfile('executar.args', [
        f'-Xmx{memoria}', '-Djava.awt.headless=true', '-Dnetbeans.user=.',
        '-Djava.util.prefs.userRoot=./prefs',
        '-Djava.util.prefs.PreferencesFactory=RenderGephi$EphemeralPreferences',
        '--add-opens=java.base/java.net=ALL-UNNAMED', '--add-opens=java.base/java.nio=ALL-UNNAMED',
        '--add-opens=java.base/java.lang=ALL-UNNAMED', '--add-opens=java.desktop/java.awt=ALL-UNNAMED',
        '-cp', cp, 'RenderGephi'])
    prefixo.parent.mkdir(parents=True, exist_ok=True)
    args = [str(JDK / 'java.exe'), '@executar.args', str(Path(gexf).resolve()), str(Path(prefixo).resolve())] + [f'{k}={v}' for k, v in opcoes.items()]
    r = subprocess.run(args, cwd=BUILD, text=True, capture_output=True, encoding='utf-8', errors='replace')
    log = (r.stdout or '') + (r.stderr or '')
    (prefixo.parent / f'{prefixo.name}_gephi.log').write_text(log, encoding='utf-8')
    # a saída do Java vem na codificação do console (acentos do caminho variam): confere o projeto gravado
    projeto = Path(f'{Path(prefixo).resolve()}.gephi')
    if r.returncode != 0 or '\nok ' not in '\n' + r.stdout or not projeto.exists() or projeto.stat().st_size == 0:
        sys.stderr.write(log[-4000:])
        raise RuntimeError(f'Gephi falhou para {gexf}')
    print([l for l in r.stdout.splitlines() if l.startswith(('importado', 'ok'))])
