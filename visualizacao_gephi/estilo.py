"""Estilo de impressão comum: 17,5 cm de largura, texto ≥ 8 pt, fontes editáveis no PDF/SVG."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from visualizacao_gephi.comum import COR, FIGURAS  # noqa: E402

CM = 1 / 2.54
LARGURA = 17.5 * CM
FONTE_MIN = 8.0

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
    'font.size': 8,
    'axes.titlesize': 8.5,
    'legend.fontsize': 8,
    'pdf.fonttype': 42,      # TrueType embutida: texto editável no PDF
    'svg.fonttype': 'none',  # texto como texto no SVG
    'savefig.facecolor': 'white',
    'figure.facecolor': 'white',
    'axes.unicode_minus': False,
    'path.simplify': False,
})


def num(x, casas=1):
    """Número no padrão brasileiro (vírgula decimal, ponto de milhar)."""
    s = f'{x:,.{casas}f}'
    return s.replace(',', '§').replace('.', ',').replace('§', '.')


def inteiro(x):
    return f'{int(x):,}'.replace(',', '.')


def letra_painel(ax, letra, titulo, x=0.0, y=1.0):
    ax.text(x, y, letra, transform=ax.transAxes, fontsize=10, fontweight='bold', va='top', ha='left', color=COR['texto'])
    ax.text(x + 0.045, y - 0.004, titulo, transform=ax.transAxes, fontsize=8.5, va='top', ha='left', color=COR['texto'])


def legenda_marcadores(ax, itens, **kw):
    """itens: [(rótulo, marcador, cor_face, cor_borda, tamanho)] ou [(rótulo, 'linha', cor, largura, alpha)]."""
    h = []
    for it in itens:
        if it[1] == 'linha':
            h.append(Line2D([0], [0], color=it[2], lw=it[3], alpha=it[4], label=it[0]))
        else:
            h.append(Line2D([0], [0], marker=it[1], ls='', markerfacecolor=it[2], markeredgecolor=it[3],
                            markeredgewidth=0.6, markersize=it[4], label=it[0]))
    kw.setdefault('frameon', False)
    kw.setdefault('handletextpad', 0.4)
    kw.setdefault('borderaxespad', 0.0)
    return ax.legend(handles=h, **kw)


def salvar(fig, nome: str, meta: dict, dpi_png=600, rasterizar_dpi=600):
    FIGURAS.mkdir(exist_ok=True)
    md = {'Title': meta.get('titulo', nome), 'Author': 'Betgraph', 'Subject': meta.get('assunto', ''),
          'Keywords': 'Gephi 0.10.1; matplotlib'}
    saidas = {}
    for ext in ('pdf', 'svg', 'png'):
        f = FIGURAS / f'{nome}.{ext}'
        kw = {'dpi': dpi_png if ext == 'png' else rasterizar_dpi}
        if ext == 'pdf':
            kw['metadata'] = md
        fig.savefig(f, **kw)
        saidas[ext] = f
    largura_cm, altura_cm = fig.get_size_inches() / CM
    info = {**meta, 'largura_cm': round(largura_cm, 2), 'altura_cm': round(altura_cm, 2),
            'arquivos': {k: v.name for k, v in saidas.items()},
            'bytes': {k: v.stat().st_size for k, v in saidas.items()}}
    (FIGURAS / 'recortes' / nome).mkdir(parents=True, exist_ok=True)
    (FIGURAS / 'recortes' / nome / 'figura.json').write_text(json.dumps(info, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    verificar_fontes(fig, nome)
    plt.close(fig)
    return saidas


def verificar_fontes(fig, nome):
    pequenos = []
    for t in fig.findobj(matplotlib.text.Text):
        if t.get_text().strip() and t.get_visible() and t.get_fontsize() < FONTE_MIN - 1e-6:
            pequenos.append((t.get_text()[:30], t.get_fontsize()))
    if pequenos:
        raise AssertionError(f'{nome}: textos abaixo de {FONTE_MIN} pt: {pequenos[:5]}')
