"""Formatação numérica, LaTeX e estilo visual comuns às tabelas e aos gráficos."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTADOS = ROOT / 'resultados' / 'artigo'
EXTRA = RESULTADOS / 'extra'
ENTREGA = ROOT / 'entrega'

FONTE = 'Helvetica, Arial, "Liberation Sans", sans-serif'
TINTA = '#1f2328'
TINTA_SUAVE = '#59636e'
LINHA = '#d0d7de'
FUNDO_CABECALHO = '#24292f'
FUNDO_ALTERNADO = '#f6f8fa'

# Slots 1-3 da paleta categórica de referência (validada para daltonismo em todos os pares).
# O verde-água fica abaixo de 3:1 de contraste: cada método tem também marcador próprio e legenda.
METODOS = {
    'GNN híbrida': {'rotulo': 'GNN híbrida', 'cor': '#2a78d6', 'marcador': 'circle'},
    'Relacional': {'rotulo': 'Relacional direto', 'cor': '#eb6834', 'marcador': 'square'},
    'Similaridade da GNN': {'rotulo': 'Similaridade isolada', 'cor': '#1baf7a', 'marcador': 'diamond'},
}
NEUTRO = '#8a8986'  # séries de referência (controles, diagonal)
GRADE = '#e6e5e1'
SUPERFICIE = '#ffffff'
ORDEM_METODOS = list(METODOS)


def exportar_imagens(figuras, bases, escala_png=3, formatos=('.png', '.pdf')):
    """PNG e formatos vetoriais em lote, com o tamanho de cada figura (em lote o kaleido ignora o layout).

    No Windows, o kaleido às vezes falha ao fechar o navegador depois de gravar tudo; essa falha é
    ignorada quando todos os arquivos daquele formato existem e são mais novos que o início.
    """
    import time
    import plotly.io as pio
    tamanho = dict(width=[f.layout.width for f in figuras], height=[f.layout.height for f in figuras])
    for sufixo in formatos:
        extra = dict(scale=escala_png) if sufixo == '.png' else {}
        caminhos = [b.with_suffix(sufixo) for b in bases]
        inicio = time.time()
        try:
            pio.write_images(figuras, caminhos, **tamanho, **extra)
        except RuntimeError as erro:
            gravados = all(c.exists() and c.stat().st_mtime >= inicio - 1 for c in caminhos)
            if 'close' not in str(erro) or not gravados:
                raise
            print(f'Aviso: o navegador não fechou sozinho depois de gravar os {sufixo}; arquivos conferidos.', flush=True)


def numero(valor, casas=0):
    """Número no padrão brasileiro: ponto nos milhares e vírgula decimal."""
    texto = f'{valor:,.{casas}f}'
    return texto.replace(',', '\0').replace('.', ',').replace('\0', '.')


def contagem(valor):
    """Contagem média: inteira quando exata, com duas casas caso contrário."""
    return numero(valor) if float(valor).is_integer() else numero(valor, 2)


def percentual(valor, casas=1):
    return f'{numero(valor, casas)}%'


_ESCAPES = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_',
            '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}',
            # Símbolos que o pdfLaTeX não aceita direto em UTF-8.
            '→': r'$\rightarrow$', 'ρ': r'$\rho$', '±': r'$\pm$', '×': r'$\times$', 'Δ': r'$\Delta$'}


def tex(texto):
    return ''.join(_ESCAPES.get(c, c) for c in str(texto))
