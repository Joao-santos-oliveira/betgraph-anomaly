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

METODOS = {
    'GNN híbrida': {'rotulo': 'GNN híbrida', 'cor': '#2a6fdb', 'marcador': 'circle'},
    'Relacional': {'rotulo': 'Relacional direto', 'cor': '#e07b00', 'marcador': 'square'},
    'Similaridade da GNN': {'rotulo': 'Similaridade isolada', 'cor': '#7a52c7', 'marcador': 'diamond'},
}
ORDEM_METODOS = list(METODOS)


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
