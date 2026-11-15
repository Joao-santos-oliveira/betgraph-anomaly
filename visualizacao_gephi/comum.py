"""Leitura dos dados do dia, regra de modalidade/evento, paleta e gravação de recortes."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import xml.sax.saxutils as sx
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
ARESTAS = RAW / 'betgraph_arestas_16set_final.csv'
MERCADOS = RAW / 'betgraph_nos_evento_16set_final.csv'
RESULTADOS = ROOT / 'resultados' / 'artigo'
FIGURAS = ROOT / 'figuras'
RECORTES = FIGURAS / 'recortes'

LIDERES = [17762, 18767, 20748, 18753, 17487]
GEPHI_VERSAO = '0.10.1'

# Okabe–Ito (seguro para daltonismo); a mesma codificação em todas as figuras.
COR = {
    'carteira': '#0072B2',
    'mercado': '#E69F00',
    'foco': '#B2182B',
    'parceiro': '#6A3D9A',
    'compartilhado': '#009E73',
    'exclusivo': '#E69F00',
    'intervencao': '#CC79A7',
    'neutro': '#9E9E9E',
    'aresta': '#5F6B7A',
    'texto': '#1A1A1A',
}

# Modalidades: ordem de exibição e cor de contorno/faixa.
# Tons pastel (ColorBrewer Set3) para não competir com as cores saturadas dos papéis dos nós.
MODALIDADES = [
    ('Futebol', '#8DD3C7'), ('CS2', '#FDB462'), ('Dota 2', '#FB8072'), ('LoL', '#BEBADA'),
    ('Valorant', '#FCCDE5'), ('Rocket League', '#B3DE69'), ('Outros esports', '#FFED6F'),
    ('Tênis', '#80B1D3'), ('Beisebol', '#BC80BD'), ('Futebol americano', '#CCEBC5'),
    ('Basquete', '#FFFFB3'), ('Pickleball', '#D9D9D9'), ('Luta (MMA)', '#E5C494'),
    ('Golfe', '#B3CDE3'), ('Outros esportes', '#DECBE4'), ('Política', '#C7E9C0'),
    ('Cripto/Finanças', '#FDDAEC'), ('Outros temas', '#E0E0E0'), ('indeterminado', '#F2F2F2'),
]
COR_MODALIDADE = dict(MODALIDADES)
ORDEM_MODALIDADE = {m: i for i, (m, _) in enumerate(MODALIDADES)}


def modalidade(tags: str, categoria: str) -> str:
    """Regra explícita e auditável (tags da Polymarket → modalidade)."""
    t = f' {str(tags).lower()} '
    regras = [
        ('counter strike', 'CS2'), ('league of legends', 'LoL'), ('dota', 'Dota 2'),
        ('valorant', 'Valorant'), ('rocket league', 'Rocket League'), (' esports ', 'Outros esports'),
        (' soccer ', 'Futebol'), (' tennis ', 'Tênis'), (' baseball ', 'Beisebol'), (' mlb ', 'Beisebol'),
        (' nfl ', 'Futebol americano'), (' cfb ', 'Futebol americano'), (' ncaaf ', 'Futebol americano'),
        (' basketball ', 'Basquete'), (' nba ', 'Basquete'), (' wnba ', 'Basquete'),
        (' pickleball ', 'Pickleball'), (' ufc ', 'Luta (MMA)'), (' mma ', 'Luta (MMA)'),
        (' pga ', 'Golfe'), (' golf ', 'Golfe'),
    ]
    for chave, nome in regras:
        if chave in t:
            return nome
    cat = str(categoria)
    if cat == 'Sports':
        return 'Outros esportes'
    if cat in ('Politics', 'Iran', 'Geopolitics', 'Elections'):
        return 'Política'
    if cat in ('Crypto', 'Finance', 'Economy'):
        return 'Cripto/Finanças'
    if cat and cat != 'nan':
        return 'Outros temas'
    return 'indeterminado'


_VS = re.compile(r'^(?:[\w .\-]+?:\s*)?(?P<a>.+?)\s+vs\.?\s+(?P<b>.+?)(?:\s*[:(]|\s+[\-–]\s+|\s+end in a draw|\?|$)', re.I)


def _norm(s: str) -> str:
    s = unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode().lower()
    return re.sub(r'[^a-z0-9]+', ' ', s).strip()


def _limpar_equipe(s: str) -> str:
    s = re.sub(r'^(will|counter-strike|lol|dota 2|valorant|rocket league)\s*:?\s*', '', s.strip(), flags=re.I)
    return s.strip(' .?')


def eventos(df: pd.DataFrame) -> pd.Series:
    """Evento (partida) inferido do texto, com regra registrada.

    1) pergunta "A vs B" → evento "A × B";
    2) pergunta sem "vs" que cita o nome de uma equipe de um evento do item 1 com as mesmas tags
       → esse evento (apenas se o nome casar com um único evento);
    3) caso contrário → "indeterminado" (não significa ausência de evento).
    """
    ev = pd.Series('indeterminado', index=df.index, dtype=object)
    regra = pd.Series('indeterminado', index=df.index, dtype=object)
    equipes: dict[tuple[str, str], set[str]] = {}
    for i, q, tags in zip(df.index, df.question.astype(str), df.tags.astype(str)):
        m = _VS.search(q)
        if not m:
            continue
        a, b = _limpar_equipe(m['a']), _limpar_equipe(m['b'])
        if len(_norm(a)) < 2 or len(_norm(b)) < 2:
            continue
        nome = f'{a} × {b}'
        ev[i], regra[i] = nome, 'pergunta_vs'
        for eq in (a, b):
            equipes.setdefault((tags, _norm(eq)), set()).add(nome)
    for i, q, tags in zip(df.index, df.question.astype(str), df.tags.astype(str)):
        if regra[i] != 'indeterminado':
            continue
        nq = f' {_norm(q)} '
        achados = {n for (tg, eq), ns in equipes.items() if tg == tags and len(eq) >= 4 and f' {eq} ' in nq for n in ns}
        if len(achados) == 1:
            ev[i], regra[i] = achados.pop(), 'nome_equipe'
    return pd.DataFrame({'evento': ev, 'regra_evento': regra})


@lru_cache(maxsize=1)
def pares() -> pd.DataFrame:
    """Pares distintos carteira–mercado (uma aresta por par, sem direções do PyG)."""
    e = pd.read_csv(ARESTAS, usecols=['wallet_id', 'condition_node_id', 'n_trades', 'volume_usd'])
    p = (e.groupby(['wallet_id', 'condition_node_id'], as_index=False)
          .agg(n_trades=('n_trades', 'sum'), volume_usd=('volume_usd', 'sum'), linhas=('n_trades', 'size')))
    return p


@lru_cache(maxsize=1)
def mercados() -> pd.DataFrame:
    ev = pd.read_csv(MERCADOS)
    ev['modalidade'] = [modalidade(t, c) for t, c in zip(ev.tags, ev.categoria_principal)]
    ev = pd.concat([ev, eventos(ev)], axis=1)
    return ev.set_index('condition_node_id')


def graus() -> tuple[pd.Series, pd.Series]:
    p = pares()
    return p.groupby('wallet_id').size(), p.groupby('condition_node_id').size()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for bloco in iter(lambda: f.read(1 << 20), b''):
            h.update(bloco)
    return h.hexdigest()


def gravar_recorte(nome: str, nos: pd.DataFrame, arestas: pd.DataFrame, manifesto: dict) -> Path:
    """Grava nos.csv, arestas.csv, manifesto.json e um GEXF 1.3 com cor/tamanho/posição nativos."""
    pasta = RECORTES / nome
    pasta.mkdir(parents=True, exist_ok=True)
    nos.to_csv(pasta / 'nos.csv', index=False, encoding='utf-8')
    arestas.to_csv(pasta / 'arestas.csv', index=False, encoding='utf-8')
    manifesto = {'figura': nome, 'gephi': GEPHI_VERSAO, **manifesto,
                 'entradas_sha256': {p.name: sha256(p) for p in (ARESTAS, MERCADOS)}}
    (pasta / 'manifesto.json').write_text(json.dumps(manifesto, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    escrever_gexf(pasta / f'{nome}.gexf', nos, arestas)
    return pasta


def _rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip('#')
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def escrever_gexf(path: Path, nos: pd.DataFrame, arestas: pd.DataFrame) -> None:
    """nos: Id, Label, Tipo, x, y, size, color (+ atributos); arestas: Source, Target, Weight (+ atributos)."""
    fixos_n = {'Id', 'Label', 'x', 'y', 'size', 'color'}
    fixos_e = {'Source', 'Target', 'Weight', 'color', 'alpha'}
    atr_n = [c for c in nos.columns if c not in fixos_n]
    atr_e = [c for c in arestas.columns if c not in fixos_e]

    def tipo(s: pd.Series) -> str:
        if pd.api.types.is_bool_dtype(s):
            return 'boolean'
        if pd.api.types.is_integer_dtype(s):
            return 'long'
        if pd.api.types.is_float_dtype(s):
            return 'double'
        return 'string'

    q = lambda v: sx.quoteattr(str(v))
    with open(path, 'w', encoding='utf-8') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n'
                '<gexf xmlns="http://gexf.net/1.3" xmlns:viz="http://gexf.net/1.3/viz" version="1.3">\n'
                '<graph defaultedgetype="undirected" mode="static">\n<attributes class="node">\n')
        for i, c in enumerate(atr_n):
            f.write(f'<attribute id="n{i}" title={q(c)} type="{tipo(nos[c])}"/>\n')
        f.write('</attributes>\n<attributes class="edge">\n')
        for i, c in enumerate(atr_e):
            f.write(f'<attribute id="e{i}" title={q(c)} type="{tipo(arestas[c])}"/>\n')
        f.write('</attributes>\n<nodes>\n')
        cols = list(nos.columns)
        for row in nos.itertuples(index=False):
            r = dict(zip(cols, row))
            f.write(f'<node id={q(r["Id"])} label={q(r.get("Label", ""))}><attvalues>')
            for i, c in enumerate(atr_n):
                v = r[c]
                if isinstance(v, bool):
                    v = str(v).lower()
                f.write(f'<attvalue for="n{i}" value={q(v)}/>')
            cr, cg, cb = _rgb(r['color'])
            f.write(f'</attvalues><viz:color r="{cr}" g="{cg}" b="{cb}"/>'
                    f'<viz:position x="{r["x"]:.4f}" y="{r["y"]:.4f}" z="0.0"/><viz:size value="{r["size"]:.4f}"/></node>\n')
        f.write('</nodes>\n<edges>\n')
        cols = list(arestas.columns)
        for k, row in enumerate(arestas.itertuples(index=False)):
            r = dict(zip(cols, row))
            f.write(f'<edge id="{k}" source={q(r["Source"])} target={q(r["Target"])} weight="{float(r.get("Weight", 1.0)):.4f}">')
            if atr_e:
                f.write('<attvalues>')
                for i, c in enumerate(atr_e):
                    v = r[c]
                    if isinstance(v, bool):
                        v = str(v).lower()
                    f.write(f'<attvalue for="e{i}" value={q(v)}/>')
                f.write('</attvalues>')
            if 'color' in r:
                cr, cg, cb = _rgb(r['color'])
                f.write(f'<viz:color r="{cr}" g="{cg}" b="{cb}" a="{float(r.get("alpha", 1.0)):.3f}"/>')
            f.write('</edge>\n')
        f.write('</edges>\n</graph>\n</gexf>\n')
