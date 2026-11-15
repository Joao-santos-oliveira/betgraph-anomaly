"""Regenera as cópias do teste PAREADO (sorteio 51011) e exporta só o grupo 0 para a Fig. 4.

Roda no WSL, no ambiente com torch/PyG, a partir da raiz do repositório:
    python -m visualizacao_gephi.intervencao_wsl RODADA

RODADA é a pasta de `make artigo` (contém dia/ e 51011/rotulos.csv). Usa exatamente
`experimentos.artigo.pareado.preparar_doses` — mesma seed, mesma divisão e mesmos rótulos —,
interceptando apenas a gravação de cada cópia para ler as arestas de contexto. Nenhum treino é feito.
"""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import torch

from experimentos.artigo import pareado
from experimentos.artigo.preparar import FORWARD, protocol, separar

SORTEIO = 51011
GRUPO = 0
ALVOS = [7896, 36905, 10780, 7219, 9621]
GRAUS = {7896: 34, 36905: 27, 10780: 24, 7219: 31, 9621: 42}


def arestas_ids(graph, edge_index):
    ids = graph['user'].original_id.numpy()
    mk = graph['event'].original_id.numpy()
    ei = edge_index.numpy()
    return pd.DataFrame({'wallet_id': ids[ei[0]], 'condition_node_id': mk[ei[1]]})


def main(run: Path, destino: Path):
    source = run / 'dia'
    split = separar(torch.load(source / 'hetero.pt', weights_only=False), seed=protocol['divisao'])
    frames = [pd.read_csv(source / f'{k}.csv') for k in ['user', 'event', 'edge']]
    graph = split['context']
    capturas = {0: arestas_ids(graph, graph[FORWARD].edge_index)}
    graus_orig = torch.bincount(graph[FORWARD].edge_index[0])

    def capturar(folder, changed, users, events, rows):
        ei = changed['context'][FORWARD].edge_index
        assert torch.equal(torch.bincount(ei[0]), graus_orig)  # grau de todas as carteiras preservado
        capturas[int(Path(folder).name)] = arestas_ids(changed['context'], ei)

    pareado.save_case = capturar
    with TemporaryDirectory() as tmp:
        out = Path(tmp) / str(SORTEIO)
        pareado.preparar_doses(run, out, SORTEIO, split, *frames)
        trocas = pd.read_csv(out / 'trocas.csv')
        rotulos = pd.read_csv(out / 'rotulos.csv')

    g = rotulos[(rotulos.grupo == GRUPO) & (rotulos.papel == 'alvo')].wallet_id.tolist()
    assert sorted(g) == sorted(ALVOS), g
    destino.mkdir(parents=True, exist_ok=True)
    linhas = []
    graus_cond = {}
    for dose, df in sorted(capturas.items()):
        sub = df[df.wallet_id.isin(ALVOS)].assign(dose=dose)
        graus_cond[dose] = sub.groupby('wallet_id').size().to_dict()
        assert graus_cond[dose] == GRAUS, (dose, graus_cond[dose])
        linhas.append(sub)
    arestas = pd.concat(linhas, ignore_index=True)
    arestas.to_csv(destino / 'arestas_grupo0_por_dose.csv', index=False)
    tg = trocas[(trocas.grupo == GRUPO)]
    tg.to_csv(destino / 'trocas_grupo0.csv', index=False)
    novos = tg.groupby('ordem').mercado_novo.first()
    # grau global (contexto) dos mercados envolvidos em cada dose
    mercados_vis = sorted(arestas.condition_node_id.unique())
    gm = {d: df[df.condition_node_id.isin(mercados_vis)].groupby('condition_node_id').size() for d, df in capturas.items()}
    pd.DataFrame(gm).fillna(0).astype(int).rename_axis('condition_node_id').to_csv(destino / 'grau_mercados_por_dose.csv')
    controles = rotulos[(rotulos.grupo == GRUPO) & (rotulos.papel == 'controle')]
    manifesto = {
        'sorteio': SORTEIO, 'grupo': GRUPO, 'teste': 'PAREADO (preparar_doses, mercados aninhados, trocas cumulativas)',
        'divisao_seed': protocol['divisao'], 'rodada': str(run), 'alvos': ALVOS, 'graus_contexto': GRAUS,
        'graus_por_dose_verificados': {str(k): v for k, v in graus_cond.items()},
        'mercados_novos_ordem': [int(x) for x in novos.tolist()],
        'doadoras_grupo0': sorted(int(x) for x in tg.doadora.unique()),
        'controles_grupo0': controles[['wallet_id', 'grau']].to_dict('records'),
        'arestas_contexto_por_dose': {str(k): int(len(v)) for k, v in capturas.items()},
        'verificacoes': ['graus de todas as carteiras idênticos ao contexto original em cada dose',
                         'graus dos cinco alvos = 34, 27, 24, 31, 42 em todas as doses',
                         'asserções internas de preparar_doses (graus de mercados, pares reservados, aninhamento)'],
    }
    (destino / 'manifesto_intervencao.json').write_text(json.dumps(manifesto, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: manifesto[k] for k in ('mercados_novos_ordem', 'arestas_contexto_por_dose')}))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('rodada', type=Path)
    ap.add_argument('--destino', type=Path, default=Path('figuras/recortes/gephi_intervencao/fonte'))
    a = ap.parse_args()
    main(a.rodada, a.destino)
