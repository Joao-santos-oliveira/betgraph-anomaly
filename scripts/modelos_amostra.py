"""Etapas da comparação na amostra ou no dia completo."""
import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import pandas as pd
import torch

# Executado pelo pipeline na cópia isolada da implementação de referência.
REPO = Path.cwd()
sys.path[:0] = [str(REPO), str(REPO / 'src')]
from baseline import ajustar_ranking
from experimentos.validacao_ampliada.executar import rank_graph, volume_injection
from experimentos.piloto_sintetico.gerar import inject, Inviavel
from experimentos.piloto_sintetico.avaliar import score_ranking
from features_contexto import features_contexto
from preparar_gnn import FORWARD
from ranking_gnn import exportar
from treinar_gnn import treinar

ENTRADA = REPO / 'data/processed/validacao_ampliada_v1/novo'
OUT = REPO.parent / 'resultados'


def intervir(split, frames, cenario, seed):
    users, _, edges = frames
    if cenario == 'volume':
        alteradas, rotulos, diagnostico = volume_injection(split, edges, seed, 10)
        mask = alteradas.volume_usd.ne(edges.volume_usd)
        alteradas.loc[mask, 'ticket_medio'] = alteradas.loc[mask, 'volume_usd'] / alteradas.loc[mask, 'n_trades']
    else:
        protocolo = json.loads((REPO/'experimentos/piloto_sintetico/protocolo_v2.json').read_text())
        protocolo.update(selection_seed=seed, swap_seed=seed+1, concentration=0.6)
        try:
            split, alteradas, trocas, rotulos, diagnostico = inject(split, users, edges, protocolo)
        except Inviavel as exc:
            (OUT/'inviavel.json').write_text(json.dumps({'motivo':str(exc),'diagnostico':exc.diagnostic},indent=2))
            raise
        trocas.to_csv(OUT/'trocas.csv', index=False)
    if rotulos.alvo.sum() != 50:
        raise ValueError('Esperados 50 alvos artificiais.')
    rotulos.to_csv(OUT/'rotulos.csv', index=False)
    (OUT/'injecao.json').write_text(json.dumps(diagnostico,indent=2))
    frames[2] = alteradas
    return split


def preparar(escopo, cenario, seed):
    OUT.mkdir()
    split = torch.load(ENTRADA / 'divisao.pt', weights_only=False)
    frames = [pd.read_csv(ENTRADA / f'{kind}.csv') for kind in ('user', 'event', 'edge')]
    if cenario != 'original':
        split = intervir(split, frames, cenario, seed)
    base, _, _, info = features_contexto(split, *frames)
    graph = base['context']
    expected = 5000 if escopo == '5000' else len(frames[0])
    if graph['user'].num_nodes != expected:
        raise ValueError('Universo de carteiras divergente.')
    torch.save(base, OUT / 'contexto.pt')
    pares = graph[FORWARD].edge_index.contiguous().numpy()
    info.update(carteiras=expected, mercados=graph['event'].num_nodes, seed_divisao=9103 if escopo=='5000' else 42,
                escopo=escopo, cenario=cenario, seed_injecao=seed if cenario!='original' else None,
                pares_contexto_sha256=hashlib.sha256(pares.tobytes()).hexdigest())
    (OUT / 'coorte.json').write_text(json.dumps(info, indent=2) + '\n')


def executar(modelo, seed, epocas):
    start = time.perf_counter()
    source = OUT / 'contexto.pt'
    base = torch.load(source, weights_only=False)
    users = pd.DataFrame({'wallet_id': base['context']['user'].original_id.numpy()})
    folder = OUT / f'{modelo}_{seed}'
    custos = {'metodo': modelo, 'seed': seed}
    if modelo in ('gnn', 'mlp'):
        report = treinar(source, folder, seed=seed, epochs=epocas, model_kind=modelo, feature_condition='context_only')
        ranking = exportar(source, folder/'melhor_modelo.pt', ENTRADA/'user.csv', folder/'ranking')
        custos.update(epocas=report['epochs_executadas'], parametros=report['n_parametros'],
                      auc_ligacoes_validacao=report['melhor_validacao']['auc'])
    else:
        folder.mkdir()
        if modelo == 'relacional':
            ranking = rank_graph(base)
        else:
            _, ranking, _ = ajustar_ranking(users, base['context']['user'].x.numpy(), seed=seed)
    if len(ranking) != len(users) or set(ranking.wallet_id) != set(users.wallet_id):
        raise ValueError('Cobertura incorreta no ranking.')
    ranking[['wallet_id','score_anomalia']].to_csv(folder/'ranking.csv', index=False)
    custos.update(segundos=time.perf_counter()-start,
                  pico_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    if (OUT/'rotulos.csv').exists():
        avaliacao = ranking[['wallet_id','score_anomalia']].copy()
        avaliacao['posicao'] = range(1,len(avaliacao)+1)
        medidas, _ = score_ranking(avaliacao,pd.read_csv(OUT/'rotulos.csv'))
        pd.DataFrame(medidas).assign(metodo=modelo,seed=seed).to_csv(folder/'metricas.csv',index=False)
    (folder/'custos.json').write_text(json.dumps(custos, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('metodo', choices=['preparar','relacional','floresta','gnn','mlp'])
    parser.add_argument('--escopo', choices=['5000','completo'], default='5000')
    parser.add_argument('--cenario', choices=['original','estrutura','volume'], default='original')
    parser.add_argument('--injecao', type=int, default=40101)
    parser.add_argument('--epocas', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.metodo == 'preparar':
        preparar(args.escopo, args.cenario, args.injecao)
    else:
        executar(args.metodo, args.seed, args.epocas)
