"""Features estruturais calculadas exclusivamente no grafo de contexto."""
import copy
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from experimentos.investigacao_anomalias.relacional import scores_relacionais

NAMES=['grau_contexto','contagem_coparticipacao','surpresa_coparticipacao',
       'mercados_compartilhados','sobreposicao_esperada_aprox']
FORWARD=('user','bets_on','event')


def enriquecer(split):
    graph=split['context']
    stats=scores_relacionais(graph[FORWARD].edge_index.numpy(),graph['user'].num_nodes,graph['event'].num_nodes)
    raw=np.column_stack([stats[k] for k in NAMES]);active=stats['grau_contexto']>0
    if not active.any():raise ValueError('Sem nos ativos para ajustar scaler')
    scaler=StandardScaler().fit(np.log1p(raw[active]))
    x=scaler.transform(np.log1p(raw)).astype(np.float32)
    result=copy.deepcopy(split)
    result['context']['user'].x=torch.cat([graph['user'].x,torch.from_numpy(x)],dim=1)
    return result,{'scaler':scaler,'names':['log1p_'+k for k in NAMES],'raw':raw},stats
