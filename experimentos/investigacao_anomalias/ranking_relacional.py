"""Ranking estrutural interpretavel; sem acesso a rotulos ou mercados-alvo."""
import argparse
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import torch
from .relacional import scores_relacionais


def gerar(source, output):
    if output.exists():
        raise FileExistsError(output)
    split=torch.load(source,map_location='cpu',weights_only=False)
    graph=split['context']
    graph.validate(raise_on_error=True)
    pairs=graph['user','bets_on','event'].edge_index.numpy()
    ids=graph['user'].original_id.numpy()
    if len(np.unique(ids))!=len(ids):
        raise ValueError('IDs duplicados')
    stats=scores_relacionais(pairs,graph['user'].num_nodes,graph['event'].num_nodes)
    rank=pd.DataFrame({'wallet_id':ids,**stats})
    index=rank.pop('melhor_parceiro_indice').to_numpy()
    rank['parceiro_wallet_id']=pd.array([int(ids[i]) if i>=0 else None for i in index],dtype='Int64')
    rank['score_anomalia']=rank.surpresa_coparticipacao
    rank=rank.sort_values(['score_anomalia','wallet_id'],ascending=[False,True]).reset_index(drop=True)
    rank.insert(0,'posicao',np.arange(1,len(rank)+1))
    output.mkdir(parents=True)
    rank.to_csv(output/'ranking.csv',index=False)
    rank.head(100).to_csv(output/'top100.csv',index=False)
    (output/'manifest.json').write_text(json.dumps({'source':str(source.resolve()),
        'sha256_source':hashlib.sha256(source.read_bytes()).hexdigest(),
        'method':'surpresa_coparticipacao','top_partners':3,'support':2,
        'score':'media das 3 maiores surpresas; cauda Poisson aproximada; nao e p-valor calibrado',
        'labels_used':False,'test_evaluated':False,'n_users':len(ids)},indent=2))
    return rank


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();gerar(args.input,args.output)
