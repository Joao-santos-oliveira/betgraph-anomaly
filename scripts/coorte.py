"""Regenera a segunda coorte disjunta usando o gerador publicado."""
import sys
from pathlib import Path
repo = Path(sys.argv[1])
sys.path.insert(0,str(repo))
import pandas as pd
import torch
from experimentos.validacao_ampliada.executar import new_cohort

users=pd.read_csv(repo/'data/raw/betgraph_wallets_16set_nos.csv').sort_values('wallet_id')
previous=users.sample(n=5000,random_state=42).wallet_id
split,users,events,edges=new_cohort(previous)
out=repo/'data/processed/validacao_ampliada_v1/novo'
out.mkdir(parents=True,exist_ok=False)
for key,frame in [('user',users),('event',events),('edge',edges)]:
    frame.to_csv(out/f'{key}.csv',index=False)
torch.save(split,out/'divisao.pt')
