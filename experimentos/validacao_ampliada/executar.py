from pathlib import Path
import copy,json,hashlib,time
import numpy as np
import pandas as pd
import torch,joblib
from torch_geometric.data import HeteroData
from experimentos.piloto_sintetico.comum import legacy_imports,read_json,write_json,sha
from experimentos.piloto_sintetico.gerar import inject,Inviavel
from experimentos.piloto_sintetico.avaliar import score_ranking
from experimentos.investigacao_anomalias.relacional import scores_relacionais
legacy_imports()
from preparar_gnn import separar,FORWARD,REVERSE
from features_contexto import features_contexto
from baseline import ajustar_ranking
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/processed/validacao_ampliada_v1'
OLD=ROOT/'data/processed/sintetico/piloto_v2_8101_8102/original'


def rank_graph(split):
    g=split['context'];stats=scores_relacionais(g[FORWARD].edge_index.numpy(),g['user'].num_nodes,g['event'].num_nodes)
    frame=pd.DataFrame({'wallet_id':g['user'].original_id.numpy(),**stats})
    frame['score_anomalia']=frame.surpresa_coparticipacao
    return ordenar(frame)


def ordenar(frame):
    frame=frame.sort_values(['score_anomalia','wallet_id'],ascending=[False,True]).reset_index(drop=True)
    frame['posicao']=np.arange(1,len(frame)+1)
    return frame


def volume_injection(split,edges,seed,factor):
    g=split['context'];uids=g['user'].original_id.numpy();eids=g['event'].original_id.numpy();pairs=g[FORWARD].edge_index.numpy()
    allowed=pd.MultiIndex.from_arrays([uids[pairs[0]],eids[pairs[1]]])
    mask=pd.MultiIndex.from_frame(edges[['wallet_id','condition_node_id']]).isin(allowed)
    deg=np.bincount(pairs[0],minlength=len(uids))
    vols=edges.loc[mask].groupby('wallet_id').volume_usd.sum().reindex(uids,fill_value=0).to_numpy()
    eligible=(deg>=5)&(vols>0)
    targets=np.random.default_rng(seed).choice(np.flatnonzero(eligible),50,replace=False)
    labels=pd.DataFrame({'wallet_id':uids,'alvo':np.isin(np.arange(len(uids)),targets).astype(int),'doadora':False,'elegivel':eligible})
    labels['grupo']=np.where(labels.alvo,'alvo','intacta')
    changed=edges.copy();hit=mask & edges.wallet_id.isin(uids[targets]).to_numpy()
    changed.loc[hit,'volume_usd']*=factor
    pd.testing.assert_frame_equal(changed.loc[~hit],edges.loc[~hit])
    pd.testing.assert_frame_equal(changed.drop(columns='volume_usd'),edges.drop(columns='volume_usd'))
    return changed,labels,{'rows_changed':int(hit.sum()),'factor':factor,'topology_unchanged':True}


def randomizar(split,seed):
    g=split['context'];original=g[FORWARD].edge_index.numpy().T.copy();pairs=original.copy()
    current=set(map(tuple,pairs));reserved=set(map(tuple,split['all_positive'].numpy().T))-current
    rng=np.random.default_rng(seed);accepted=0
    for _ in range(5*len(pairs)):
        i,j=rng.integers(len(pairs),size=2);u,a=pairs[i];v,b=pairs[j]
        if u==v or a==b or (u,b) in current or (v,a) in current or (u,b) in reserved or (v,a) in reserved:continue
        current.remove((u,a));current.remove((v,b));current.update(((u,b),(v,a)))
        pairs[i]=[u,b];pairs[j]=[v,a];accepted+=1
    for axis in (0,1):assert np.array_equal(np.bincount(original[:,axis]),np.bincount(pairs[:,axis]))
    assert len(current)==len(pairs) and not current&reserved
    result=copy.deepcopy(split);t=torch.tensor(pairs.T.copy());result['context'][FORWARD].edge_index=t;result['context'][REVERSE].edge_index=t.flip(0)
    # Score estrutural usa somente contexto; manter all_positive coerente mesmo sem treinar.
    result['all_positive']=torch.unique(torch.cat([t]+[result[k] for k in ('train_positive','val_positive','test_positive')],dim=1),dim=1)
    return result,accepted


def new_cohort(oldids):
    raw=ROOT/'data/raw';u=pd.read_csv(raw/'betgraph_wallets_16set_nos.csv').sort_values('wallet_id')
    u=u.loc[~u.wallet_id.isin(oldids)].sample(n=5000,random_state=9100).sort_values('wallet_id').reset_index(drop=True)
    a=pd.read_csv(raw/'betgraph_arestas_16set_final.csv');a=a.loc[a.wallet_id.isin(u.wallet_id)].copy()
    e=pd.read_csv(raw/'betgraph_nos_evento_16set_final.csv');e=e.loc[e.condition_node_id.isin(a.condition_node_id)].sort_values('condition_node_id').reset_index(drop=True)
    g=HeteroData()
    for kind,frame,col in [('user',u,'wallet_id'),('event',e,'condition_node_id')]:
        g[kind].original_id=torch.tensor(frame[col].to_numpy());g[kind].x=torch.zeros((len(frame),1))
    ui=dict(zip(u.wallet_id,range(len(u))));ei=dict(zip(e.condition_node_id,range(len(e))))
    g[FORWARD].edge_index=torch.tensor(np.array([a.wallet_id.map(ui),a.condition_node_id.map(ei)]));g[REVERSE].edge_index=g[FORWARD].edge_index.flip(0)
    return separar(g,seed=9103),u,e,a


def main():
    OUT.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    sources=list((ROOT/'experimentos/validacao_ampliada').glob('*.py'))+list((ROOT/'experimentos/investigacao_anomalias').glob('*.py'))+list((ROOT/'experimentos/piloto_sintetico').glob('*.py'))+[ROOT/'docs/validacao_ampliada/protocolo.md']+list((ROOT/'data/raw').glob('*final.csv'))+[ROOT/'data/raw/betgraph_wallets_16set_nos.csv',OLD/'entrada/divisao.pt']
    sources += list((ROOT/'src').glob('*.py')) + list((OLD/'entrada').glob('*.csv')) + [ROOT/'tests/test_validacao_ampliada.py', ROOT/'experimentos/piloto_sintetico/protocolo_v2.json']
    frozen={str(p):sha(p) for p in sources};write_json(OUT/'congelamento.json',{'sha256':frozen,'time_utc':pd.Timestamp.now(tz='UTC').isoformat()})
    split=torch.load(OLD/'entrada/divisao.pt',weights_only=False,map_location='cpu')
    u,e,a=[pd.read_csv(OLD/'entrada'/f'{k}.csv') for k in ('user','event','edge')]
    cohorts=[('anterior',split,u,e,a),('novo',*new_cohort(u.wallet_id))]
    records=[];statuses=[];nulls=[]
    for name,split,u,e,a in cohorts:
        folder=OUT/name;folder.mkdir()
        for k,f in [('user',u),('event',e),('edge',a)]:f.to_csv(folder/f'{k}.csv',index=False)
        torch.save(split,folder/'divisao.pt')
        base,raw,bundle,_=features_contexto(split,u,e,a)
        original_rel=rank_graph(base);_,original_if,_=ajustar_ranking(u,base['context']['user'].x.numpy())
        original_rel.to_csv(folder/'ranking_relacional_original.csv',index=False);raw.to_csv(folder/'features_original.csv',index=False)
        for seed in (9501,9502):
            randomized,accepted=randomizar(split,seed);r=rank_graph(randomized)
            r.to_csv(folder/f'nulo_{seed}.csv',index=False)
            nulls.append({'recorte':name,'seed':seed,'trocas_aceitas':accepted,'propostas':5*split['context'][FORWARD].num_edges,'top100_intersecao':len(set(r.head(100).wallet_id)&set(original_rel.head(100).wallet_id)), 'q99_original':float(original_rel.score_anomalia.quantile(.99)),'q99_nulo':float(r.score_anomalia.quantile(.99))})
        for family,levels in [('coparticipacao',[.3,.6,.9]),('volume',[10,50,100])]:
            for level in levels:
                for seed in (9201,9301):
                    dest=folder/f'{family}_{level}_{seed}';dest.mkdir();started=time.monotonic()
                    try:
                        if family=='coparticipacao':
                            p=read_json(ROOT/'experimentos/piloto_sintetico/protocolo_v2.json');p.update(concentration=level,selection_seed=seed,swap_seed=seed+1)
                            altered,changed,swaps,labels,diagnostic=inject(split,u,a,p);swaps.to_csv(dest/'trocas.csv',index=False)
                        else:
                            changed,labels,diagnostic=volume_injection(split,a,seed,level);altered=copy.deepcopy(split)
                        transformed,raw,bundle,report=features_contexto(altered,u,e,changed)
                        model,ir,_=ajustar_ranking(u,transformed['context']['user'].x.numpy());rr=rank_graph(transformed)
                        labels.to_csv(dest/'rotulos.csv',index=False);raw.to_csv(dest/'features.csv',index=False);joblib.dump(bundle,dest/'preprocessador.joblib');joblib.dump(model,dest/'if.joblib')
                        torch.save(altered,dest/'divisao.pt');changed.to_csv(dest/'edge.csv',index=False)
                        for scenario,ranks in [('original',{'relacional':original_rel,'features_if':original_if}),('injetado',{'relacional':rr,'features_if':ir})]:
                            for method,r in ranks.items():
                                r.to_csv(dest/f'ranking_{scenario}_{method}.csv',index=False)
                                rows,_=score_ranking(r,labels)
                                records.extend({'recorte':name,'familia':family,'intensidade':level,'seed':seed,'scenario':scenario,'model':method,**row} for row in rows)
                        write_json(dest/'manifest.json',{'diagnostic':diagnostic,'features':report,'teste_avaliado':False})
                        statuses.append({'recorte':name,'familia':family,'nivel':level,'seed':seed,'status':'concluido','seconds':time.monotonic()-started})
                    except Inviavel as exc:
                        write_json(dest/'inviavel.json',{'reason':str(exc),'diagnostic':exc.diagnostic});statuses.append({'recorte':name,'familia':family,'nivel':level,'seed':seed,'status':'inviavel'})
                    pd.DataFrame(records).to_csv(OUT/'metricas.csv',index=False);write_json(OUT/'status.json',statuses)
                    print(statuses[-1],flush=True)
    pd.DataFrame(nulls).to_csv(OUT/'nulos.csv',index=False)
    for path,h in frozen.items():assert sha(path)==h,path
    write_json(OUT/'integridade.json',{'fontes_intactas':True,'sem_sobreposicao_carteiras':not set(cohorts[0][2].wallet_id)&set(cohorts[1][2].wallet_id),'teste_avaliado':False,'n_cenarios':len(statuses)})

if __name__=='__main__':main()
