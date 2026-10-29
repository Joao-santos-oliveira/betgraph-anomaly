"""Intervenções e entradas sem acesso a rótulos pelo encoder."""
import copy
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import HeteroData
from experimentos.piloto_sintetico.comum import ROOT, read_json, legacy_imports
from .trocas import inject
legacy_imports()
from features_contexto import features_contexto
FWD=('user','bets_on','event')
REL=[('user',f'bets_{i}','event') for i in range(2)]+[('event',f'rev_{i}','user') for i in range(2)]

def context_rows(split,edges):
    g=split['context'];p=g[FWD].edge_index.numpy()
    allowed=pd.MultiIndex.from_arrays([g['user'].original_id.numpy()[p[0]],g['event'].original_id.numpy()[p[1]]])
    return edges.loc[pd.MultiIndex.from_frame(edges[['wallet_id','condition_node_id']]).isin(allowed)].copy()

def catalog(split,edges):
    a=context_rows(split,edges);out={}
    for mid,b in a.groupby('condition_node_id',sort=True):
        vals=b[['token_outcome_id','token_outcome']].drop_duplicates().sort_values('token_outcome_id')
        assert 1<=len(vals)<=2 and vals.token_outcome_id.is_unique
        out[int(mid)]=[(int(r.token_outcome_id),str(r.token_outcome)) for r in vals.itertuples()]
    return out

def seconds(values):
    t=pd.to_datetime(values,utc=True,format="mixed")
    return (t-t.dt.normalize()).dt.total_seconds().to_numpy()

def validate_change(original,altered,before,after,cat):
    g=original['context'];h=altered['context'];a=g[FWD].edge_index;b=h[FWD].edge_index
    for k in range(2):
        n=g['user' if k==0 else 'event'].num_nodes
        assert torch.equal(torch.bincount(a[k],minlength=n),torch.bincount(b[k],minlength=n))
    assert b.unique(dim=1).shape==b.shape
    for key in ('train_positive','val_positive','test_positive'):
        assert torch.equal(original[key],altered[key])
    reserved=set(map(tuple,torch.cat([original[k] for k in ('train_positive','val_positive','test_positive')],1).T.tolist()))
    assert not reserved.intersection(map(tuple,b.T.tolist()))
    keys=['wallet_id','condition_node_id','token_outcome_id']
    before_reserved=before.drop(context_rows(original,before).index).sort_values(keys).reset_index(drop=True)
    after_reserved=after.drop(context_rows(altered,after).index).sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(before_reserved,after_reserved,check_dtype=False)
    rows=context_rows(altered,after)
    assert not rows.duplicated(keys).any()
    for r in rows.itertuples():assert (int(r.token_outcome_id),str(r.token_outcome)) in cat[int(r.condition_node_id)]
    actual=set(map(tuple,rows[['wallet_id','condition_node_id']].drop_duplicates().to_numpy()))
    expected=set(zip(h['user'].original_id.numpy()[b[0]],h['event'].original_id.numpy()[b[1]]))
    assert actual==expected
    return {'graus_preservados':True,'reservados_intactos':True,'outcomes_coerentes':True,'linhas_antes':len(before),'linhas_depois':len(after)}

def intervene(split,users,edges,scenario,seed,cat):
    assert seed in (30101,30201)
    if scenario=='S3_volume':
        from experimentos.validacao_ampliada.executar import volume_injection
        changed,labels,diag=volume_injection(split,edges,seed,10)
        hit=context_rows(split,changed).index.intersection(changed.index[changed.volume_usd.ne(edges.volume_usd)])
        changed.loc[hit,'ticket_medio']=changed.loc[hit,'volume_usd']/changed.loc[hit,'n_trades']
        altered=copy.deepcopy(split);swaps=pd.DataFrame();ledger=pd.concat([edges.loc[hit].assign(estado='antes'),changed.loc[hit].assign(estado='depois')],ignore_index=True)
        diag['affected_markets']=sorted(int(v) for v in changed.loc[hit,'condition_node_id'].unique());diag['n_donors']=0
    else:
        p=read_json(ROOT/'experimentos/piloto_sintetico/protocolo_v2.json')
        p.update(selection_seed=seed,swap_seed=seed+1,concentration=.3 if scenario=='S1' else .6)
        binary={k for k,v in cat.items() if len(v)==2}
        altered,changed,swaps,labels,diag=inject(split,users,edges,p,allowed_markets=binary if scenario in ('S1','S2') else None)
        ledger=pd.DataFrame()
        if scenario in ('S1','S2'):
            original=context_rows(split,edges);rng=np.random.default_rng(seed+50)
            target_ids=set(labels.loc[labels.alvo.eq(1),'wallet_id'])
            selected=context_rows(altered,changed)
            selected=selected.loc[selected.wallet_id.isin(target_ids)&selected.condition_node_id.isin(diag['target_markets'])]
            replaced=[];removed=[];audit=[]
            for mid,rows in selected.groupby('condition_node_id',sort=True):
                ref=original.loc[original.condition_node_id.eq(mid)]
                median_volume=float(ref.groupby('wallet_id').volume_usd.sum().median())
                center=float(np.clip(np.median(seconds(ref.primeiro_trade)),1800,84600))
                groups=list(rows.groupby('wallet_id',sort=True));n=len(groups)
                sides=rng.permutation(np.arange(n)%2) if scenario=='S2' else np.zeros(n,dtype=int)
                times=rng.permutation(np.linspace(.02,.98,n))*86400 if scenario=='S2' else center+rng.uniform(-1800,1800,n)
                day=pd.to_datetime(ref.primeiro_trade,utc=True).min().normalize()
                for j,(uid,block) in enumerate(groups):
                    row=block.sort_values('token_outcome_id').iloc[0].copy();oid,name=cat[int(mid)][int(sides[j])]
                    row['token_outcome_id']=oid;row['token_outcome']=name
                    row['volume_usd']=median_volume*rng.uniform(.8,1.2) if scenario=='S1' else float(block.volume_usd.sum())
                    row['n_trades']=int(block.n_trades.sum());assert row['n_trades']>0
                    row['ticket_medio']=row['volume_usd']/row['n_trades']
                    row['preco_medio']=float(ref.loc[ref.token_outcome_id.eq(oid),'preco_medio'].median())
                    row['primeiro_trade']=(day+pd.Timedelta(seconds=float(times[j]))).isoformat();row['duracao_trade_segundos']=0.
                    replaced.append(row);removed.extend(block.index)
                    audit.extend([{'estado':'antes',**r} for r in block.to_dict('records')]);audit.append({'estado':'depois',**row.to_dict()})
            changed=pd.concat([changed.drop(index=removed),pd.DataFrame(replaced)],ignore_index=True)
            ledger=pd.DataFrame(audit)
    diag['integridade']=validate_change(split,altered,edges,changed,cat)
    return altered,changed,swaps,labels,diag,ledger

def edge_table(split,edges,cat):
    g=split['context'];a=context_rows(split,edges).sort_values(['wallet_id','condition_node_id','token_outcome_id']).reset_index(drop=True)
    ui={int(v):i for i,v in enumerate(g['user'].original_id)};ei={int(v):i for i,v in enumerate(g['event'].original_id)}
    pair=np.array([a.wallet_id.map(ui),a.condition_node_id.map(ei)],dtype=np.int64)
    slots=np.array([next(i for i,(oid,_) in enumerate(cat[int(m)]) if oid==o) for m,o in zip(a.condition_node_id,a.token_outcome_id)],dtype=np.int64)
    raw=np.column_stack([np.log1p(a.volume_usd.to_numpy()),a.preco_medio.to_numpy(),seconds(a.primeiro_trade)/86400]).astype(np.float32)
    assert np.isfinite(raw).all()
    binary=np.array([len(cat[int(m)])==2 for m in a.condition_node_id])
    return a,pair,slots,raw,binary

def typed_graph(x,pairs,slots,attrs):
    g=HeteroData()
    for kind,t in x.items():g[kind].x=t
    for slot in range(2):
        keep=slots==slot;p=pairs[:,keep];v=attrs[keep]
        g[REL[slot]].edge_index=p;g[REL[slot]].edge_attr=v
        g[REL[slot+2]].edge_index=p.flip(0);g[REL[slot+2]].edge_attr=v
    return g

def prepare(split,users,events,edges,cat):
    base,raw_users,base_scalers,report=features_contexto(split,users,events,edges)
    _,pairs,slots,raw,binary=edge_table(split,edges,cat)
    scaler=StandardScaler().fit(raw);attrs=torch.tensor(scaler.transform(raw),dtype=torch.float32)
    x={};node_scalers={}
    for axis,kind in enumerate(('user','event')):
        n=base['context'][kind].num_nodes;agg=np.zeros((n,14),dtype=np.float64)
        for slot in range(2):
            mask=slots==slot;idx=pairs[axis,mask];cnt=np.bincount(idx,minlength=n);den=np.maximum(cnt,1)
            avg=np.column_stack([np.bincount(idx,weights=raw[mask,j],minlength=n)/den for j in range(3)])
            sq=np.column_stack([np.bincount(idx,weights=raw[mask,j]**2,minlength=n)/den for j in range(3)])
            agg[:,slot*7:(slot+1)*7]=np.column_stack([np.log1p(cnt),avg,np.sqrt(np.maximum(sq-avg**2,0))])
        active=np.bincount(pairs[axis],minlength=n)>0
        ns=StandardScaler().fit(agg[active]);node_scalers[kind]=ns
        x[kind]=torch.cat([base['context'][kind].x,torch.tensor(ns.transform(agg),dtype=torch.float32)],1)
    table={'pairs':torch.tensor(pairs),'slots':torch.tensor(slots),'attrs':attrs,'binary':torch.tensor(binary)}
    return base,x,table,{'base':base_scalers,'edge':scaler,'node':node_scalers},report

def corrupt(table,seed):
    rng=np.random.default_rng(seed);out={k:v.clone() for k,v in table.items()}
    out['attrs']=table['attrs'][torch.tensor(rng.permutation(len(table['slots'])))]
    out['slots'][table['binary']]=1-out['slots'][table['binary']]
    return out
