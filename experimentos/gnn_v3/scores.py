"""Scores congelados, métricas separadas de construção/treinamento."""
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score,roc_auc_score
from experimentos.investigacao_anomalias.relacional import scores_relacionais
from .dados import FWD,context_rows,seconds

def ranking(ids,scores):
    r=pd.DataFrame({'wallet_id':np.asarray(ids),'score_anomalia':scores}).sort_values(['score_anomalia','wallet_id'],ascending=[False,True]).reset_index(drop=True)
    assert r.wallet_id.is_unique and np.isfinite(r.score_anomalia).all()
    r['posicao']=np.arange(1,len(r)+1);return r

def candidates(base):
    g=base['context'];p=g[FWD].edge_index.numpy();nu,ne=g['user'].num_nodes,g['event'].num_nodes
    mat=sparse.csr_matrix((np.ones(p.shape[1]),p),shape=(nu,ne));shared=(mat@mat.T).tocsr();shared.setdiag(0);shared.data[shared.data<2]=0;shared.eliminate_zeros()
    return p,shared,np.bincount(p[0],minlength=nu)

def top3(matrix):
    out=np.zeros(matrix.shape[0]);matrix=matrix.tocsr()
    for i in range(len(out)):
        vals=matrix.data[matrix.indptr[i]:matrix.indptr[i+1]]
        if len(vals):out[i]=np.partition(vals,max(0,len(vals)-3))[-3:].sum()/3
    return out

def manual(base,edges):
    g=base['context'];ids=g['user'].original_id.numpy();nu=len(ids);ne=g['event'].num_nodes;p,shared,degree=candidates(base)
    A=scores_relacionais(p,nu,ne)['surpresa_coparticipacao']
    rows=context_rows(base,edges).sort_values(['volume_usd','token_outcome_id'],ascending=[False,True]).drop_duplicates(['wallet_id','condition_node_id']).copy()
    rows['i']=rows.wallet_id.map(dict(zip(ids,range(nu))));rows['t']=seconds(rows.primeiro_trade);rows['v']=np.log1p(rows.volume_usd)
    rr=[];cc=[];oo=[];tt=[];vv=[]
    for _,a in rows.groupby('condition_node_id',sort=True):
        if len(a)<2:continue
        i,j=np.triu_indices(len(a),1);u=a.i.to_numpy()[i];v=a.i.to_numpy()[j];den=np.sqrt(degree[u]*degree[v])
        rr.append(u);cc.append(v);o=a.token_outcome_id.to_numpy();t=a.t.to_numpy();vol=a.v.to_numpy()
        oo.append((o[i]==o[j])/den);tt.append(np.exp(-np.abs(t[i]-t[j])/1800)/den);vv.append(np.exp(-np.abs(vol[i]-vol[j])/.25)/den)
    mask=shared.astype(bool);result={'A':A}
    for key,values in zip(('O','T','V'),(oo,tt,vv)):
        if rr:
            m=sparse.coo_matrix((np.concatenate(values),(np.concatenate(rr),np.concatenate(cc))),shape=(nu,nu)).tocsr();m=m+m.T;m=m.multiply(mask).tocsr();result[key]=top3(m)
        else:result[key]=np.zeros(nu)
    result['B']=np.mean([(rankdata(result[k],method='average')-1)/max(nu-1,1) for k in ('A','O','T','V')],axis=0)
    return {k:ranking(ids,v) for k,v in result.items()}

def pair_score(base,z):
    _,shared,_=candidates(base);coo=shared.tocoo();norm=z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12)
    values=np.clip((1+(norm[coo.row]*norm[coo.col]).sum(1))/2,0,1)
    mat=sparse.csr_matrix((values,(coo.row,coo.col)),shape=shared.shape)
    return ranking(base['context']['user'].original_id.numpy(),top3(mat))

def metrics(r,labels):
    assert set(r.wallet_id)==set(labels.wallet_id) and labels.wallet_id.is_unique
    merged=r.merge(labels,on='wallet_id',validate='one_to_one').sort_values(['score_anomalia','wallet_id'],ascending=[False,True])
    out=[]
    for universe,mask in [('completo',np.ones(len(merged),bool)),('elegivel',merged.elegivel.astype(bool)),('sem_doadoras',~merged.doadora.astype(bool))]:
        m=merged.loc[mask];y=m.alvo.to_numpy().astype(int);s=m.score_anomalia.to_numpy();n=int(y.sum());assert n==50
        out.append({'universo':universe,'n':len(y),'alvos':n,'top50':int(y[:50].sum()),'top100':int(y[:100].sum()),'recall100':float(y[:100].sum()/n),'precision50':float(y[:50].sum()/50),'AP':float(average_precision_score(y,s)),'AUC':float(roc_auc_score(y,s))})
    return out
