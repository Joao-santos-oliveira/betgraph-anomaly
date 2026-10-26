"""Detector individual e ranking hibrido de pares; nao usam rotulos."""
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import poisson


def ranking(ids,scores):
    if len(ids)!=len(scores) or len(set(ids))!=len(ids) or not np.isfinite(scores).all():
        raise ValueError('IDs ou scores invalidos')
    r=pd.DataFrame({'wallet_id':ids,'score_anomalia':scores})
    r=r.sort_values(['score_anomalia','wallet_id'],ascending=[False,True]).reset_index(drop=True)
    r.insert(0,'posicao',np.arange(1,len(r)+1))
    return r


def hibrido(pairs,n_users,n_events,z,block_size=256):
    if z.ndim!=2 or len(z)!=n_users or not np.isfinite(z).all():raise ValueError('Embeddings invalidos')
    matrix=sparse.csr_matrix((np.ones(pairs.shape[1],dtype=np.int64),pairs),shape=(n_users,n_events))
    if matrix.nnz!=pairs.shape[1]:raise ValueError('Pares duplicados')
    du=np.asarray(matrix.sum(axis=1)).ravel();de=np.asarray(matrix.sum(axis=0)).ravel();E=pairs.shape[1]
    factor=np.sum(de*(de-1))/(E*(E-1)) if E>1 else 0.
    z=z.astype(np.float64);z=z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12)
    scores=np.zeros(n_users)
    for start in range(0,n_users,block_size):
        overlap=(matrix[start:start+block_size]@matrix.T).tocsr()
        for local in range(overlap.shape[0]):
            u=start+local;lo,hi=overlap.indptr[local:local+2]
            vs,cs=overlap.indices[lo:hi],overlap.data[lo:hi];keep=(vs!=u)&(cs>=2);vs,cs=vs[keep],cs[keep]
            if not len(vs):continue
            surprise=np.minimum(-poisson.logsf(cs-1,du[u]*du[vs]*factor)/np.log(10),300.)
            cosine=np.clip(z[vs]@z[u],-1,1)
            contributions=surprise*(1+cosine)/2
            scores[u]=np.sort(contributions)[-3:].sum()/3
    if not np.isfinite(scores).all():raise ValueError('Score nao finito')
    return scores
