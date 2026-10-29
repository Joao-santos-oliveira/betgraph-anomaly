"""Uma configuração: atenção com arestas ou MLP com as mesmas entradas."""
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import HeteroConv,TransformerConv
from experimentos.piloto_sintetico.comum import legacy_imports,write_json,sha
from .dados import REL,typed_graph,corrupt
legacy_imports()
from treinar_gnn import validar_divisao,amostrar_negativos,codigos,exemplos

class Encoder(nn.Module):
    def __init__(self,dims,kind):
        super().__init__();self.kind=kind
        if kind=='C':
            self.layers=nn.ModuleList([HeteroConv({r:TransformerConv((dims[r[0]],dims[r[2]]),32,heads=1,concat=False,dropout=.1,edge_dim=3) for r in REL},aggr='sum'),HeteroConv({r:TransformerConv((32,32),16,heads=1,concat=False,dropout=.1,edge_dim=3) for r in REL},aggr='sum')])
        else:
            assert kind=='D'
            self.mlps=nn.ModuleDict({k:nn.Sequential(nn.Linear(d,32),nn.ReLU(),nn.Dropout(.1),nn.Linear(32,16)) for k,d in dims.items()})
        self.link=nn.Sequential(nn.Linear(32,32),nn.ReLU(),nn.Linear(32,1))
        self.aux=nn.Sequential(nn.Linear(37,32),nn.ReLU(),nn.Linear(32,1))
    def forward(self,g):
        if self.kind=='D':return {k:self.mlps[k](v) for k,v in g.x_dict.items()}
        x=self.layers[0](g.x_dict,g.edge_index_dict,edge_attr_dict=g.edge_attr_dict)
        x={k:F.dropout(F.relu(v),p=.1,training=self.training) for k,v in x.items()}
        return self.layers[1](x,g.edge_index_dict,edge_attr_dict=g.edge_attr_dict)
    def decode(self,z,pairs):return self.link(torch.cat([z['user'][pairs[0]],z['event'][pairs[1]]],1)).flatten()
    def edge_logits(self,z,t,idx):
        p=t['pairs'][:,idx]
        return self.aux(torch.cat([z['user'][p[0]],z['event'][p[1]],t['attrs'][idx],F.one_hot(t['slots'][idx],2).float()],1)).flatten()

def evaluation_examples(split):
    observed=validar_divisao(split);g=split['context'];nu,ne=g['user'].num_nodes,g['event'].num_nodes
    nv=amostrar_negativos(nu,ne,split['val_positive'].shape[1],observed,np.random.default_rng(43))
    nt=amostrar_negativos(nu,ne,split['test_positive'].shape[1],observed|codigos(nv,ne),np.random.default_rng(44))
    vp,vl=exemplos(split['val_positive'],nv)
    return vp,vl,observed|codigos(nv,ne)|codigos(nt,ne)

def graph(x,t):return typed_graph(x,t['pairs'],t['slots'],t['attrs'])

def auxiliary(model,z,zc,t,tc,idx):
    real=model.edge_logits(z,t,idx);fake=model.edge_logits(zc,tc,idx)
    return F.binary_cross_entropy_with_logits(torch.cat([real,fake]),torch.cat([torch.ones_like(real),torch.zeros_like(fake)]))

def train(base,x,table,out,kind,seed):
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.manual_seed(seed);torch.use_deterministic_algorithms(True)
    dims={k:v.shape[1] for k,v in x.items()};model=Encoder(dims,kind);opt=torch.optim.Adam(model.parameters(),lr=.001)
    vp,vl,forbidden=evaluation_examples(base);torch.save({'pairs':vp,'labels':vl},out/'validacao.pt')
    n=len(table['slots']);order=np.random.default_rng(55).permutation(n);nv=max(1,int(.2*n))
    vi=torch.tensor(order[:nv]);ti=torch.tensor(order[nv:]);torch.save({'train':ti,'val':vi},out/'aux_particao.pt')
    g=graph(x,table);cv=corrupt(table,56);gv=graph(x,cv)
    def validation():
        model.eval()
        with torch.no_grad():
            z=model(g);zc=model(gv);link=F.binary_cross_entropy_with_logits(model.decode(z,vp),vl);aux=auxiliary(model,z,zc,table,cv,vi)
        return {'loss':float(link+.25*aux),'link_loss':float(link),'aux_loss':float(aux)}
    def save(epoch,val):torch.save({'model_state':model.state_dict(),'dims':dims,'kind':kind,'seed':seed,'epoch':epoch,'validation':val},out/'melhor_modelo.pt')
    started=time.perf_counter();best=validation();save(0,best);best_epoch=0;stale=0;history=[];rng=np.random.default_rng(42)
    nu,ne=x['user'].shape[0],x['event'].shape[0]
    for epoch in range(1,61):
        model.train();opt.zero_grad(set_to_none=True)
        negative=amostrar_negativos(nu,ne,base['train_positive'].shape[1],forbidden,rng)
        pairs,labels=exemplos(base['train_positive'],negative)
        tc=corrupt(table,57+epoch);z=model(g);zc=model(graph(x,tc))
        link=F.binary_cross_entropy_with_logits(model.decode(z,pairs),labels);aux=auxiliary(model,z,zc,table,tc,ti);loss=link+.25*aux
        if not torch.isfinite(loss):raise ValueError('loss não finita')
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);opt.step();val=validation()
        history.append({'epoch':epoch,'train_link':float(link.detach()),'train_aux':float(aux.detach()),**val})
        if val['loss']<best['loss']-.0001:best=val;best_epoch=epoch;stale=0;save(epoch,val)
        else:stale+=1
        if stale>=10:break
    saved=torch.load(out/'melhor_modelo.pt',weights_only=False);model.load_state_dict(saved['model_state']);model.eval()
    with torch.no_grad():z={k:v.numpy() for k,v in model(g).items()}
    np.savez_compressed(out/'embeddings.npz',**z);pd.DataFrame(history).to_csv(out/'historico.csv',index=False)
    manifest={'kind':kind,'seed':seed,'parameters':sum(v.numel() for v in model.parameters()),'seconds':time.perf_counter()-started,'epochs_run':len(history),'best_epoch':best_epoch,'validation':best,'teste_avaliado':False,'checkpoint_sha256':sha(out/'melhor_modelo.pt')}
    write_json(out/'manifest.json',manifest);return z['user'],manifest
