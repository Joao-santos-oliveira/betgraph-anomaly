"""Treino modular: encoder original, objetivo opcional, checkpoint por validacao."""
import copy,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from experimentos.piloto_sintetico.comum import legacy_imports,write_json,sha
legacy_imports()
from modelo_gnn import MODELOS
from treinar_gnn import validar_divisao,amostrar_negativos,codigos,exemplos,avaliar_validacao
from .objetivos import Contraste,corromper


def treinar(split,output,kind,seed,contrast=False,epochs=100,patience=15):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(seed);torch.use_deterministic_algorithms(True)
    observed=validar_divisao(split);g=split['context'];nu,ne=g['user'].num_nodes,g['event'].num_nodes
    config={'user_features':g['user'].x.shape[1],'event_features':g['event'].x.shape[1],'hidden':64,'embedding':32,'dropout':.1}
    model=MODELOS[kind](**config)
    # Criar cabeca depois do encoder preserva inicializacao da parte comum.
    head=Contraste(32) if contrast else None
    parameters=list(model.parameters())+(list(head.parameters()) if head else [])
    optimizer=torch.optim.Adam(parameters,lr=.001)
    nv=amostrar_negativos(nu,ne,split['val_positive'].shape[1],observed,np.random.default_rng(43))
    nt=amostrar_negativos(nu,ne,split['test_positive'].shape[1],observed|codigos(nv,ne),np.random.default_rng(44))
    forbidden=observed|codigos(nv,ne)|codigos(nt,ne)
    vp,vl=exemplos(split['val_positive'],nv)
    torch.save({'pairs':vp,'labels':vl},output/'validacao.pt')
    torch.save({'pairs':nt,'positive':split['test_positive'],'avaliado':False},output/'teste_reservado.pt')
    context=g['user','bets_on','event'].edge_index
    active={'user':torch.bincount(context[0],minlength=nu)>0,'event':torch.bincount(context[1],minlength=ne)>0}
    rng=np.random.default_rng(42);corruption=torch.Generator().manual_seed(seed+10000)
    best=avaliar_validacao(model,g,vp,vl);best_epoch=0;stale=0;history=[]
    def save(epoch,metrics):
        torch.save({'model_state':model.state_dict(),'head_state':head.state_dict() if head else None,
                    'config':config,'kind':kind,'seed':seed,'contrast':contrast,'epoch':epoch,'validation':metrics},output/'melhor_modelo.pt')
    save(0,best);start=time.perf_counter()
    for epoch in range(1,epochs+1):
        model.train();optimizer.zero_grad(set_to_none=True)
        negative=amostrar_negativos(nu,ne,split['train_positive'].shape[1],forbidden,rng)
        pairs,labels=exemplos(split['train_positive'],negative)
        z=model(g.x_dict,g.edge_index_dict);link=F.binary_cross_entropy_with_logits(model.decode(z,pairs),labels)
        auxiliary=torch.tensor(0.)
        if head:
            shuffled=corromper(g.x_dict,active,corruption)
            corrupt=model(shuffled,g.edge_index_dict)
            auxiliary=head(z,corrupt,active)
        loss=link+.1*auxiliary
        if not torch.isfinite(loss):raise ValueError('Loss nao finita')
        loss.backward();torch.nn.utils.clip_grad_norm_(parameters,5.,error_if_nonfinite=True);optimizer.step()
        val=avaliar_validacao(model,g,vp,vl)
        history.append({'epoch':epoch,'link_loss':float(link.detach()),'contrast_loss':float(auxiliary.detach()),'val_loss':val['loss'],'val_ap':val['ap'],'val_auc':val['auc']})
        if val['loss']<best['loss']-.0001:best=val;best_epoch=epoch;stale=0;save(epoch,best)
        else:stale+=1
        if stale>=patience:break
    saved=torch.load(output/'melhor_modelo.pt',weights_only=False,map_location='cpu');model.load_state_dict(saved['model_state']);model.eval()
    with torch.no_grad():emb={k:v.numpy() for k,v in model(g.x_dict,g.edge_index_dict).items()}
    pd.DataFrame(history).to_csv(output/'historico.csv',index=False)
    np.savez_compressed(output/'embeddings.npz',**emb,user_original_id=g['user'].original_id.numpy(),event_original_id=g['event'].original_id.numpy())
    report={'kind':kind,'contrast':contrast,'seed':seed,'config':config,'best_epoch':best_epoch,'epochs_run':len(history),
            'seconds':time.perf_counter()-start,'parameters':sum(p.numel() for p in parameters),'validation':best,
            'teste_avaliado':False,'contrast_weight':.1 if contrast else 0.,'checkpoint_sha256':sha(output/'melhor_modelo.pt')}
    write_json(output/'manifest.json',report)
    return emb['user'],report
