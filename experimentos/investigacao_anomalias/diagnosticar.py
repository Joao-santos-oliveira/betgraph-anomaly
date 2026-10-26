"""Diagnosticos de desenvolvimento; nao escolhe parametros automaticamente."""
import argparse
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from modelo_gnn import MODELOS
from preparar_gnn import FORWARD
from .relacional import scores_relacionais


def avaliar_scores(scores, ids, labels):
    y = labels.set_index('wallet_id').loc[ids, 'alvo'].to_numpy()
    order = np.lexsort((ids, -scores))
    return {'ap': float(average_precision_score(y,scores)), 'auc':float(roc_auc_score(y,scores)),
            'top50':int(y[order[:50]].sum()),'top100':int(y[order[:100]].sum())}


def diagnosticar(run, output):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    labels=pd.read_csv(run/'rotulos.csv')
    rows=[]; summaries=[]; shifts=[]
    splits={s:torch.load(run/s/'resultados/divisao.pt',map_location='cpu',weights_only=False) for s in ('original','injetado')}
    ids=splits['original']['context']['user'].original_id.numpy()
    aligned=labels.set_index('wallet_id').loc[ids]
    target=aligned.alvo.to_numpy().astype(bool)
    for scenario,split in splits.items():
        graph=split['context']; pairs=graph[FORWARD].edge_index.numpy()
        structural=scores_relacionais(pairs,graph['user'].num_nodes,graph['event'].num_nodes)
        sf=pd.DataFrame({'wallet_id':ids,**structural})
        sf.to_csv(output/f'estrutural_{scenario}.csv',index=False)
        for name in ('surpresa_coparticipacao','contagem_coparticipacao'):
            score=structural[name]
            rows.append({'scenario':scenario,'method':name,'seed':None,**avaliar_scores(score,ids,labels)})
        features=pd.read_csv(run/scenario/'resultados/features_contexto.csv').set_index('wallet_id').loc[ids]
        for kind in ('gnn','mlp'):
            for seed in (42,73,2026):
                folder=run/scenario/'resultados'/f'{kind}_seed_{seed}'
                rank=pd.read_csv(folder/'ranking/ranking.csv').set_index('wallet_id').loc[ids]
                score=rank.score_anomalia.to_numpy()
                rows.append({'scenario':scenario,'method':kind+'_if','seed':seed,**avaliar_scores(score,ids,labels)})
                z=np.load(folder/'ranking/embeddings.npz')['user']
                lof=LocalOutlierFactor(n_neighbors=20,n_jobs=1)
                lof.fit_predict(StandardScaler().fit_transform(z))
                alt=-lof.negative_outlier_factor_
                rows.append({'scenario':scenario,'method':kind+'_lof20','seed':seed,**avaliar_scores(alt,ids,labels)})
                top=np.lexsort((ids,-score))[:100]
                summaries.append({'scenario':scenario,'model':kind,'seed':seed,
                    'spearman_score_grau':float(pd.Series(score).corr(pd.Series(structural['grau_contexto']),method='spearman')),
                    'spearman_score_volume':float(pd.Series(score).corr(pd.Series(features.volume_total_usd.to_numpy()),method='spearman')),
                    'mediana_grau_top100':float(np.median(structural['grau_contexto'][top])),
                    'mediana_grau_alvos':float(np.median(structural['grau_contexto'][target])),
                    'mediana_volume_top100':float(np.median(features.volume_total_usd.to_numpy()[top])),
                    'mediana_volume_alvos':float(np.median(features.volume_total_usd.to_numpy()[target]))})
                if kind=='gnn':
                    ck=torch.load(folder/'melhor_modelo.pt',map_location='cpu',weights_only=False)
                    model=MODELOS[kind](**ck['model_config']);model.load_state_dict(ck['model_state']);model.eval()
                    with torch.no_grad():
                        embed=model(graph.x_dict,graph.edge_index_dict)
                        losses=torch.nn.functional.softplus(-model.decode(embed,graph[FORWARD].edge_index)).numpy()
                    deg=structural['grau_contexto']
                    rec=np.bincount(pairs[0],weights=losses,minlength=len(ids))/np.maximum(deg,1)
                    rows.append({'scenario':scenario,'method':'gnn_erro_contexto','seed':seed,**avaliar_scores(rec,ids,labels)})
    # Congela encoder original; varia so x, so arestas, ou ambos para separar efeitos.
    for seed in (42,73,2026):
        folder=run/'original/resultados'/f'gnn_seed_{seed}'
        ck=torch.load(folder/'melhor_modelo.pt',map_location='cpu',weights_only=False)
        model=MODELOS['gnn'](**ck['model_config']);model.load_state_dict(ck['model_state']);model.eval()
        original=splits['original']['context'];changed=splits['injetado']['context']
        with torch.no_grad():
            base=model(original.x_dict,original.edge_index_dict)['user'].numpy()
            for name,x,edges in [('so_features',changed.x_dict,original.edge_index_dict),
                                  ('so_topologia',original.x_dict,changed.edge_index_dict),
                                  ('ambos',changed.x_dict,changed.edge_index_dict)]:
                z=model(x,edges)['user'].numpy()
                distance=np.linalg.norm(z-base,axis=1)
                for group,mask in [('alvo',target),('doadora',aligned.doadora.to_numpy()),('demais',~(target|aligned.doadora.to_numpy()))]:
                    shifts.append({'seed':seed,'intervencao':name,'grupo':group,'n':int(mask.sum()),
                        'distancia_media':float(distance[mask].mean()),'distancia_mediana':float(np.median(distance[mask]))})
    pd.DataFrame(rows).to_csv(output/'metricas_diagnostico.csv',index=False)
    pd.DataFrame(summaries).to_csv(output/'atividade_top.csv',index=False)
    pd.DataFrame(shifts).to_csv(output/'efeitos_encoder_congelado.csv',index=False)
    print(pd.DataFrame(rows).groupby(['scenario','method'])[['ap','auc','top50','top100']].mean().to_string(),flush=True)
    print(pd.DataFrame(summaries).groupby(['scenario','model']).mean(numeric_only=True).to_string(),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();diagnosticar(args.run,args.output)
