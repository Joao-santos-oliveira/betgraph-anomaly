"""Avalia ligações reservadas no contexto original e aplica os pesos fixos ao dia."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experimentos.artigo.preparar import FORWARD, REVERSE, protocol
from experimentos.gnn.features import enriquecer_grafo
from experimentos.gnn.detectores import hibrido, ranking
from features_contexto import atribuir_features
from modelo_gnn import MODELOS
from treinar_gnn import avaliar_validacao, exemplos, validar_divisao


def cobertura(run, destino):
    original = run / 'original'
    split = torch.load(original / 'rede.pt', weights_only=False)
    validar_divisao(split)
    context = split['context']
    pairs = split['all_positive']
    frames = [pd.read_csv(run / 'dia' / f'{k}.csv') for k in ('user','event','edge')]
    # Reconstruir os ajustes históricos e verificar suas saídas contra o treino salvo.
    restored = copy.deepcopy(context)
    _, ajuste, _ = atribuir_features(restored, *frames)
    estrutural, _ = enriquecer_grafo(restored)
    for k in ('user','event'):
        assert torch.equal(restored[k].x, context[k].x)
    ids = context['user'].original_id.numpy()
    models, report, held_scores = {}, {}, []
    for seed in protocol['seeds_modelo']:
        path = original / f'gnn_{seed}'
        if not path.exists():
            path = original / f'v2_{seed}'
        ckpt = torch.load(path/'melhor_modelo.pt', weights_only=False)
        model = MODELOS['gnn'](**ckpt['config'])
        model.load_state_dict(ckpt['model_state'])
        model.eval()
        held = torch.load(path/'teste_reservado.pt', weights_only=False)
        assert torch.equal(held['positive'],split['test_positive'])
        inputs, labels = exemplos(held['positive'],held['pairs'])
        test = avaliar_validacao(model,context,inputs,labels)
        # Score individual de cada ligação reservada: permite reconstruir as curvas ROC e PR.
        with torch.inference_mode():
            logits = model.decode(model(context.x_dict,context.edge_index_dict),inputs)
        held_scores.append(pd.DataFrame({'seed':seed,'carteira':inputs[0].numpy(),'mercado':inputs[1].numpy(),
                                         'rotulo':labels.numpy().astype(int),'score':logits.numpy()}))
        models[seed] = (model, pd.read_csv(path/'ranking.csv'))
        report[seed] = {k:test[k] for k in ('n','positivos','auc','ap')}

    full = copy.deepcopy(context)
    full[FORWARD].edge_index = pairs
    full[REVERSE].edge_index = pairs.flip(0)
    atribuir_features(full,*frames,ajuste=ajuste)
    _, stats = enriquecer_grafo(full,estrutural['scaler'])
    data = pd.DataFrame({'wallet_id':ids})
    embeddings = {}
    for seed,(model,before) in models.items():
        with torch.inference_mode():
            z = model(full.x_dict,full.edge_index_dict)['user'].numpy()
        embeddings[f'z_{seed}'] = z
        score = hibrido(pairs.numpy(),len(ids),full['event'].num_nodes,z)
        rank = ranking(ids,score).set_index('wallet_id')
        data[f'contexto_{seed}'] = data.wallet_id.map(before.set_index('wallet_id').posicao)
        data[f'total_{seed}'] = data.wallet_id.map(rank.posicao)
        data[f'score_total_{seed}'] = score
        assert data[f'total_{seed}'].nunique()==len(ids)
        report[seed]['top100_comuns'] = len(set(before.wallet_id[:100]) & set(rank.index[:100]))
        report[seed]['primeiro_contexto'] = int(before.wallet_id.iloc[0])
        report[seed]['primeiro_total'] = int(rank.index[0])
    data['mercados_contexto'] = torch.bincount(context[FORWARD].edge_index[0],minlength=len(ids)).numpy()
    data['mercados_total'] = stats['grau_contexto'].astype(int)
    destino.mkdir(parents=True,exist_ok=True)
    data.sort_values('total_42').to_csv(destino/'cobertura_total.csv',index=False)
    (destino/'avaliacao_cobertura.json').write_text(json.dumps(report,indent=2)+'\n')
    pd.concat(held_scores,ignore_index=True).to_csv(destino/'scores_teste.csv.gz',index=False)
    np.savez_compressed(destino/'embeddings_total.npz',user_original_id=ids,**embeddings)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rodada',type=Path)
    parser.add_argument('destino',type=Path)
    args=parser.parse_args()
    cobertura(args.rodada,args.destino)
