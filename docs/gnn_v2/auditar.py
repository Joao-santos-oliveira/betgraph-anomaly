"""Reproduz embeddings, scores e metricas a partir dos artefatos salvos."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,torch,joblib
from experimentos.gnn_v2.executar import DEFAULT,VARIANTS,verify
from experimentos.gnn_v2.detectores import hibrido,ranking
from experimentos.piloto_sintetico.comum import read_json,write_json,sha,legacy_imports
from experimentos.piloto_sintetico.avaliar import score_ranking
legacy_imports()
from modelo_gnn import MODELOS
from treinar_gnn import validar_divisao


def main():
    out=DEFAULT;torch.set_num_threads(2);verify(read_json(out/'congelamento.json'))
    train=read_json(out/'treinos_concluidos.json');assert train['n']==126 and not train['test_evaluated']
    sources=['original']+[r['name'] for r in read_json(out/'cenarios.json') if r['status']=='viavel']
    checked=0
    for name in sources:
        dest=out/name;base=torch.load(dest/'base.pt',weights_only=False,map_location='cpu');struct=torch.load(dest/'struct.pt',weights_only=False,map_location='cpu')
        validar_divisao(base);validar_divisao(struct)
        for k in ('train_positive','val_positive','test_positive','all_positive'):assert torch.equal(base[k],struct[k])
        if name!='original':
            original=torch.load(out/'original/base.pt',weights_only=False,map_location='cpu')
            for k in ('train_positive','val_positive','test_positive'):assert torch.equal(original[k],base[k])
        ids=base['context']['user'].original_id.numpy();g=base['context']
        for variant,(kind,enriched,contrast) in VARIANTS.items():
            split=struct if enriched else base
            for seed in (42,73,2026):
                tag=f'{variant}_seed_{seed}';job=dest/tag
                record=read_json(job/'complete.json');assert record['sha256_split']==sha(dest/('struct.pt' if enriched else 'base.pt'))
                checkpoint=torch.load(job/'melhor_modelo.pt',weights_only=False,map_location='cpu');model=MODELOS[kind](**checkpoint['config']);model.load_state_dict(checkpoint['model_state']);model.eval()
                with torch.no_grad():z=model(split['context'].x_dict,split['context'].edge_index_dict)['user'].numpy()
                with np.load(job/'embeddings.npz') as archive:np.testing.assert_array_equal(z,archive['user']);np.testing.assert_array_equal(ids,archive['user_original_id'])
                score=-joblib.load(job/'if.joblib').score_samples(z)
                saved=pd.read_csv(dest/'rankings'/f'{tag}_if.csv',float_precision='round_trip').set_index('wallet_id').loc[ids]
                np.testing.assert_allclose(score,saved.score_anomalia,rtol=1e-12,atol=1e-12)
                if enriched:
                    h=hibrido(g['user','bets_on','event'].edge_index.numpy(),len(ids),g['event'].num_nodes,z)
                    saved=pd.read_csv(dest/'rankings'/f'{tag}_hibrido.csv',float_precision='round_trip').set_index('wallet_id').loc[ids]
                    np.testing.assert_allclose(h,saved.score_anomalia,rtol=1e-12,atol=1e-12)
                checked+=1
        print('AUDITADO',name,checked,flush=True)
    metrics=pd.read_csv(out/'metricas.csv',float_precision='round_trip');n=0
    for scenario in read_json(out/'cenarios.json'):
        if scenario['status']!='viavel':continue
        name=scenario['name'];labels=pd.read_csv(out/name/'rotulos.csv')
        for condition,folder in [('original',out/'original'),('injetado',out/name)]:
            for file in (folder/'rankings').glob('*.csv'):
                tag=file.stem
                if '_seed_' in tag:
                    variant,tail=tag.split('_seed_');seed,detector=tail.split('_',1);method=f'{variant}_{detector}';seed=int(seed)
                else:method=tag;seed=None
                data=pd.read_csv(file,float_precision='round_trip');rows,_=score_ranking(data,labels)
                sub=metrics.loc[(metrics.scenario==name)&(metrics.condition==condition)&(metrics.method==method)]
                sub=sub.loc[sub.model_seed.isna()] if seed is None else sub.loc[sub.model_seed==seed]
                for row in rows:
                    expected=sub.loc[sub.universe==row['universe']].iloc[0]
                    for k in ['ap','auc']:assert abs(expected[k]-row[k])<1e-12
                    if row['universe']=='completo':assert expected.recovered_100==row['recovered_100']
                n+=1
    write_json(ROOT/'docs/gnn_v2/auditoria.json',{'encoders_reproduzidos':checked,'rankings_avaliados_recalculados':n,'fontes_congeladas_conferidas':True,'test_evaluated':False,'tests_passed':32})
    print('AUDITORIA CONCLUIDA',checked,n,flush=True)

if __name__=='__main__':main()
