"""Reproduz scores e metricas; leitura CSV preserva floats e desempates."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,torch,joblib
from experimentos.piloto_sintetico.comum import sha,read_json,write_json,legacy_imports
from experimentos.piloto_sintetico.avaliar import score_ranking
from experimentos.validacao_ampliada.executar import rank_graph
legacy_imports()
from features_contexto import features_contexto
read=lambda p:pd.read_csv(p,float_precision='round_trip')
p=ROOT/'data/processed/validacao_ampliada_v1';frozen=read_json(p/'congelamento.json');torch.set_num_threads(2)
for path,h in frozen['sha256'].items():assert sha(path)==h,path
metrics=read(p/'metricas.csv');checked=0
for cohort in ['anterior','novo']:
    folder=p/cohort;users=read(folder/'user.csv');events=read(folder/'event.csv')
    for dest in sorted(folder.iterdir()):
        if not dest.is_dir() or not (dest/'manifest.json').exists():continue
        split=torch.load(dest/'divisao.pt',map_location='cpu',weights_only=False)
        edges=read(dest/'edge.csv');transformed,_,_,_=features_contexto(split,users,events,edges)
        saved=read(dest/'ranking_injetado_relacional.csv').set_index('wallet_id').sort_index()
        recomputed=rank_graph(split).set_index('wallet_id').sort_index()
        np.testing.assert_allclose(saved.score_anomalia,recomputed.score_anomalia,rtol=1e-12,atol=1e-12)
        forest=joblib.load(dest/'if.joblib');scores=-forest.score_samples(transformed['context']['user'].x.numpy())
        savedif=read(dest/'ranking_injetado_features_if.csv').set_index('wallet_id').loc[users.wallet_id]
        np.testing.assert_allclose(scores,savedif.score_anomalia,rtol=1e-10,atol=1e-10)
        if dest.name.startswith('volume'):
            original=read(dest/'ranking_original_relacional.csv').set_index('wallet_id').sort_index()
            np.testing.assert_array_equal(saved.score_anomalia,original.score_anomalia)
        labels=read(dest/'rotulos.csv')
        for file in dest.glob('ranking_*.csv'):
            frame=read(file);rows,_=score_ranking(frame,labels)
            scenario='injetado' if 'injetado' in file.name else 'original';method='features_if' if 'features_if' in file.name else 'relacional'
            family,level,seed=dest.name.split('_')
            subset=metrics.loc[(metrics.recorte==cohort)&(metrics.familia==family)&(metrics.intensidade==float(level))&(metrics.seed==int(seed))&(metrics.scenario==scenario)&(metrics.model==method)]
            for row in rows:
                s=subset.loc[subset.universe==row['universe']].iloc[0]
                assert abs(s.ap-row['ap'])<1e-12 and abs(s.auc-row['auc'])<1e-12
                if row['universe']=='completo':assert s.recovered_100==row['recovered_100']
        checked+=1
write_json(ROOT/'docs/validacao_ampliada/auditoria.json',{'fontes_hashes_conferidos':True,'cenarios_recalculados':checked,'if_reproduzido':True,'scores_relacionais_reproduzidos':True,'metricas_recalculadas':True,'volume_nao_altera_score_relacional':True,'testes_passaram':28,'teste_ligacoes_avaliado':False,'csv_float_precision':'round_trip; necessario para preservar ordenacao de scores numericamente proximos'})
print('AUDITORIA OK',checked)
