"""Executa as duas realizacoes reservadas sem alterar o metodo congelado."""
from pathlib import Path
import json
import time
import pandas as pd
from experimentos.piloto_sintetico.comum import ROOT, sha, read_json, write_json, require
from experimentos.piloto_sintetico.gerar import generate, Inviavel
from experimentos.piloto_sintetico.executar import execute
from experimentos.piloto_sintetico.avaliar import evaluate, score_ranking
from .ranking_relacional import gerar


def conferir(freeze):
    for name, expected in freeze['sha256'].items():
        require(sha(ROOT/name)==expected, f'Metodo congelado alterado: {name}')


def main():
    freeze=read_json(ROOT/'docs/investigacao_anomalias/validacao_congelada.json')
    conferir(freeze)
    base=ROOT/'data/processed/investigacao_anomalias_v1/validacao'
    base.mkdir(parents=True,exist_ok=False)
    development=ROOT/'data/processed/sintetico/piloto_v2_8101_8102'
    all_metrics=[]; statuses=[]
    for selection,swap in freeze['realizacoes']:
        started=time.monotonic()
        name=f'piloto_v2_{selection}_{swap}'
        folder=base/name;folder.mkdir()
        protocol=read_json(ROOT/'experimentos/piloto_sintetico/protocolo_v2.json')
        protocol.update(selection_seed=selection,swap_seed=swap)
        write_json(folder/'protocolo.json',protocol)
        run=ROOT/'data/processed/sintetico'/name
        print(f'GERANDO {name}',flush=True)
        try:
            generate(folder/'protocolo.json',development/'original/entrada',
                     development/'original/entrada/divisao.pt',run)
        except Inviavel as exc:
            statuses.append({'realizacao':name,'status':'inviavel','reason':str(exc)})
            write_json(base/'status.json',statuses)
            continue
        print(f'TREINANDO {name}',flush=True)
        execute(run)
        metrics=evaluate(run)
        labels=pd.read_csv(run/'rotulos.csv')
        extra=[]
        for scenario in ('original','injetado'):
            rank=gerar(run/scenario/'resultados/divisao.pt',folder/scenario)
            for model,score in [('relacional','surpresa_coparticipacao'),('contagem','contagem_coparticipacao')]:
                frame=rank.copy();frame['score_anomalia']=frame[score]
                frame=frame.sort_values(['score_anomalia','wallet_id'],ascending=[False,True]).reset_index(drop=True)
                frame['posicao']=range(1,len(frame)+1)
                rows,_=score_ranking(frame,labels)
                extra.extend({'scenario':scenario,'model':model,'seed':None,**r} for r in rows)
        metrics=pd.concat([metrics,pd.DataFrame(extra)],ignore_index=True)
        metrics['realizacao']=name
        metrics.to_csv(folder/'metricas.csv',index=False)
        all_metrics.append(metrics)
        pd.concat(all_metrics,ignore_index=True).to_csv(base/'metricas.csv',index=False)
        conferir(freeze)
        statuses.append({'realizacao':name,'status':'concluido','seconds':time.monotonic()-started,
                         'teste_ligacoes_avaliado':False})
        write_json(base/'status.json',statuses)
        print(metrics.query("scenario=='injetado' and universe=='completo'")[['model','seed','ap','auc','recovered_50','recovered_100']].to_string(index=False),flush=True)
    conferir(freeze)
    write_json(base/'integridade.json',{'codigo_congelado_conferido':True,'status':statuses,
        'teste_ligacoes_avaliado':False,'seeds_substituidas':False})


if __name__=='__main__':
    main()
