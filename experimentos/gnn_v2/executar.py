"""Bateria preregistrada da v2, com retomada verificada e avaliacao separada."""
from pathlib import Path
import argparse,contextlib,json,time
import numpy as np
import pandas as pd
import torch,joblib
from experimentos.piloto_sintetico.comum import ROOT,legacy_imports,read_json,write_json,sha
from experimentos.piloto_sintetico.gerar import inject,Inviavel
from experimentos.piloto_sintetico.avaliar import score_ranking
from experimentos.validacao_ampliada.executar import volume_injection
from .features import enriquecer,FORWARD
from .detectores import ranking,hibrido
from .treino import treinar
legacy_imports()
from features_contexto import features_contexto
from baseline import ajustar_ranking

VARIANTS={'gnn_base':('gnn',False,False),'mlp_base':('mlp',False,False),
          'gnn_struct':('gnn',True,False),'mlp_struct':('mlp',True,False),
          'gnn_contrast':('gnn',True,True),'mlp_contrast':('mlp',True,True)}
SOURCE=ROOT/'data/processed/validacao_ampliada_v1/novo'
DEFAULT=ROOT/'data/processed/gnn_v2_v1'


def verify(freeze):
    for name,h in freeze['sha256'].items():
        if sha(ROOT/name)!=h:raise ValueError(f'Fonte mudou apos congelamento: {name}')


def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    sources=list((ROOT/'experimentos/gnn_v2').glob('*.py'))+list((ROOT/'src').glob('*.py'))
    sources+=list((ROOT/'experimentos/piloto_sintetico').glob('*.py'))
    sources+=[ROOT/'experimentos/piloto_sintetico/protocolo_v2.json',ROOT/'experimentos/validacao_ampliada/executar.py',ROOT/'experimentos/investigacao_anomalias/relacional.py',ROOT/'tests/test_gnn_v2.py',ROOT/'docs/gnn_v2/protocolo.md']
    sources += [SOURCE/f for f in ('user.csv','event.csv','edge.csv','divisao.pt')]
    freeze={'created_utc':pd.Timestamp.now(tz='UTC').isoformat(),'sha256':{str(p.relative_to(ROOT)):sha(p) for p in sources},'model_seeds':[42,73,2026],'injection_seeds':[20101,20201],'variants':VARIANTS,'test_evaluated':False}
    write_json(out/'congelamento.json',freeze)
    split=torch.load(SOURCE/'divisao.pt',map_location='cpu',weights_only=False)
    users,events,edges=[pd.read_csv(SOURCE/f'{kind}.csv',float_precision='round_trip') for kind in ('user','event','edge')]
    situations=[('original',split,edges,None,None)];statuses=[]
    for family,level in [('coparticipacao',.3),('coparticipacao',.6),('volume',10)]:
        for seed in (20101,20201):
            name=f'{family}_{level}_{seed}';dest=out/name;dest.mkdir()
            try:
                if family=='coparticipacao':
                    p=read_json(ROOT/'experimentos/piloto_sintetico/protocolo_v2.json');p.update(concentration=level,selection_seed=seed,swap_seed=seed+1)
                    changed,a,swaps,labels,diagnostic=inject(split,users,edges,p);swaps.to_csv(dest/'trocas.csv',index=False)
                else:
                    a,labels,diagnostic=volume_injection(split,edges,seed,level);changed=split
                labels.to_csv(dest/'rotulos.csv',index=False)
                situations.append((name,changed,a,labels,diagnostic))
                statuses.append({'name':name,'status':'viavel','family':family,'level':level,'seed':seed})
            except Inviavel as exc:
                write_json(dest/'inviavel.json',{'reason':str(exc),'diagnostic':exc.diagnostic});statuses.append({'name':name,'status':'inviavel','family':family,'level':level,'seed':seed})
    for name,s,a,_,diagnostic in situations:
        dest=out/name;dest.mkdir(exist_ok=True)
        base,raw,bundle,report=features_contexto(s,users,events,a)
        enriched,struct,stats=enriquecer(base)
        torch.save(base,dest/'base.pt');torch.save(enriched,dest/'struct.pt')
        joblib.dump(bundle,dest/'preprocessador.joblib');joblib.dump(struct,dest/'estrutural.joblib')
        raw.to_csv(dest/'features.csv',index=False);a.to_csv(dest/'edge.csv',index=False)
        users.to_csv(dest/'user.csv',index=False)
        mapping={k:{str(x):i for i,x in enumerate(base['context'][k].original_id.tolist())} for k in ('user','event')}
        write_json(dest/'indices_originais.json',mapping)
        write_json(dest/'preparacao.json',{'context':report,'injection':diagnostic,'input_hashes':{f:sha(dest/f) for f in ('base.pt','struct.pt','edge.csv','user.csv','preprocessador.joblib','estrutural.joblib')}})
    write_json(out/'cenarios.json',statuses);verify(freeze)
    print('PREPARADO',len(situations),'grafos',flush=True)


def fit(out):
    freeze=read_json(out/'congelamento.json');verify(freeze)
    scenarios=['original']+[s['name'] for s in read_json(out/'cenarios.json') if s['status']=='viavel']
    progress=[]
    for name in scenarios:
        dest=out/name
        for filename,h in read_json(dest/'preparacao.json')['input_hashes'].items():
            if sha(dest/filename)!=h:raise ValueError('Entrada de cenario alterada')
        base=torch.load(dest/'base.pt',weights_only=False,map_location='cpu');struct=torch.load(dest/'struct.pt',weights_only=False,map_location='cpu')
        graph=base['context'];ids=graph['user'].original_id.numpy();users=pd.read_csv(dest/'user.csv')
        # Ranked representations never receive injection labels.
        rankings=dest/'rankings';rankings.mkdir(exist_ok=True)
        if not (rankings/'baselines.json').exists():
            original_stats=joblib.load(dest/'estrutural.joblib')
            # Feature order registered in features.NAMES: surprise is column2.
            ranking(ids,original_stats['raw'][:,2]).to_csv(rankings/'relacional.csv',index=False)
            for tag,split in [('features_if',base),('struct_if',struct)]:
                model,r,_=ajustar_ranking(users,split['context']['user'].x.numpy())
                r.to_csv(rankings/f'{tag}.csv',index=False);joblib.dump(model,rankings/f'{tag}.joblib')
            scores=hibrido(graph[FORWARD].edge_index.numpy(),len(ids),graph['event'].num_nodes,struct['context']['user'].x.numpy())
            ranking(ids,scores).to_csv(rankings/'features_hibrido.csv',index=False)
            write_json(rankings/'baselines.json',{'complete':True})
        for variant,(kind,enriched,contrast) in VARIANTS.items():
            source=struct if enriched else base
            for seed in (42,73,2026):
                job=dest/f'{variant}_seed_{seed}';tag=f'{variant}_seed_{seed}'
                if not (job/'complete.json').exists():
                    if job.exists():raise RuntimeError(f'Treino parcial em {job}; preservar e investigar antes de retomar')
                    z,report=treinar(source,job,kind,seed,contrast)
                    detector,r,_=ajustar_ranking(users,z);r.to_csv(rankings/f'{tag}_if.csv',index=False);joblib.dump(detector,job/'if.joblib')
                    if enriched:
                        scores=hibrido(graph[FORWARD].edge_index.numpy(),len(ids),graph['event'].num_nodes,z)
                        ranking(ids,scores).to_csv(rankings/f'{tag}_hibrido.csv',index=False)
                    write_json(job/'complete.json',{'sha256_split':sha(dest/('struct.pt' if enriched else 'base.pt')),'sha256_ranking_if':sha(rankings/f'{tag}_if.csv'),'teste_avaliado':False})
                    print(f'CONCLUIDO {name} {tag}: {report["seconds"]:.1f}s epoca{report["best_epoch"]}',flush=True)
                progress.append({'scenario':name,'variant':variant,'seed':seed,**read_json(job/'manifest.json')})
                write_json(out/'progresso.json',progress)
        verify(freeze)
    write_json(out/'treinos_concluidos.json',{'n':len(progress),'test_evaluated':False})


def evaluate(out):
    verify(read_json(out/'congelamento.json'))
    if not (out/'treinos_concluidos.json').exists():raise ValueError('Concluir rankings antes de avaliar')
    records=[];positions=[]
    for scenario in read_json(out/'cenarios.json'):
        if scenario['status']!='viavel':continue
        name=scenario['name'];labels=pd.read_csv(out/name/'rotulos.csv')
        for condition,folder in [('original',out/'original'),('injetado',out/name)]:
            for file in sorted((folder/'rankings').glob('*.csv')):
                tag=file.stem
                if '_seed_' in tag:
                    variant,tail=tag.split('_seed_');seed,detector=tail.split('_',1);method=f'{variant}_{detector}';seed=int(seed)
                else:method=tag;seed=None
                frame=pd.read_csv(file,float_precision='round_trip');rows,joined=score_ranking(frame,labels)
                records.extend({'scenario':name,'family':scenario['family'],'level':scenario['level'],'injection_seed':scenario['seed'],'condition':condition,'method':method,'model_seed':seed,**r} for r in rows)
                target=joined.loc[joined.alvo==1,['wallet_id','posicao','score_anomalia']].copy();target['scenario']=name;target['condition']=condition;target['method']=method;target['model_seed']=seed;positions.append(target)
    pd.DataFrame(records).to_csv(out/'metricas.csv',index=False);pd.concat(positions).to_csv(out/'posicoes_alvos.csv',index=False)
    write_json(out/'avaliacao.json',{'test_evaluated':False,'synthetic_labels_evaluation_only':True,'n_rows':len(records),'n_training':read_json(out/'treinos_concluidos.json')['n']})
    print('AVALIACAO CONCLUIDA',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DEFAULT);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.resume:verify(read_json(args.output/'congelamento.json'))
    else:prepare(args.output)
    fit(args.output);evaluate(args.output)

if __name__=='__main__':main()
