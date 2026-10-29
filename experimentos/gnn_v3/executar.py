"""Rodada única: congelar, preparar, treinar, avaliar e auditar sem tuning."""
import argparse
import concurrent.futures
import copy
import json
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from experimentos.piloto_sintetico.comum import ROOT,sha,read_json,write_json,version,legacy_imports
from experimentos.piloto_sintetico.gerar import Inviavel
from .dados import catalog,prepare,intervene,FWD
from .modelos import train,evaluation_examples,Encoder,graph
from .scores import manual,pair_score,metrics,ranking
legacy_imports()
from baseline import ajustar_ranking
INPUT=ROOT/'data/processed/validacao_ampliada_v1/novo'
OUT=ROOT/'data/processed/gnn_v3_v1'
DOC=ROOT/'docs/gnn_v3'
SEEDS=(42,73,2026)
SCENARIOS=('S1','S2','S3_estrutura','S3_volume')

def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()

def freeze():
    assert git('branch','--show-current')=='feat/gnn-v2-anomalias'
    assert git('rev-parse','HEAD').startswith('c0b7390')
    assert not (DOC/'congelamento.json').exists() and not OUT.exists(),'Congelamento já existe: não sobrescrever'
    protected=git('ls-files').splitlines()
    new=list((ROOT/'experimentos/gnn_v3').glob('*.py'))+[ROOT/'tests/test_gnn_v3.py',DOC/'protocolo.md',DOC/'agrupamento_mercados.md',DOC/'protocolo_v1.md',DOC/'congelamento_v1.json']
    inputs=[INPUT/f for f in ('user.csv','event.csv','edge.csv','divisao.pt')]+[ROOT/'data/raw/betgraph_nos_evento_16set_final.csv']
    saved={'time_utc':pd.Timestamp.now(tz='UTC').isoformat(),'branch':git('branch','--show-current'),'commit':git('rev-parse','HEAD'),'python':platform.python_version(),'versions':{p:version(p) for p in ('torch','torch-geometric','numpy','pandas','scipy','scikit-learn','joblib')},'protected':{p:sha(ROOT/p) for p in protected},'new_code':{str(p.relative_to(ROOT)):sha(p) for p in new},'inputs':{str(p.relative_to(ROOT)):sha(p) for p in inputs},'reserved_realizations_opened':False,'teste_avaliado':False}
    OUT.mkdir(parents=True,exist_ok=False);write_json(DOC/'congelamento.json',saved);write_json(OUT/'congelamento.json',saved)
    print('Protocolo e código congelados antes das métricas.',flush=True)

def verify():
    saved=read_json(DOC/'congelamento.json');assert saved==read_json(OUT/'congelamento.json')
    assert git('rev-parse','HEAD')==saved['commit'] and git('branch','--show-current')==saved['branch']
    for group in ('protected','new_code','inputs'):
        for name,h in saved[group].items():assert sha(ROOT/name)==h,f'Hash alterado: {name}'
    return saved

def make_case(dest,split,users,events,edges,cat):
    edges.to_csv(dest/'edge.csv',index=False)
    edges=pd.read_csv(dest/'edge.csv')
    base,x,table,scalers,feature_report=prepare(split,users,events,edges,cat)
    torch.save({'base':base,'x':x,'table':table},dest/'entradas.pt');joblib.dump(scalers,dest/'scalers.joblib')
    torch.save(split,dest/'divisao.pt')
    vp,vl,_=evaluation_examples(base);torch.save({'pairs':vp,'labels':vl},dest/'validacao_comum.pt')
    write_json(dest/'features.json',feature_report)
    ids=base['context']['user'].original_id.numpy();users=pd.DataFrame({'wallet_id':ids})
    started=time.perf_counter();ranks=manual(base,edges);manual_time=time.perf_counter()-started
    for key,r in ranks.items():r.to_csv(dest/f'ranking_{key}.csv',index=False)
    model,r,times=ajustar_ranking(users,base['context']['user'].x.numpy(),seed=42,trees=100)
    r.to_csv(dest/'ranking_E_features.csv',index=False);joblib.dump(model,dest/'features_if.joblib')
    write_json(dest/'custos_baselines.json',{'manual_cinco_scores_seconds':manual_time,'features_if':times,'parametros_treinaveis_neurais_baselines':0})

def prepare_all():
    verify();assert not (OUT/'casos.json').exists(),'Preparação já concluída'
    split=torch.load(INPUT/'divisao.pt',weights_only=False,map_location='cpu')
    users,events,edges=[pd.read_csv(INPUT/f'{k}.csv') for k in ('user','event','edge')]
    cat=catalog(split,edges);write_json(OUT/'catalogo_outcomes.json',cat)
    torch.set_num_threads(2);statuses=[]
    original=OUT/'original';original.mkdir(exist_ok=False);make_case(original,split,users,events,edges,cat)
    statuses.append({'caso':'original','status':'preparado'})
    for scenario in SCENARIOS:
        for seed in (30101,30201):
            name=f'{scenario}_{seed}';dest=OUT/name;dest.mkdir(exist_ok=False)
            try:
                altered,changed,swaps,labels,diag,ledger=intervene(split,users,edges,scenario,seed,cat)
                labels.to_csv(dest/'rotulos.csv',index=False);swaps.to_csv(dest/'trocas.csv',index=False);ledger.to_csv(dest/'atributos_alterados.csv',index=False)
                write_json(dest/'injecao.json',diag);make_case(dest,altered,users,events,changed,cat)
                statuses.append({'caso':name,'cenario':scenario,'realizacao':seed,'status':'preparado'})
            except Inviavel as exc:
                write_json(dest/'inviavel.json',{'reason':str(exc),'diagnostic':exc.diagnostic})
                statuses.append({'caso':name,'cenario':scenario,'realizacao':seed,'status':'nao_concluido','motivo':str(exc)})
            write_json(OUT/'casos.json',statuses);print(statuses[-1],flush=True)
    verify()

def worker(case,kind,seed):
    verify();folder=OUT/case;data=torch.load(folder/'entradas.pt',weights_only=False,map_location='cpu');base=data['base'];dest=folder/f'{kind}_{seed}'
    assert not dest.exists(),f'Treino não pode ser sobrescrito: {dest}'
    if kind=='E_v2':
        from experimentos.gnn_v2.features import enriquecer
        from experimentos.gnn_v2.treino import treinar
        from experimentos.gnn_v2.detectores import hibrido
        enriched,bundle,_=enriquecer(base);z,report=treinar(enriched,dest,'gnn',seed,contrast=True)
        joblib.dump(bundle,dest/'scaler_estrutural.joblib');g=base['context']
        r=ranking(g['user'].original_id.numpy(),hibrido(g[FWD].edge_index.numpy(),g['user'].num_nodes,g['event'].num_nodes,z));r.to_csv(dest/'ranking.csv',index=False)
    else:
        z,report=train(base,data['x'],data['table'],dest,kind,seed);pair_score(base,z).to_csv(dest/'ranking.csv',index=False)
        model,r,times=ajustar_ranking(pd.DataFrame({'wallet_id':base['context']['user'].original_id.numpy()}),z,seed=42,trees=100)
        r.to_csv(dest/'ranking_IF.csv',index=False);joblib.dump(model,dest/'if.joblib');write_json(dest/'custo_IF.json',times)
    common=torch.load(folder/'validacao_comum.pt',weights_only=False);actual=torch.load(dest/'validacao.pt',weights_only=False)
    for key in ('pairs','labels'):assert torch.equal(common[key],actual[key])
    write_json(dest/'concluido.json',{'validacao_identica':True,'teste_avaliado':False,'kind':kind,'seed':seed})
    return {'caso':case,'metodo':kind,'seed':seed,'status':'concluido','seconds':report['seconds'],'parameters':report['parameters'],'epochs':report['epochs_run'],'best_epoch':report['best_epoch']}

def train_all():
    verify();tasks=[]
    for case in read_json(OUT/'casos.json'):
        if case['status']=='preparado':
            for kind in ('C','D','E_v2'):
                for seed in SEEDS:tasks.append((case['caso'],kind,seed))
    # A restart may only skip fully completed jobs, never retrain a partially completed job.
    history=read_json(OUT/'treinos.json') if (OUT/'treinos.json').exists() else []
    completed={(r['caso'],r['metodo'],r['seed']) for r in history}
    tasks=[t for t in tasks if t not in completed]
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(worker,*t):t for t in tasks}
        for future in concurrent.futures.as_completed(futures):
            case,kind,seed=futures[future]
            try:row=future.result()
            except Exception as exc:
                row={'caso':case,'metodo':kind,'seed':seed,'status':'nao_concluido','motivo':str(exc)}
                (OUT/f'erro_{case}_{kind}_{seed}.txt').write_text(''.join(traceback.format_exception(exc)))
            history.append(row);write_json(OUT/'treinos.json',history);print(row,flush=True)
    verify()

def rankings(case):
    folder=OUT/case;result={}
    for method in ('A','B','O','T','V','E_features'):
        path=folder/f'ranking_{method}.csv'
        if path.exists():result[(method,0)]=pd.read_csv(path)
    for kind in ('C','D','E_v2'):
        for seed in SEEDS:
            d=folder/f'{kind}_{seed}'
            if not (d/'concluido.json').exists():continue
            result[(kind,seed)]=pd.read_csv(d/'ranking.csv')
            if kind!='E_v2':result[(kind+'_IF',seed)]=pd.read_csv(d/'ranking_IF.csv')
    return result

def evaluate():
    verify();records=[];incremental=[];original=rankings('original')
    for case in read_json(OUT/'casos.json'):
        if case.get('cenario') is None or case['status']!='preparado':continue
        name=case['caso'];labels=pd.read_csv(OUT/name/'rotulos.csv');ranks=rankings(name)
        for stage,collection in [('antes',original),('injetado',ranks)]:
            for (method,seed),r in collection.items():
                for row in metrics(r,labels):records.append({'caso':name,'cenario':case['cenario'],'realizacao':case['realizacao'],'fase':stage,'metodo':method,'seed_modelo':seed,**row})
        targets=set(labels.loc[labels.alvo.eq(1),'wallet_id'])
        for (method,seed),r in ranks.items():
            if method not in ('C','C_IF'):continue
            for universe,ids in [('completo',set(labels.wallet_id)),('elegivel',set(labels.loc[labels.elegivel.astype(bool),'wallet_id'])),('sem_doadoras',set(labels.loc[~labels.doadora.astype(bool),'wallet_id']))]:
                top=set(r.loc[r.wallet_id.isin(ids)].head(100).wallet_id);a=set(ranks[('A',0)].loc[lambda x:x.wallet_id.isin(ids)].head(100).wallet_id);b=set(ranks[('B',0)].loc[lambda x:x.wallet_id.isin(ids)].head(100).wallet_id)
                incremental.append({'caso':name,'cenario':case['cenario'],'realizacao':case['realizacao'],'metodo':method,'seed_modelo':seed,'universo':universe,'novos_vs_A':len((top-a)&targets),'novos_vs_B':len((top-b)&targets),'novos_fora_uniao_AB':len((top-a-b)&targets),'overlap_A':len(top&a),'overlap_B':len(top&b)})
    frame=pd.DataFrame(records);frame.to_csv(OUT/'metricas.csv',index=False);frame.to_csv(DOC/'metricas.csv',index=False)
    inc=pd.DataFrame(incremental);inc.to_csv(OUT/'incremental.csv',index=False);inc.to_csv(DOC/'incremental.csv',index=False)
    grouped=frame.groupby(['cenario','fase','metodo','universo'])[['top50','top100','recall100','precision50','AP','AUC']].agg(['mean','min','max']);grouped.columns=['_'.join(c) for c in grouped.columns];grouped.reset_index().to_csv(DOC/'resumo_metricas.csv',index=False)
    by_realization=frame.groupby(['cenario','realizacao','fase','metodo','universo'])[['top50','top100','recall100','precision50','AP','AUC']].agg(['mean','min','max']);by_realization.columns=['_'.join(c) for c in by_realization.columns];by_realization.reset_index().to_csv(DOC/'resumo_por_realizacao.csv',index=False)
    checks=[]
    def find(s,r,m,seed):
        f=frame.loc[(frame.cenario==s)&(frame.realizacao==r)&(frame.metodo==m)&(frame.seed_modelo==seed)&(frame.fase=='injetado')&(frame.universo=='completo')]
        return None if len(f)!=1 else f.iloc[0]
    for realization in (30101,30201):
        for seed in SEEDS:
            c=find('S1',realization,'C',seed);a=find('S1',realization,'A',0);b=find('S1',realization,'B',0)
            ok=None if any(x is None for x in (c,a,b)) else bool(c.recall100>a.recall100 and c.recall100>b.recall100 and c.AP>a.AP and c.AP>b.AP)
            checks.append({'condicao':'S1_supera_A_B','realizacao':realization,'seed':seed,'passou':ok})
            c=find('S2',realization,'C',seed);checks.append({'condicao':'S2_zero_decoys_top100','realizacao':realization,'seed':seed,'passou':None if c is None else bool(c.top100==0)})
            for s,ref,rs in [('S3_estrutura','E_v2',seed),('S3_volume','E_features',0)]:
                c=find(s,realization,'C',seed);r=find(s,realization,ref,rs);ok=None if c is None or r is None else bool(c.top100>=r.top100-5 and c.AP>=.9*r.AP)
                checks.append({'condicao':s+'_sem_piora_relevante','realizacao':realization,'seed':seed,'passou':ok})
    decision={'utilidade_demonstrada':all(r['passou'] is True for r in checks),'resultado':'positivo' if all(r['passou'] is True for r in checks) else 'negativo' if any(r['passou'] is False for r in checks) else 'inconclusivo','checks':checks,'teste_avaliado':False,'realizacoes_reservadas_abertas':False}
    write_json(DOC/'decisao.json',decision);write_json(OUT/'decisao.json',decision)
    report(frame,inc,decision)
    print('Avaliação concluída; decisão: '+decision['resultado'],flush=True)

def report(frame,inc,decision):
    lines=['# Resultado — GNN v3, rodada única','',f"Decisão pelo protocolo: **{decision['resultado']}**. Utilidade adicional demonstrada: **{'sim' if decision['utilidade_demonstrada'] else 'não'}**.",'','As métricas abaixo medem recuperação de intervenções artificiais, não fraude. S2 mede priorização indesejada de decoys. Seeds dos encoders: 42, 73, 2026; realizações 30101/30201. IF seed42. Nenhum ajuste após resultados.','', '## Todos os métodos e cenários','', 'Universo completo, após injeção. Média [mínimo–máximo] sobre realizações e seeds disponíveis; baselines determinísticos têm duas observações, encoders até seis. Top100 conta alvos/decoys de50. AP e AUC em escala0–1. Valores completos antes/depois e universos elegível/sem doadoras estão em metricas.csv; resumo_metricas.csv contém média/mín/máx para todas as métricas; resumo_por_realizacao.csv resume as três seeds separadamente por realização.','', '| Cenário | Método | n | top50 | top100 | recall100 | precision50 | AP | AUC |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    complete=frame.loc[(frame.fase=='injetado')&(frame.universo=='completo')]
    for scenario in SCENARIOS:
        for method in ('A','B','O','T','V','C','C_IF','D','D_IF','E_v2','E_features'):
            rows=complete.loc[(complete.cenario==scenario)&(complete.metodo==method)]
            if len(rows)==0:lines.append(f'| {scenario} | {method} | 0 | não concluído | — | — | — | — | — |');continue
            vals=[]
            for key in ('top50','top100','recall100','precision50','AP','AUC'):
                v=rows[key];vals.append(f'{v.mean():.3f} [{v.min():.3f}–{v.max():.3f}]')
            lines.append(f'| {scenario} | {method} | {len(rows)} | '+' | '.join(vals)+' |')
    lines+=['','A/B são os comparadores primários; O/T/V são canais isolados pré-especificados. C usa similaridade aprendida; C_IF é secundário e não substitui C na decisão. D recebe os mesmos agregados sem mensagens. E_v2 é a referência contrastiva híbrida anterior; E_features é features+IsolationForest.','', '## Verificação da regra','', '| Condição | Passou | Falhou | Não concluída |','|---|---:|---:|---:|']
    for cond in sorted({r['condicao'] for r in decision['checks']}):
        rs=[r for r in decision['checks'] if r['condicao']==cond];lines.append(f"| {cond} | {sum(r['passou'] is True for r in rs)} | {sum(r['passou'] is False for r in rs)} | {sum(r['passou'] is None for r in rs)} |")
    lines+=['','A regra exige superar A e B em recall100 e AP em todas as seis combinações de S1, zero decoys no top100 de S2 e limites de regressão em S3. Consulte protocolo.md e decisao.json para o limiar e cada combinação.','', '## Ganho incremental e overlap','', '| Cenário | Método | Novos fora A+B (média/mín/máx) | Overlap A (média) | Overlap B (média) |','|---|---|---:|---:|---:|']
    for (s,m),v in inc.loc[inc.universo.eq('completo')].groupby(['cenario','metodo']):lines.append(f"| {s} | {m} | {v.novos_fora_uniao_AB.mean():.2f}/{v.novos_fora_uniao_AB.min()}/{v.novos_fora_uniao_AB.max()} | {v.overlap_A.mean():.2f} | {v.overlap_B.mean():.2f} |")
    history=read_json(OUT/'treinos.json');pd.DataFrame(history).to_csv(DOC/'custos_treinos.csv',index=False)
    lines+=['','Ganhos separados contra A/B, por seed e universo, estão em incremental.csv. Novos alvos podem coexistir com perda de outros; isso não prova superioridade global.','', '## Execução e custos','']
    for kind in ('C','D','E_v2'):
        rs=[r for r in history if r['metodo']==kind and r['status']=='concluido']
        if rs:lines.append(f"- {kind}: {len(rs)} treinos, parâmetros {min(r['parameters'] for r in rs)}–{max(r['parameters'] for r in rs)}; soma dos tempos de treino {sum(r['seconds'] for r in rs):.1f}s; média {np.mean([r['seconds'] for r in rs]):.1f}s. Inclui original e intervenções; até três processos CPU em paralelo. IF e scores adicionais têm custos separados nos artefatos.")
    missing=[r for r in history if r['status']!='concluido']+[r for r in read_json(OUT/'casos.json') if r['status']!='preparado']
    lines+=['',f'Casos/treinos não concluídos: {len(missing)}. Detalhes: casos.json e treinos.json nos artefatos; não houve substituição de seeds.','', '## Limites e interpretação','', '- Um único dia, 5.000 carteiras. Novas realizações de injeção não são novos dias nem validação externa.','- Atributos descrevem arestas agregadas: primeiro horário e volume total, não sequência de apostas.','- Dois lados locais por mercado. O nome de uma equipe não estabelece correspondência entre mercados.','- S2 balanceia lados; não torna50carteiras mutuamente opostas. A regra de zero decoys é deliberadamente estrita, definida antes dos resultados.','- Seleção de checkpoint auto-supervisionada transdutiva; loss de ligação/auxiliar não é qualidade do ranking. O auxiliar de validação permanece no contexto de mensagens.','- Semelhança alta entre embeddings pode refletir comportamento comum. O score principal foi fixado antes dos resultados e não foi trocado pelo IF.','- A hipótese de sinais individualmente fracos deve ser confrontada com O/T/V; canais isolados fortes enfraquecem essa premissa.','- Resultado negativo rejeita utilidade desta configuração sob este protocolo; não prova impossibilidade de toda GNN.','- Teste final de ligações e realizações30301/30401 permanecem reservados. Agrupamento e nulo estratificado foram apenas propostos.','', '## Reprodução e arquivos de origem','', 'Na raiz do repositório, com a .venv e os mesmos arquivos de entrada:','', '```bash','.venv/bin/python -B -m unittest discover -s tests -p "test_gnn_v3.py" -v','.venv/bin/python -B -m experimentos.gnn_v3.executar congelar','.venv/bin/python -B -m experimentos.gnn_v3.executar preparar','.venv/bin/python -B -m experimentos.gnn_v3.executar treinar','.venv/bin/python -B -m experimentos.gnn_v3.executar avaliar','.venv/bin/python -B -m experimentos.gnn_v3.executar auditar','```','', 'Congelar/preparar recusam sobrescrever uma rodada existente. Para reprodução integral, usar cópia limpa da base c0b7390 com estes novos arquivos e mesmas dependências; nunca remover a rodada original para substituí-la. Avaliar/auditar apenas recalculam a partir dos artefatos existentes.','', '- Entradas: data/processed/validacao_ampliada_v1/novo/{user,event,edge}.csv e divisao.pt.','- Execução: data/processed/gnn_v3_v1/<cenário>_<realização>/; original/ para comparações antes da injeção.','- Scores: ranking_A/B/O/T/V/E_features.csv e C/D/E_v2_<seed>/ranking.csv; variantes IF em ranking_IF.csv.','- Custos: manifest.json de cada treino, custo_IF.json e custos_baselines.json; resumo custos_treinos.csv.','- Decisão/métricas: metricas.csv, resumo_metricas.csv, incremental.csv, decisao.json nesta pasta.','- Congelamento: congelamento.json registra SHA-256 das entradas, código novo, protocolo e arquivos anteriores. Auditoria final: auditoria.json. Saída de testes: testes.txt.','- Implementação da convolução: documentação oficial https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.TransformerConv.html.','', 'Nenhum commit ou push foi realizado nesta tarefa.']
    (DOC/'resultado.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def audit():
    saved=verify();torch.set_num_threads(2);checked=[]
    for case in read_json(OUT/'casos.json'):
        if case['status']!='preparado':continue
        folder=OUT/case['caso'];data=torch.load(folder/'entradas.pt',weights_only=False);base=data['base'];ranks=rankings(case['caso'])
        ids=base['context']['user'].original_id.numpy();common=torch.load(folder/'validacao_comum.pt',weights_only=False)
        def same(actual,expected):
            a=actual.set_index('wallet_id').sort_index();b=expected.set_index('wallet_id').sort_index();assert a.index.equals(b.index);assert np.allclose(a.score_anomalia,b.score_anomalia,rtol=1e-6,atol=1e-7);assert np.array_equal(actual.wallet_id.to_numpy(),expected.wallet_id.to_numpy())
        m=manual(base,pd.read_csv(folder/'edge.csv'))
        for kind,r in m.items():same(r,ranks[(kind,0)])
        model=joblib.load(folder/'features_if.joblib');same(ranking(ids,-model.score_samples(base['context']['user'].x.numpy())),ranks[('E_features',0)])
        for kind in ('C','D','E_v2'):
            for seed in SEEDS:
                if (kind,seed) not in ranks:continue
                d=folder/f'{kind}_{seed}';c=torch.load(d/'melhor_modelo.pt',weights_only=False);v=torch.load(d/'validacao.pt',weights_only=False)
                for key in ('pairs','labels'):assert torch.equal(common[key],v[key])
                assert read_json(d/'manifest.json')['teste_avaliado'] is False
                if kind=='E_v2':
                    from modelo_gnn import MODELOS
                    from experimentos.gnn_v2.features import enriquecer
                    from experimentos.gnn_v2.detectores import hibrido
                    enriched,_,_=enriquecer(base);model=MODELOS['gnn'](**c['config']);model.load_state_dict(c['model_state']);model.eval()
                    with torch.no_grad():z=model(enriched['context'].x_dict,enriched['context'].edge_index_dict)['user'].numpy()
                    g=base['context'];r=ranking(ids,hibrido(g[FWD].edge_index.numpy(),g['user'].num_nodes,g['event'].num_nodes,z))
                else:
                    model=Encoder(c['dims'],kind);model.load_state_dict(c['model_state']);model.eval()
                    with torch.no_grad():z=model(graph(data['x'],data['table']))['user'].numpy()
                    r=pair_score(base,z);im=joblib.load(d/'if.joblib');same(ranking(ids,-im.score_samples(z)),ranks[(kind+'_IF',seed)])
                same(r,ranks[(kind,seed)]);assert np.allclose(z,np.load(d/'embeddings.npz')['user'],rtol=1e-6,atol=1e-7)
                checked.append({'caso':case['caso'],'metodo':kind,'seed':seed})
    before={name:sha(DOC/name) for name in ('metricas.csv','incremental.csv','decisao.json')};evaluate()
    assert all(sha(DOC/name)==h for name,h in before.items())
    verify();result={'hashes_intactos':True,'n_arquivos_protegidos':len(saved['protected']),'n_entradas':len(saved['inputs']),'n_codigo_novo_protocolo':len(saved['new_code']),'checkpoints_scores_reproduzidos':len(checked),'metricas_reproduzidas':True,'validacao_identica':True,'teste_avaliado':False,'realizacoes_reservadas_abertas':False,'treinos_conferidos':checked}
    write_json(DOC/'auditoria.json',result);write_json(OUT/'auditoria.json',result)
    seal={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='artefatos_sha256.json'};write_json(OUT/'artefatos_sha256.json',seal)
    print('Auditoria concluída: hashes, checkpoints, scores e métricas reproduzidos.',flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('fase',choices=('congelar','preparar','treinar','avaliar','auditar'));args=parser.parse_args()
    {'congelar':freeze,'preparar':prepare_all,'treinar':train_all,'avaliar':evaluate,'auditar':audit}[args.fase]()

if __name__=='__main__':main()
