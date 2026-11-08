"""Benchmark CPU dos dois detectores com artefatos próprios do dia completo."""
import csv
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import hashlib

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/processed/escala_v1'
DOC = ROOT / 'resultados/escala'
STAGES = ('preparar', 'grafo', 'contexto', 'relacional', 'floresta')


def save(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def execute(stage):
    import pandas as pd
    import torch
    import joblib
    from experimentos.piloto_sintetico.comum import legacy_imports
    legacy_imports()
    from preparar_features import preparar
    from construir_hetero import construir
    from preparar_gnn import separar, FORWARD
    from features_contexto import features_contexto
    from baseline import ajustar_ranking
    from experimentos.investigacao_anomalias.relacional import scores_relacionais
    torch.set_num_threads(2)

    if stage == 'preparar':
        preparar(ROOT / 'data/raw', OUT, sample_size=0, seed=42)
        edges = pd.read_csv(OUT / 'edge.csv', usecols=['wallet_id','condition_node_id','token_outcome_id'])
        if edges.duplicated().any():
            raise ValueError('Linhas carteira–mercado–outcome duplicadas')
    elif stage == 'grafo':
        construir(OUT, ROOT)
    elif stage == 'contexto':
        graph = torch.load(OUT / 'hetero.pt', weights_only=False)
        split = separar(graph, seed=42)
        users, events, edges = [pd.read_csv(OUT / f'{k}.csv') for k in ('user','event','edge')]
        base, raw, scalers, info = features_contexto(split, users, events, edges)
        torch.save(base['context'], OUT / 'contexto.pt')
        joblib.dump(scalers, OUT / 'scalers_contexto.joblib')
        raw.to_csv(OUT / 'features_contexto.csv', index=False)
        save('contexto.json', info)
    else:
        g = torch.load(OUT / 'contexto.pt', weights_only=False)
        ids = g['user'].original_id.numpy()
        if stage == 'relacional':
            stats = scores_relacionais(g[FORWARD].edge_index.numpy(),len(ids),g['event'].num_nodes)
            frame = pd.DataFrame({'wallet_id':ids, **stats})
            frame['score_anomalia'] = frame.surpresa_coparticipacao
            partner = frame.melhor_parceiro_indice
            frame['parceiro_id'] = partner.where(partner < 0, partner.map(dict(enumerate(ids))))
            frame = frame.sort_values(['score_anomalia','wallet_id'],ascending=[False,True])
        else:
            model, frame, times = ajustar_ranking(pd.DataFrame({'wallet_id':ids}),g['user'].x.numpy())
            joblib.dump(model,OUT/'floresta.joblib')
            save('tempos_internos_if.json',times)
        frame.to_csv(OUT / f'ranking_{stage}.csv',index=False)
    save(f'pico_{stage}.json',{'rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024})


def main():
    import psutil
    import platform
    from importlib.metadata import version
    OUT.mkdir(parents=True, exist_ok=False)
    inputs = [ROOT/'data/raw'/name for name in ('betgraph_wallets_16set_nos.csv','betgraph_nos_evento_16set_final.csv','betgraph_arestas_16set_final.csv')]
    inputs += list(ROOT.glob('mapeamento_*.json')) + list((ROOT/'src').glob('*.py'))
    inputs += [Path(__file__).resolve(), DOC/'protocolo.json', ROOT/'experimentos/investigacao_anomalias/relacional.py', ROOT/'experimentos/piloto_sintetico/comum.py']
    hashes = {str(path.relative_to(ROOT)):digest(path) for path in inputs}
    save('congelamento.json',{'sha256':hashes,'python':platform.python_version(),'versoes':{k:version(k) for k in ('torch','torch-geometric','numpy','scikit-learn','pandas','scipy','psutil')},'host':platform.machine(),'ram_bytes':psutil.virtual_memory().total})
    env = dict(os.environ, OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    costs = {}
    for stage in STAGES:
        start = time.monotonic()
        with (OUT/f'{stage}.log').open('w') as log:
            proc = subprocess.Popen([sys.executable,'-B','-m','experimentos.avaliacao_escala.executar',stage],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            monitored = psutil.Process(proc.pid)
            status = 'concluido'
            while proc.poll() is None:
                try:
                    over_memory = monitored.memory_info().rss > 5*1024**3
                except psutil.NoSuchProcess:
                    break
                if over_memory or time.monotonic()-start>900:
                    status = 'limite_memoria' if over_memory else 'limite_tempo'
                    proc.kill()
                    break
                time.sleep(.2)
            rc=proc.wait()
        costs[stage]={'segundos':round(time.monotonic()-start,3),'status':status if rc==0 or status!='concluido' else 'erro','codigo':rc}
        peak=OUT/f'pico_{stage}.json'
        if peak.exists():
            costs[stage].update(json.loads(peak.read_text()))
        save('custos.json',costs)
        print(stage,costs[stage],flush=True)
        if rc:raise SystemExit(rc)
    rankings={}
    summary={}
    for method in STAGES[-2:]:
        with (OUT/f'ranking_{method}.csv').open() as f:
            rows=list(csv.DictReader(f))
        ids=[int(row['wallet_id']) for row in rows]
        scores=[float(row['score_anomalia']) for row in rows]
        import math
        assert len(ids)==len(set(ids)) and all(map(math.isfinite,scores))
        assert list(zip(scores,ids))==sorted(zip(scores,ids),key=lambda pair:(-pair[0],pair[1]))
        rankings[method]=ids
        summary[method]={'carteiras':len(ids),'empates_no_score_da_posicao_100':scores.count(scores[99])}
        if method=='relacional':summary[method]['score_positivo']=sum(v>0 for v in scores)
    a,b=rankings.values()
    assert set(a)==set(b)
    summary['intersecao']={k:len(set(a[:k])&set(b[:k])) for k in (50,100)}
    assert all(digest(ROOT/name)==value for name,value in hashes.items())
    save('resumo.json',summary)


if __name__=='__main__':
    if len(sys.argv)==1:main()
    elif sys.argv[1] in STAGES:execute(sys.argv[1])
    else:raise SystemExit('Estágio desconhecido')
