"""Agendamento tecnico: mesmos treinos congelados em processos separados."""
from pathlib import Path
import sys,shutil,contextlib,multiprocessing
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experimentos.gnn_v2.executar import DEFAULT,VARIANTS,fit,evaluate,verify
from experimentos.piloto_sintetico.comum import read_json,write_json,sha


def worker(shard):
    with (shard/'execucao.log').open('a',buffering=1) as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):fit(shard)
    return shard.name


def main():
    out=DEFAULT;verify(read_json(out/'congelamento.json'))
    root=out/'agendamento';root.mkdir(exist_ok=True)
    # Trabalhos interrompidos ficam preservados, nunca apagados.
    moved=[]
    for scene in ['original']+[s['name'] for s in read_json(out/'cenarios.json') if s['status']=='viavel']:
        for variant in VARIANTS:
            for seed in (42,73,2026):
                job=out/scene/f'{variant}_seed_{seed}'
                if job.exists() and not (job/'complete.json').exists():
                    target=root/'interrompidos'/scene/job.name;target.parent.mkdir(parents=True,exist_ok=True)
                    if target.exists():raise FileExistsError(target)
                    shutil.move(str(job),str(target));moved.append(str(target.relative_to(out)))
    write_json(root/'mudanca_execucao.json',{'reason':'paralelizar cenarios independentes; sem mudanca de modelo/parametro/seed','workers':3,'threads_per_training':2,'partial_jobs_archived':moved,'scheduler_sha256':sha(Path(__file__)),'evaluation_not_opened':not (out/'metricas.csv').exists()})
    def shard(name,scenarios):
        dest=root/name;dest.mkdir(exist_ok=True)
        for scene in ['original']+[s['name'] for s in scenarios]:
            link=dest/scene
            if not link.exists():link.symlink_to(out/scene,target_is_directory=True)
        shutil.copyfile(out/'congelamento.json',dest/'congelamento.json')
        write_json(dest/'cenarios.json',scenarios)
        return dest
    original=shard('somente_original',[]);print('COMPLETANDO ORIGINAL',flush=True);worker(original)
    scenarios=[s for s in read_json(out/'cenarios.json') if s['status']=='viavel']
    jobs=[shard(s['name'],[s]) for s in scenarios]
    with ProcessPoolExecutor(max_workers=3,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(worker,p) for p in jobs]
        for f in as_completed(futures):print('CENARIO COMPLETO',f.result(),flush=True)
    records=[]
    for scene in ['original']+[s['name'] for s in scenarios]:
        for variant in VARIANTS:
            for seed in (42,73,2026):
                job=out/scene/f'{variant}_seed_{seed}';assert (job/'complete.json').exists()
                records.append({'scenario':scene,'variant':variant,'seed':seed,**read_json(job/'manifest.json')})
    write_json(out/'progresso.json',records)
    write_json(out/'treinos_concluidos.json',{'n':len(records),'test_evaluated':False,'workers':3})
    verify(read_json(out/'congelamento.json'));evaluate(out)

if __name__=='__main__':main()
