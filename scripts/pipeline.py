"""Reconstrói uma execução isolada a partir dos dois CSVs publicados."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import mkdtemp

ROOT = Path(__file__).resolve().parents[1]
REFERENCIA = ROOT / 'resultados' / 'entradas.json'
BASE = {'main': '1a1d456', 'v2': 'c0b7390', 'v3': 'c0b7390'}
ENV = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def conferir(folder, hashes):
    for name, expected in hashes.items():
        if not (folder/name).is_file() or digest(folder/name) != expected:
            raise ValueError(f'Arquivo ausente ou diferente da edição documentada: {folder/name}')


def python(repo, *args):
    subprocess.run([sys.executable, '-B', *map(str,args)], cwd=repo, env=ENV, check=True)


def preparar_copia(repo, versao):
    subprocess.run(['git','clone','--quiet','--local','--no-checkout',str(ROOT),str(repo)],check=True)
    branch = 'feat/gnn-v2-anomalias' if versao in ('v2','v3') else 'main'
    subprocess.run(['git','switch','-C',branch,BASE[versao]], cwd=repo, check=True)
    if versao == 'v3':
        # Restaurar a base exata exigida pelo congelamento, apenas nesta cópia nova.
        subprocess.run(['git','restore','--source=deb24e5','--worktree','--',
                        'experimentos/gnn_v3','docs/gnn_v3','tests/test_gnn_v3.py'],cwd=repo,check=True)
        (repo/'docs/gnn_v3/congelamento.json').unlink()


def filtrar(repo, raw, manifest):
    destino = repo/'data/raw'
    destino.mkdir(parents=True)
    for name in manifest['brutos']:
        shutil.copyfile(raw/name,destino/name)
    conferir(destino, manifest['brutos'])
    edge='data/raw/betgraph_arestas_16set'
    event='data/raw/betgraph_nos_evento_16set'
    python(repo,'src/processar_graficos_arestas.py',edge+'.csv')
    python(repo,'src/processar_graficos_nos.py',event+'.csv')
    python(repo,'src/filtrar_recorrentes.py',event+'_processado.csv',edge+'_processado.csv')
    python(repo,'src/processar_graficos_wallets.py',edge+'_final.csv')
    conferir(destino,manifest['tratados'])


def executar_main(repo):
    shutil.copytree(ROOT/'experimentos/avaliacao_escala',repo/'experimentos/avaliacao_escala')
    shutil.copytree(ROOT/'scripts',repo/'scripts',dirs_exist_ok=True)
    shutil.copytree(ROOT/'resultados/reproducao',repo/'resultados/reproducao')
    shutil.copytree(ROOT/'resultados/escala',repo/'resultados/escala')
    python(repo,'scripts/reproducao.py','amostra')
    python(repo,'scripts/reproducao.py','verificar')
    python(repo,'-m','experimentos.avaliacao_escala.executar')
    conferir(repo/'data/processed/escala_v1', json.loads((ROOT/'resultados/escala/rankings_sha256.json').read_text()))
    obtido=json.loads((repo/'data/processed/escala_v1/resumo.json').read_text())
    esperado=json.loads((ROOT/'resultados/escala/resumo.json').read_text())
    if obtido != esperado:
        raise ValueError('O resumo dos rankings completos divergiu da referência.')


def executar_pilotos(repo):
    dev='data/processed/dev'
    piloto='data/processed/sintetico/piloto_v2_8101_8102'
    python(repo,'src/preparar_features.py','--sample-size','5000','--seed','42')
    python(repo,'src/construir_hetero.py')
    python(repo,'src/preparar_gnn.py')
    original=dev+'/experimento_seeds_42_73_2026'
    python(repo,'src/experimento_gnn.py','--seeds','42','73','2026','--output',original)
    python(repo,'src/avaliar_curvas.py','--input',original,'--output',original+'/curvas')
    python(repo,'src/resumir_roc.py','--input',original+'/curvas')
    python(repo,'-m','experimentos.piloto_sintetico.gerar','--protocolo',
           'experimentos/piloto_sintetico/protocolo_v2.json','--input',dev,'--split',
           dev+'/gnn_seed_42/divisao.pt','--output',piloto)
    for fase in ('executar','avaliar'):
        python(repo,'-m','experimentos.piloto_sintetico.'+fase,'--run',piloto)
    destino='data/processed/investigacao_anomalias_v1'
    python(repo,'-m','experimentos.investigacao_anomalias.ranking_relacional',
           '--input',piloto+'/original/resultados/divisao.pt','--output',destino+'/ranking_original')
    python(repo,'-m','experimentos.investigacao_anomalias.diagnosticar','--run',piloto,
           '--output',destino+'/desenvolvimento')
    python(repo,'-m','experimentos.investigacao_anomalias.validar')
    python(repo,'-m','experimentos.validacao_ampliada.executar')
    python(repo,'-m','experimentos.validacao_ampliada.casos')
    python(repo,'docs/validacao_ampliada/consolidar.py')
    python(repo,'docs/validacao_ampliada/auditar.py')


def publicar_saidas(repo, versao, parcial, historico):
    """Expõe apenas resultados gerados nesta execução, com links relativos."""
    from os.path import relpath
    dest = repo.parent / 'resultados'
    dest.mkdir()
    processed = repo / 'data/processed'
    if versao == 'main':
        escala = processed / 'escala_v1'
        execucao, = (repo / 'data/reproducao').glob('execucao-*')
        arquivos = {p.name:p for p in [*escala.glob('ranking_*.csv'), escala/'resumo.json',
                    escala/'custos.json', execucao/'verificacao.json']}
        if historico:
            arquivos.update(ligacoes=processed/'dev/experimento_seeds_42_73_2026',
                            piloto=processed/'sintetico/piloto_v2_8101_8102/avaliacao',
                            validacao=processed/'validacao_ampliada_v1',
                            investigacao=processed/'investigacao_anomalias_v1')
    else:
        rodada = processed / f'gnn_{versao}_v1'
        arquivos = {'rodada':rodada}
        if not parcial:
            import pandas as pd
            tabela = rodada/'metricas.csv'
            referencia = ROOT/'resultados'/versao/'metricas.csv'
            chaves = (['scenario','condition','method','model_seed','universe'] if versao=='v2'
                      else ['caso','fase','metodo','seed_modelo','universo'])
            def ler(path):
                return pd.read_csv(path).sort_values(chaves).reset_index(drop=True)
            pd.testing.assert_frame_equal(ler(tabela), ler(referencia), rtol=1e-6, atol=1e-9)
            arquivos.update(metricas=tabela, auditoria=repo/'docs'/f'gnn_{versao}'/'auditoria.json')
    for nome, caminho in arquivos.items():
        if not caminho.exists():
            raise FileNotFoundError(caminho)
        (dest/nome).symlink_to(relpath(caminho,dest))


def executar_amostra(repo, grupo, manifest, escopo, cenario, injecao, epocas, detector):
    import csv
    entrada = repo/'data/processed/validacao_ampliada_v1/novo'
    if escopo == '5000':
        python(repo, ROOT/'scripts/coorte.py', repo)
        conferir(entrada, manifest['coorte_csv'])
    else:
        python(repo,'src/preparar_features.py','--sample-size','0','--output',entrada)
        python(repo,'src/construir_hetero.py','--input',entrada)
        python(repo,'src/preparar_gnn.py','--input',entrada,'--output',entrada,'--seed','42')
    runner = ROOT/'scripts/modelos_amostra.py'
    python(repo, runner, 'preparar','--escopo',escopo,'--cenario',cenario,'--injecao',injecao)
    custos = []
    metricas = []
    for modelo in ('relacional', 'floresta', 'gnn', 'mlp'):
        neural = modelo in ('gnn', 'mlp')
        if (grupo == 'redes' and not neural) or (grupo == 'baselines' and neural):
            continue
        for seed in ((42, 73, 2026) if neural else (42,)):
            python(repo, runner, modelo, '--seed', seed, '--epocas', epocas, '--detector', detector)
            arquivo = repo.parent/'resultados'/f'{modelo}_{seed}'/'custos.json'
            custos.append(json.loads(arquivo.read_text()))
            if cenario != 'original':
                with arquivo.with_name('metricas.csv').open() as stream:
                    metricas.extend(csv.DictReader(stream))
    if metricas:
        with (repo.parent/'resultados/metricas.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=metricas[0].keys())
            writer.writeheader()
            writer.writerows(metricas)
    with (repo.parent/'resultados/custos.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in custos for k in row)))
        writer.writeheader()
        writer.writerows(custos)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('versao',choices=BASE)
    parser.add_argument('--raw',type=Path,default=ROOT/'data/raw')
    parser.add_argument('--destino',type=Path,default=ROOT/'data/execucoes')
    parser.add_argument('--historico',action='store_true')
    parser.add_argument('--preparar-apenas',action='store_true')
    parser.add_argument('--modelos','--amostra-5000', dest='grupo', choices=('baselines','redes','todos'))
    parser.add_argument('--escopo', choices=('5000','completo'), default='5000')
    parser.add_argument('--cenario', choices=('original','volume','estrutura'), default='original')
    parser.add_argument('--injecao', type=int, choices=(40101,40201), default=40101)
    parser.add_argument('--epocas', type=int, default=100)
    parser.add_argument('--detector', choices=('principal','if'), default='principal')
    args=parser.parse_args()
    if args.epocas < 1:
        parser.error('epocas deve ser positivo')
    if args.grupo and (args.historico or args.preparar_apenas):
        parser.error('--amostra-5000 é exclusivo da execução da main na amostra')
    manifest=json.loads(REFERENCIA.read_text())
    conferir(args.raw,manifest['brutos'])
    args.destino.mkdir(parents=True,exist_ok=True)
    pasta=Path(mkdtemp(prefix=args.versao+'-',dir=args.destino)).resolve()
    repo=pasta/'projeto'
    print('Execução:',pasta,flush=True)
    preparar_copia(repo,args.versao)
    filtrar(repo,args.raw,manifest)
    if args.grupo:
        executar_amostra(repo, args.grupo, manifest, args.escopo, args.cenario, args.injecao, args.epocas, args.detector)
    elif args.versao=='main':
        executar_main(repo)
        if args.historico:
            executar_pilotos(repo)
    else:
        python(repo,ROOT/'scripts/coorte.py',repo)
        conferir(repo/'data/processed/validacao_ampliada_v1/novo', manifest['coorte_csv'])
        if args.versao=='v3':
            modulo='experimentos.gnn_v3.executar'
            for fase in ['congelar','preparar']+([] if args.preparar_apenas else ['treinar','avaliar','auditar']):
                python(repo,'-m',modulo,fase)
        else:
            python(repo,'-c','from experimentos.gnn_v2.executar import prepare,DEFAULT; prepare(DEFAULT)')
            if not args.preparar_apenas:
                for script in ('executar_paralelo','consolidar','auditar'):
                    python(repo,f'docs/gnn_v2/{script}.py')
    if not args.grupo:
        publicar_saidas(repo,args.versao,args.preparar_apenas,args.historico)
    resumo={'versao':args.versao,'base':BASE[args.versao],'historico':args.historico,
            'preparar_apenas':args.preparar_apenas,'status':'concluido','entradas':manifest['brutos']}
    if args.grupo:
        resumo.update(grupo=args.grupo, escopo=args.escopo, cenario=args.cenario,
                      seed_injecao=args.injecao if args.cenario!='original' else None, epocas_max=args.epocas, detector=args.detector)
        resumo['runner_sha256'] = digest(ROOT/'scripts/modelos_amostra.py')
    resumo['checkout'] = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    resumo['python'] = sys.version
    resumo['pipeline_sha256'] = digest(Path(__file__))
    (pasta/'execucao.json').write_text(json.dumps(resumo,indent=2)+'\n')
    print('Resultados:',pasta/'resultados',flush=True)


if __name__=='__main__':
    main()
