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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('versao',choices=BASE)
    parser.add_argument('--raw',type=Path,default=ROOT/'data/raw')
    parser.add_argument('--destino',type=Path,default=ROOT/'data/execucoes')
    parser.add_argument('--historico',action='store_true')
    parser.add_argument('--preparar-apenas',action='store_true')
    args=parser.parse_args()
    manifest=json.loads(REFERENCIA.read_text())
    conferir(args.raw,manifest['brutos'])
    args.destino.mkdir(parents=True,exist_ok=True)
    pasta=Path(mkdtemp(prefix=args.versao+'-',dir=args.destino)).resolve()
    repo=pasta/'projeto'
    print('Execução:',pasta,flush=True)
    preparar_copia(repo,args.versao)
    filtrar(repo,args.raw,manifest)
    if args.versao=='main':
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
    publicar_saidas(repo,args.versao,args.preparar_apenas,args.historico)
    resumo={'versao':args.versao,'base':BASE[args.versao],'historico':args.historico,
            'preparar_apenas':args.preparar_apenas,'status':'concluido','entradas':manifest['brutos']}
    resumo['checkout'] = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    resumo['python'] = sys.version
    resumo['pipeline_sha256'] = digest(Path(__file__))
    (pasta/'execucao.json').write_text(json.dumps(resumo,indent=2)+'\n')
    print('Resultados:',pasta/'resultados',flush=True)


if __name__=='__main__':
    main()
