"""Relatorio completo: compara todos os metodos, sem escolher melhor seed."""
from pathlib import Path
import sys,shutil,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,precision_recall_curve
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'data/processed/gnn_v2_v1';DOC=Path(__file__).resolve().parent
LABELS={'gnn_base_if':'GNN v1 + IF','mlp_base_if':'MLP base + IF','gnn_struct_if':'GNN estrutural + IF','mlp_struct_if':'MLP estrutural + IF','gnn_contrast_if':'GNN contrastiva + IF','mlp_contrast_if':'MLP contrastivo + IF','gnn_struct_hibrido':'GNN estrutural + híbrido','mlp_struct_hibrido':'MLP estrutural + híbrido','gnn_contrast_hibrido':'GNN contrastiva + híbrido','mlp_contrast_hibrido':'MLP contrastivo + híbrido','features_if':'Features + IF','struct_if':'Features estruturais + IF','relacional':'Relacional direto','features_hibrido':'Features + híbrido'}
ORDER=list(LABELS)


def main():
    assert (OUT/'avaliacao.json').exists(),'Esperar avaliacao antes de consolidar'
    for name in ['congelamento.json','cenarios.json','progresso.json','metricas.csv','avaliacao.json']:
        shutil.copyfile(OUT/name,DOC/name)
    shutil.copyfile(OUT/'agendamento/mudanca_execucao.json',DOC/'agendamento.json')
    d=pd.read_csv(OUT/'metricas.csv',float_precision='round_trip')
    final=d.query("universe=='completo' and condition=='injetado'")
    cols=['ap','auc','recovered_50','recovered_100','precision_100','recall_100']
    # Primeiro resumir seeds do encoder dentro de cada intervencao, depois comparar intervencoes.
    by=final.groupby(['family','level','injection_seed','method'])[cols].mean().reset_index()
    by.to_csv(DOC/'por_realizacao.csv',index=False)
    summary=by.groupby(['family','level','method'])[cols].agg(['mean','min','max'])
    summary.to_csv(DOC/'resumo.csv')
    before=d.query("universe=='completo' and condition=='original'").groupby(['family','level','injection_seed','method'])[cols].mean()
    delta=by.set_index(['family','level','injection_seed','method'])[cols]-before
    delta.to_csv(DOC/'mudanca_antes_depois.csv')
    perf=pd.DataFrame(json.loads((OUT/'progresso.json').read_text()))
    costs=perf.groupby('variant').agg(treinos=('seconds','size'),segundos_medios=('seconds','mean'),segundos_total=('seconds','sum'),parametros=('parameters','first'),epocas_medias=('epochs_run','mean'))
    costs.to_csv(DOC/'custos.csv')
    conditions=[('coparticipacao',.3),('coparticipacao',.6),('volume',10.)]
    matrix=np.array([[summary.loc[(fam,level,method),('recovered_100','mean')] for fam,level in conditions] for method in ORDER])
    fig,ax=plt.subplots(figsize=(9,8),layout='constrained');im=ax.imshow(matrix,vmin=0,vmax=50,cmap='YlGnBu',aspect='auto')
    ax.set_yticks(range(len(ORDER)),[LABELS[m] for m in ORDER]);ax.set_xticks(range(3),['Coparticipação 30%','Coparticipação 60%','Volume 10×'])
    for i in range(len(ORDER)):
        for j in range(3):ax.text(j,i,f'{matrix[i,j]:.1f}',ha='center',va='center',color='white' if matrix[i,j]>28 else 'black')
    ax.set_title('Alvos sintéticos no top 100, de 50\nMédia das duas realizações; encoders resumidos nas três seeds')
    fig.colorbar(im,ax=ax,label='Alvos recuperados');fig.savefig(DOC/'comparacao.png',dpi=160);plt.close(fig)
    selected=['gnn_base_if','gnn_struct_if','gnn_contrast_if','mlp_struct_if','gnn_struct_hibrido','gnn_contrast_hibrido','relacional','features_if']
    fig,axs=plt.subplots(1,2,figsize=(13,5.5),layout='constrained');grid=np.linspace(0,1,201)
    for method in selected:
        rocs=[];prs=[]
        for seed in (20101,20201):
            folder=OUT/f'coparticipacao_0.6_{seed}';labels=pd.read_csv(folder/'rotulos.csv').set_index('wallet_id').alvo
            files=[folder/'rankings'/f'{method}.csv'] if method in ('relacional','features_if') else []
            if not files:
                variant,detector=method.rsplit('_',1);files=[folder/'rankings'/f'{variant}_seed_{s}_{detector}.csv' for s in (42, 73, 2026)]
            for file in files:
                r=pd.read_csv(file,float_precision='round_trip');y=labels.loc[r.wallet_id].to_numpy();fpr,tpr,_=roc_curve(y,r.score_anomalia);precision,recall,_=precision_recall_curve(y,r.score_anomalia)
                rocs.append(np.interp(grid,fpr,tpr));prs.append(np.interp(grid,recall[::-1],precision[::-1]))
        auc=summary.loc[('coparticipacao',.6,method),('auc','mean')];ap=summary.loc[('coparticipacao',.6,method),('ap','mean')]
        axs[0].plot(grid,np.mean(rocs,axis=0),label=f'{LABELS[method]} ({auc:.3f})')
        axs[1].plot(grid,np.mean(prs,axis=0),label=f'{LABELS[method]} ({ap:.3f})')
    axs[0].plot([0,1],[0,1],'k--',alpha=.4);axs[1].axhline(.01,color='k',linestyle='--',alpha=.4)
    for ax in axs:ax.set_xlim(0,1);ax.set_ylim(0,1);ax.grid(alpha=.2);ax.legend(fontsize=7)
    axs[0].set(xlabel='Falsos positivos (não injetados)',ylabel='Recall dos alvos',title='ROC sintética — AUC média na legenda')
    axs[1].set(xlabel='Recall dos alvos',ylabel='Precisão sintética',title='Precisão–recall — AP média na legenda')
    fig.suptitle('Coparticipação 60%: curvas descritivas; não avaliam fraude ou outros dias');fig.savefig(DOC/'curvas_sinteticas.png',dpi=160);plt.close(fig)
    lines=['| Método | Copart. 30% | Copart. 60% | Volume 10× | AP em copart. 60% |','| --- | --- | --- | --- | --- |']
    for i,method in enumerate(ORDER):lines.append('| '+LABELS[method]+' | '+' | '.join(f'{v:.2f}' for v in matrix[i])+f' | {summary.loc[("coparticipacao",.6,method),("ap","mean")]:.4f} |')
    primary=by.query("family=='coparticipacao' and level==0.6").pivot(index='injection_seed',columns='method',values='recovered_100')
    candidates=['gnn_struct_if','gnn_contrast_if','gnn_struct_hibrido','gnn_contrast_hibrido'];adopt=[]
    for candidate in candidates:
        equivalent=candidate.replace('gnn_','mlp_')
        ap_table=by.query("family=='coparticipacao' and level==0.6").pivot(index='injection_seed',columns='method',values='ap')
        controls=['gnn_base_if',equivalent,'relacional']
        if all((primary[candidate]>primary[c]).all() and (ap_table[candidate]>ap_table[c]).all() for c in controls):adopt.append(candidate)
    decision='Nenhuma variante GNN satisfez simultaneamente o critério de superioridade em top100 e AP nas duas realizações frente à GNN base, ao MLP equivalente e ao relacional direto.' if not adopt else 'Variantes que satisfizeram o critério primário nesta rodada: '+', '.join(LABELS[a] for a in adopt)+'. Isso ainda não autoriza generalização temporal ou fraude.'
    dependence=[]
    for injection_seed in (20101,20201):
        folder=OUT/f'coparticipacao_0.6_{injection_seed}'/'rankings'
        direct=pd.read_csv(folder/'relacional.csv',float_precision='round_trip').set_index('wallet_id')
        for model_seed in (42, 73, 2026):
            hybrid=pd.read_csv(folder/f'gnn_contrast_seed_{model_seed}_hibrido.csv',float_precision='round_trip').set_index('wallet_id')
            ids=direct.head(100).index
            dependence.append({'injection_seed':injection_seed,'model_seed':model_seed,'top100_em_comum':len(set(ids)&set(hybrid.head(100).index)),'razao_score_mediana_top100':float((hybrid.loc[ids].score_anomalia/direct.loc[ids].score_anomalia).median())})
    dependence=pd.DataFrame(dependence);dependence.to_csv(DOC/'dependencia_hibrido.csv',index=False)
    recommendation=('Manter a v2 como candidato experimental, sujeito a novos dias e revisão humana antes de promover.' if adopt else 'Recomendação: manter os módulos e resultados da v2 como ablação experimental, mas não substituir o detector relacional direto por uma GNN nesta configuração. O custo adicional não demonstrou ganho consistente. Para o pipeline principal, conservar os canais relacional e de atributos, sem tratá-los como detectores de fraude confirmada.')
    report=f'''# GNN v2 — comparação experimental

Branch `feat/gnn-v2-anomalias`, base `1a1d456`. A main não foi alterada. Avaliação de alvos artificiais em novas realizações, com protocolo e código congelados antes de consultar métricas. Não é detecção confirmada de fraude.

## O que foi implementado

Cinco features estruturais acrescidas às features de contexto; encoders GraphSAGE e MLP equivalentes em entradas; objetivo contrastivo opcional, além da previsão de ligações; e detector híbrido de pares. Os módulos estão explicados em [README](README.md). A loss contrastiva é uma adaptação inspirada em [Deep Graph Infomax](https://arxiv.org/abs/1809.10341), não uma reprodução exata do artigo.

## Resultados

Número médio de alvos entre os 100 primeiros, de 50 injetados. Primeiro resumimos as seeds 42, 73, 2026 dentro de cada realização e depois as duas realizações 20101/20201. Baselines sem encoder têm uma avaliação por realização. Os valores fracionários são médias, não quantidades de carteiras em um ranking específico.

{chr(10).join(lines)}

![Comparação](comparacao.png)

Precisão sintética@100 = recuperados/100; recall@100 = recuperados/50. Por exemplo, 30 alvos significam 30% de precisão sintética e 60% de recall. Os não injetados não são normais confirmados. As métricas completas por seed, original/injetado, universo elegível e sem doadoras estão em `metricas.csv`; mudanças antes/depois em `mudanca_antes_depois.csv`.

![Curvas](curvas_sinteticas.png)

As curvas são descritivas, não intervalos de confiança. As seis curvas dos encoders compartilham duas intervenções e um recorte. AP resume o ranking com prevalência de 1%; AUC alta, isoladamente, não garante recuperação no topo.

## Decisão experimental

{decision}

{recommendation}

O híbrido contrastivo compartilhou {dependence.top100_em_comum.min()}–{dependence.top100_em_comum.max()} carteiras do top100 com o detector direto. A mediana da razão entre seus scores nesses casos variou de {dependence.razao_score_mediana_top100.min():.4f} a {dependence.razao_score_mediana_top100.max():.4f}. Isso mostra pequena modulação do score já existente; não uma vantagem demonstrada da representação aprendida.

O critério exige consistência entre realizações, não escolher a melhor seed. Ver `por_realizacao.csv`. Esta conclusão não equivale a descartar toda GNN. As features estruturais incluem o score relacional conhecido; melhora pode vir da informação manual adicionada. O híbrido também multiplica esse score pela similaridade, portanto sua eficácia não pode ser atribuída automaticamente à rede. Comparar especialmente com MLP e features+híbrido.

## Execução, controles e limites

Foram concluídos {len(perf)} treinos, com as seis variantes e três seeds nos sete grafos. O total somado dos tempos de treino foi {perf.seconds.sum()/60:.2f} minutos; isso não é duração de parede, pois os cenários foram distribuídos em três processos com duas threads cada. O agendamento preservou os trabalhos completos e arquivou um treino interrompido, sem alterar hiperparâmetros ou abrir métricas. Registro em `agendamento.json`. Parâmetros e custos por variante em `custos.csv`.

- Entrada: segunda amostra de 5.000 carteiras do mesmo dia, já utilizada na validação ampliada. As intervenções são novas; o dia e as carteiras de base não são novos dados independentes.
- Checkpoints selecionados somente por loss de validação de ligações; nenhum rótulo de injeção entrou nas features,loss ou escolha de época. Essa escolha pode favorecer a tarefa de ligações e limitar o objetivo contrastivo; foi fixada previamente.
- O contraste embaralha features, não simula necessariamente fraude ou o padrão coletivo de interesse. Alterações de arestas/tempo e outros objetivos exigem novos experimentos.
- MLP e GNN têm capacidades diferentes apesar de hidden/embedding iguais; contagens de parâmetros são explícitas. Contraste acrescenta discriminadores e custo.
- O score de pares usa cauda Poisson aproximada, não p-valor calibrado. Coparticipação legítima, modalidade e mercados relacionados continuam sendo explicações alternativas.
- Nem o conjunto completo, nem outros dias, nem o teste final de ligações foram avaliados nesta rodada. Dois cenários de atributo/estrutura não esgotam o conceito de anomalia.
- Os 32 testes passaram; auditoria dos checkpoints e métricas em `auditoria.json`. Dados e modelos permanecem fora do Git; não foi feito merge para main.

## Próximos passos até o fechamento do projeto

1. Revisar esta comparação e decidir se a v2 fica como candidato principal ou apenas como ablação documentada. Não apagar resultados negativos nem selecionar uma seed favorável.
2. Preencher independentemente os dois formulários dos casos reais já preparados; registrar explicações alternativas e desacordos. A avaliação humana está pendente.
3. Agrupar mercados relacionados e testar controles por modalidade/evento em novo protocolo; obter outros dias com a mesma extração e auditar disponibilidade temporal dos metadados.
4. Fazer benchmark de RAM/tempo no conjunto completo do dia e repetir os modelos selecionados com novos cenários reservados, sem ajustar aos resultados aqui usados.
5. Congelar escolhas finais e avaliar o teste reservado de ligações separadamente. Não confundir essa métrica com detecção de fraude.
6. Consolidar comandos/Makefile e reprodução no computador do colega, proveniência dos dados e arquivos necessários fora do Git.
7. Escrever metodologia, comparações, custos, resultados negativos e limitações; revisar com orientador e adequar ao congresso. Nenhum resultado garante aprovação.

## Reprodução

Ver [README](README.md), [protocolo](protocolo.md). Consolidação: `.venv/bin/python -B docs/gnn_v2/consolidar.py`. Auditoria: `.venv/bin/python -B docs/gnn_v2/auditar.py`. Leitura de rankings com `float_precision="round_trip"` evita empates artificiais. Artefatos completos em `data/processed/gnn_v2_v1/`.
'''
    (DOC/'resultado.md').write_text(report)
    print('RESUMO',flush=True);print(by.query("family=='coparticipacao' and level==0.6")[["injection_seed","method","ap","recovered_100"]].to_string(index=False));print(decision)

if __name__=='__main__':main()
