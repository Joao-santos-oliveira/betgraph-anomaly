"""Consolida resultados medidos, sem selecionar cenarios favoraveis."""
from pathlib import Path
import shutil,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'data/processed/validacao_ampliada_v1';DOC=Path(__file__).resolve().parent
for f in ['metricas.csv','status.json','nulos.csv','congelamento.json','integridade.json']:shutil.copyfile(OUT/f,DOC/f)
for f in ['pareamento.csv','inspecao_descritiva.csv']:shutil.copyfile(OUT/'casos_reais'/f,DOC/f)
d=pd.read_csv(DOC/'metricas.csv');selected=d.query("scenario=='injetado' and universe=='completo'")
summary=selected.groupby(['recorte','familia','intensidade','model']).agg(n=('ap','count'),ap=('ap','mean'),auc=('auc','mean'),top_min=('recovered_100','min'),top_max=('recovered_100','max'),top_media=('recovered_100','mean')).reset_index()
summary.to_csv(DOC/'resumo.csv',index=False)
fig,axs=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
for ax,family in zip(axs,['coparticipacao','volume']):
    levels=[.3,.6,.9] if family=='coparticipacao' else [10,50,100]
    for cohort,style in [('anterior','-'),('novo','--')]:
        for model,color in [('relacional','#117a65'),('features_if','#a84428')]:
            sub=summary.query('familia==@family and recorte==@cohort and model==@model').set_index('intensidade').reindex(levels)
            ax.errorbar(range(3),sub.top_media,yerr=np.array([sub.top_media-sub.top_min,sub.top_max-sub.top_media]),marker='o',linestyle=style,color=color,label=f'{model} / {cohort}',capsize=3)
    ax.set_xticks(range(3),['30%','60%','90%'] if family=='coparticipacao' else ['10×','50×','100×'])
    ax.set_title('Coparticipação' if family=='coparticipacao' else 'Volume artificial');ax.set_ylim(-2,53);ax.grid(alpha=.2);ax.set_xlabel('Intensidade da intervenção')
    ax.set_ylabel('Alvos no top 100, de 50')
axs[0].annotate('90%: só 1 de 4\nrealizações concluída',xy=(2,50),xytext=(.7,44),arrowprops={'arrowstyle':'->'},fontsize=9)
axs[1].legend(fontsize=8,loc='upper left')
fig.suptitle('Robustez: médias e intervalos min–máx; 2 realizações por condição quando viáveis')
fig.savefig(DOC/'robustez.png',dpi=170);fig.savefig(DOC/'robustez.pdf')
lines=['| Recorte | Padrão | Intensidade | Método | n concluídos | AP média | AUC média | Alvos top100 (min–máx) |','| --- | --- | --- | --- | --- | --- | --- | --- |']
for r in summary.itertuples():lines.append(f'| {r.recorte} | {r.familia} | {r.intensidade:g} | {r.model} | {r.n} | {r.ap:.4f} | {r.auc:.4f} | {r.top_min:g}–{r.top_max:g} |')
report='''# Validação ampliada — resultados de 26/10/2026

Executamos a parte computacional e uma inspeção descritiva dos casos reais. **Não concluímos validação de fraude nem revisão humana independente.** Apenas um dia de dados está disponível. O novo recorte usa outras 5.000 carteiras do mesmo dia, sem sobreposição de carteiras com a amostra anterior; mercados podem ser compartilhados.

## Resultado principal

- **Coparticipação a 60%:** o detector relacional recuperou 24–29 alvos no top100 na amostra anterior e 31–32 na nova, de50alvos. Features+IF recuperou0–1 e0, respectivamente. O ganho se repetiu em novas seleções e carteiras.
- **Coparticipação a 30%:** o detector relacional recuperou apenas1–2alvos por realização em ambos os recortes. A sensibilidade depende da força do padrão; não é um detector geral de qualquer coordenação.
- **Coparticipação a 90%:**3das4tentativas não completaram a intervenção após1milhão de propostas; uma concluiu e recuperou50alvos no top100. Não usar essa execução isolada como resultado representativo nem chamar as demais de recuperação zero. O gerador não demonstrou impossibilidade matemática: apenas não concluiu sob as regras e orçamento registrados.
- **Volume:** IF em features recuperou11–17alvos no recorte anterior e9–11no novo com fator10; com fator100,27–30e24–26. O score relacional permaneceu exatamente igual ao anterior à injeção porque as conexões não mudaram; seus3–6alvos no top100 já estavam lá. Isso não é detecção da mudança de volume.

Foram planejados24cenários:21concluídos e3não concluídos, sem troca de seeds. São dois tipos de intervenção (estrutura e atributo), três intensidades e dois recortes. As duas seeds de injeção são9201e9301; IF permanece42. Esta rodada **não retreinou GNN/MLP**; avalia o detector relacional adicionado contra features+IF. Os resultados anteriores desfavoráveis à GNN permanecem válidos para aquele protocolo.

![Robustez](robustez.png)

## Métricas completas

TABELA

A AP e AUC referem-se à recuperação de alvos artificiais, não fraude. Todas as avaliações original/injetado, incluindo universos elegível e sem doadoras, estão em `metricas.csv`. Os alvos elegíveis de coparticipação podem mudar entre intensidades; as curvas não são um efeito causal nos mesmos indivíduos. Para volume, os mesmos alvos são mantidos entre fatores por seed/recorte. Não há intervalo de confiança populacional: barras são mínimo/máximo de duas realizações, ou nenhuma amplitude com apenas uma.

## Controles com conexões aleatorizadas

Foram feitas quatro aleatorizações, duas por recorte, preservando graus e evitando positivos reservados. O percentil99do score caiu de8,65para3,43–3,52na amostra anterior e de8,82para3,68–3,76na nova. Só6–9e4–5carteiras, respectivamente, permaneceram nos mesmos top100. Esses controles mostram dependência da organização das conexões além dos graus, mas não provam anomalia: a aleatorização também destrói especialização legítima em modalidades e mercados relacionados. Não presumimos amostragem uniforme dos grafos possíveis nem calculamos p-valores/FPR a partir de duas aleatorizações.

## Casos reais

A seleção congelada gerou25casos da união top10dos quatro métodos,15controles semelhantes em grau/volume e10aleatórios. Dez casos não tiveram controle dentro do limite definido. As50fichas trazem dados e subgrafos limitados, e há dois formulários vazios para revisão independente. Ver [inspeção preliminar](inspecao_preliminar.md).

Exemplo:13537e seu controle13826têm71mercados cada e volume próximo (US$1689,83vs1851,34), mas39vs7mercados em comum com seus respectivos parceiros de maior surpresa. Alguns casos destacam e-sports ou tênis, incluindo mercados relacionados à mesma partida. Isso oferece uma explicação legítima plausível e mostra a necessidade de controlar modalidade/evento antes de alegar coordenação. A diferença também é parcialmente esperada pela seleção por score; não é validação independente.

## Conclusão e próximos passos

A evidência ficou mais forte para **recuperação de coparticipação suficientemente concentrada** e para a complementaridade entre estrutura e atributos. Ainda não autoriza afirmar que o algoritmo identifica anomalias reais com precisão conhecida.

1. Dois revisores preencherem as fichas sem chave/método e registrarem explicações alternativas e evidência insuficiente. Esta pendência exige avaliação humana; não foi preenchida como se já tivesse ocorrido.
2. Novo protocolo para agrupamento de mercados relacionados e nulos estratificados por modalidade/evento; esses fatores não foram controlados neste teste. Não ajustar os scores usando estas realizações e continuar chamando-as de reservadas.
3. Obter outros dias com o mesmo esquema. Não há CSV de outro dia nesta pasta, portanto validação temporal não foi realizada. Não confundir carteira nova com dia novo.
4. Depois, benchmark e repetição no conjunto completo do dia. O recorte novo não é o conjunto completo. O teste final de ligações permanece fechado e medirá ligações, não fraude.

## Reprodução

Na raiz do projeto, em uma cópia sem a pasta de saída desta rodada:

```bash
.venv/bin/python -B -m unittest discover -s tests -v
.venv/bin/python -B -m experimentos.validacao_ampliada.executar
.venv/bin/python -B -m experimentos.validacao_ampliada.casos
.venv/bin/python -B docs/validacao_ampliada/consolidar.py
```

Não sobrescreve experimentos anteriores. Parâmetros em `protocolo.md`; hashes anteriores à execução em `congelamento.json`; estados inclusive falhas em `status.json`. Rankings, rótulos, CSVs modificados, divisões, modelos IF e transformadores ficam em `data/processed/validacao_ampliada_v1/`, ignorado pelo Git. Nas falhas, salvamos diagnóstico e trocas parciais, sem avaliá-las como intervenção completa. Os28testes passaram. Nenhum commit foi feito.

Auditoria reproduzível: `.venv/bin/python -B docs/validacao_ampliada/auditar.py`. Ao ler rankings, usar `pd.read_csv(..., float_precision="round_trip")`: a leitura padrão pode arredondar scores muito próximos e criar empates artificiais. Os84rankings foram conferidos preservando precisão. Ver `auditoria.json`.
'''
# Melhora a legibilidade dos números sem alterar o protocolo congelado.
import re
report=re.sub(r'(?<=[A-Za-zÀ-ÿ])(\d)',r' \1',report)
report=re.sub(r'(\d)(?=[A-Za-zÀ-ÿ])',r'\1 ',report)
(DOC/'resultado.md').write_text(report.replace('TABELA','\n'.join(lines)))
print(summary.to_string(index=False))
