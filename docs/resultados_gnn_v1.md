# Experimento controlado GNN — rodada 1

Execução local em 17/10/2026. Artefatos: `data/processed/dev/experimento_controlado_v1/`. Resultados exploratórios de validação, sem avaliação final de teste. Esta rodada não demonstra detecção de fraude.

## O que foi executado

12 treinos em CPU: GNN e MLP sem mensagens × features do dia inteiro e de contexto × seeds 42, 43 e 44. Foram preservadas as 5.000 carteiras, 9.189 mercados e a divisão existente de 39.562 pares únicos. Os 7.912 exemplos de validação (3.956 positivos e 3.956 negativos) são exatamente iguais entre execuções. Negativos reservados de validação/teste não entram no treino.

Inicialização e dropout variam por seed. Divisão, negativos de treino, exemplos de avaliação e IsolationForest permanecem fixos. Cada treino usa até 100 épocas, Adam 0,001, hidden 64, embedding 32, dropout 0,1, clipping 5 e duas threads. A seleção é por BCE de validação, com min_delta 0,0001 e paciência 15. O protocolo e os hashes de código/dados foram registrados antes dos treinos em `protocolo.json`.

Em `context_only`, os agregados de carteiras usam somente arestas dos pares de contexto, incluindo todos os outcomes desses pares. Os transformadores são ajustados somente nos nós com contexto. Carteiras isoladas recebem volume/contagens/ticket zero, preço mediano dos ativos e flag `sem_contexto`. Metadados de mercados continuam transdutivos. A condição `full_day` preserva as features anteriores, com informação agregada de conexões ocultadas.

## Aprendizado e controles

Média ± desvio-padrão amostral entre três seeds. AP e AUC são métricas de ligação contra negativos uniformes, não precisão de identificação de anomalias.

| Features | Encoder | BCE ↓ | AP ↑ | ROC-AUC ↑ | Parâmetros |
| --- | --- | --- | --- | --- | --- |
| Dia inteiro | GNN | 0,29548 ± 0,00171 | 0,94501 ± 0,00055 | 0,94662 ± 0,00060 | 20.673 |
| Dia inteiro | MLP | 0,41448 ± 0,00142 | 0,90172 ± 0,00068 | 0,88754 ± 0,00085 | 12.545 |
| Contexto | GNN | 0,31882 ± 0,00296 | 0,93907 ± 0,00094 | 0,93780 ± 0,00105 | 20.289 |
| Contexto | MLP | 0,43254 ± 0,00190 | 0,89335 ± 0,00087 | 0,87621 ± 0,00163 | 12.353 |

Controle simples por produto dos graus no contexto: AP **0,85222**, ROC-AUC **0,78763**. É uma única avaliação determinística, comum às condições. Não calculamos BCE para esse score não calibrado.

A perda caiu em todos os treinos. Todas as execuções atingiram 100 épocas, e o checkpoint selecionado foi o da época 100. As curvas ainda apresentavam melhora; o limite de orçamento foi atingido, não um critério de convergência. Históricos completos estão nas pastas de cada execução.

A GNN superou os controles implementados nesta tarefa de validação, inclusive com agregados restritos ao contexto. Isso sustenta continuar a investigação, mas não prova que a identidade dos vizinhos explica todo o ganho: os encoders têm capacidades diferentes, e a ablação estrutural preservando graus permanece pendente. Negativos uniformes também tornam parte da tarefa fácil. Não houve busca Optuna nem escolha de hiperparâmetros após inspecionar esses resultados.

Os manifestos incluem resultados separados por pares com ambos os endpoints no contexto e com algum endpoint isolado. As prevalências desses grupos diferem; suas APs não devem ser comparadas diretamente entre grupos.

## Rankings e estabilidade

Foram gerados 14 rankings: 12 de embeddings (GNN/MLP) e dois de features individuais. Cada ranking inclui todas as 5.000 carteiras e uma fila principal de 50 carteiras (top 1%). IsolationForest: 100 árvores, max_samples automático, contamination automático, seed 42 e uma thread. O score é `-score_samples`, sem interpretação probabilística. Empates usam wallet_id crescente.

Os embeddings são extraídos do checkpoint selecionado, com dropout desligado, usando o mesmo contexto fixo. Nenhuma conexão reservada foi adicionada para gerar embeddings. Arquivos NPZ contêm embeddings de carteiras/mercados e IDs originais; JSON contém ID original → índice.

| Condição / encoder | Spearman entre seeds (intervalo) | Carteiras comuns entre tops de 50 | Jaccard dos tops (intervalo) |
| --- | --- | --- | --- |
| Dia inteiro / GNN | 0,8221–0,8930 | 41–46 | 0,6949–0,8519 |
| Dia inteiro / MLP | 0,7830–0,8703 | 43–48 | 0,7544–0,9231 |
| Contexto / GNN | 0,9385–0,9458 | 46–48 | 0,8519–0,9231 |
| Contexto / MLP | 0,7525–0,7613 | 42–45 | 0,7241–0,8182 |

Cada intervalo resume as três comparações entre pares de seeds; não é intervalo de confiança. Spearman usa postos médios para empates. O arquivo `comparacoes_rankings.csv` também inclui cortes de 0,5% e 2%, além das comparações entre métodos na mesma condição.

No contexto, cada top 50 da GNN compartilha **23 carteiras** com o top 50 do IsolationForest nas features de contexto. Com features do dia inteiro, a interseção é de 24–27. Diferença de seleção não demonstra qualidade superior: precisamos investigar os casos e controles.

As 975 carteiras sem contexto foram preservadas, sinalizadas e não apareceram no top 50 dos 12 rankings de embeddings. Na condição de contexto, todas recebem a mesma entrada e, sem vizinhos, o mesmo embedding; isso causa empates e impede distingui-las com a informação disponível. Não significa que sejam normais. Não houve colapso global dos embeddings: todas as dimensões de carteira apresentaram variância positiva nesta rodada.

## Tempo, verificação e reprodução

Os loops de treinamento somaram **99,76 segundos**. Média por treino GNN: 10,29 s no dia inteiro e 10,44 s no contexto; MLP: 5,92 s e 6,60 s. Esses tempos incluem escrita durante o loop, mas excluem preparação, inferência posterior e IsolationForest. Não estimam o custo do conjunto completo. Tempos do detector estão nos manifestos de ranking.

Oito testes automatizados passaram. Cobrem preparação, negativos válidos, partições disjuntas, checkpoint reproduzível, igualdade de avaliação ao variar modelo/seed, ausência de influência das arestas ocultadas nas features de contexto, alinhamento de IDs e exportação. O código rejeita checkpoint de outra divisão e destinos já existentes. O relatório `integridade.json` confirma 12 execuções, 14 rankings e teste não avaliado.

Para repetir, na raiz com a .venv ativada:

```bash
python -m unittest discover -s tests -v
python src/experimento_gnn.py --output data/processed/dev/experimento_controlado_v2
```

Escolha uma pasta nova. Para ler os resultados, comece por `resultados.csv`, `comparacoes_rankings.csv` e por `context_only/gnn_seed_42/ranking/top.csv`. A seed 42 é apenas uma referência de navegação, não uma vencedora selecionada. Consulte as outras seeds antes de caracterizar casos.

`data/` permanece ignorado pelo Git. O colega precisa receber esses artefatos separadamente ou reproduzir a execução a partir dos mesmos dados; compartilhar só o código não compartilha checkpoints nem rankings. Este relatório mantém o resumo numérico no repositório.

## Próximos passos

Planejar a revisão dos casos e controles antes de abri-los; depois gerar visualizações dos subgrafos. Ainda faltam a ablação estrutural, sensibilidade à seed do detector, revisão humana, dados adicionais/recorte completo e avaliação final de teste após congelar decisões. Três seeds em uma amostra de um dia são uma primeira evidência descritiva, não validação geral nem garantia de aprovação acadêmica.
