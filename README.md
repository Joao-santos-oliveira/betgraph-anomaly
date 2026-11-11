![Betgraph Anomaly: carteiras e mercados de previsão](assets/betgraph.svg)

# Betgraph Anomaly

## Estudo principal

O Betgraph Anomaly avalia uma **rede neural em grafos heterogêneos (GNN híbrida)** para priorizar carteiras com coparticipação atípica na Polymarket. O recorte de 16/09/2026 contém 27.039 carteiras e 16.693 mercados após a filtragem. No grafo bipartido, cada ligação representa uma carteira que operou em um mercado. A pergunta de pesquisa é quais padrões de atuação em grupo a GNN consegue trazer ao topo da fila de investigação e como a recuperação varia com a intensidade.

### Um dia de atividade real

Um par é uma ligação entre uma carteira e um mercado. Se uma carteira operou cinco vezes no mesmo mercado, isso conta como um único par, mesmo que tenha operado em mais de um resultado. No dia filtrado há **229.072 pares distintos**. A divisão aleatória com seed 42 atribui a cada par uma função.

| Função | Pares | Uso |
|---|---:|---|
| Contexto | 137.444 | Ligações visíveis pelas quais a GNN agrega informações dos vizinhos. |
| Alvos positivos de treino | 45.814 | Ligações reais que a GNN aprende a prever a partir do contexto. |
| Validação | 22.907 | Ligações reservadas para acompanhar o aprendizado e selecionar o checkpoint. |
| Teste | 22.907 | Ligações reservadas para avaliação final. |

Essa divisão é por ligação. O ranking continua cobrindo as 27.039 carteiras do recorte. A carteira 18753 ocupa a primeira posição da GNN nas seeds 42, 73 e 2026. Ela compartilhou 64 mercados com a carteira 18626, ante uma sobreposição esperada de aproximadamente 7,51 pelo modelo relacional. Mercados de uma mesma partida podem explicar parte de uma coincidência. Os dados permitem investigar essa relação, mas atribuir coordenação ou fraude exige evidência independente. [Top 100 da GNN na seed 42](resultados/artigo/top100_dia_real.csv) · [Posições das 27.039 carteiras por método e seed](resultados/artigo/posicoes_dia_real.csv)

### Controle positivo e ablação do score

Em cada realização, 50 carteiras existentes formam dez grupos de cinco. Cada grupo passa a compartilhar a mesma lista de 4, 10 ou 20 novos mercados. As trocas preservam o grau de cada carteira e mercado e os pares reservados. Cada alvo tem um controle intacto de mesmo grau. O desenho alterna as doses entre três sorteios e totaliza 50 participações-alvo por intensidade. [Protocolo](experimentos/artigo/protocolo.json)

Os três scores consideram os mesmos pares candidatos com pelo menos dois mercados em comum e agregam as três maiores contribuições por carteira, dividindo a soma por três. O relacional mede a surpresa da sobreposição. A ablação usa apenas a similaridade de cosseno dos embeddings da GNN, reescalada para o intervalo de 0 a 1. A GNN híbrida multiplica os dois termos para cada par. É uma ablação do **score final**. Os embeddings continuam recebendo os atributos estruturais usados no treinamento original.

| Novos mercados comuns por grupo | GNN híbrida | Apenas similaridade da GNN | Relacional direto |
|---|---:|---:|---:|
| 4 | 2 / 7 | 0 / 0 | 2 / 7 |
| 10 | 0 / 24,33 | 0 / 0 | 0 / 26 |
| 20 | 50 / 50 | 0 / 0 | 50 / 50 |

Cada célula informa **alvos no top 100 / alvos no top 541** após a intervenção, acumulados nos três sorteios. As contagens das redes são médias entre as três seeds sobre os mesmos alvos. O top 100 é o corte principal. O top 541 cobre cerca de 2% do dia. Esse corte complementar foi introduzido após a primeira análise do dia completo e constava do protocolo do teste de grupos antes de calcular seus rankings. O sinal de surpresa sozinho recuperou todos os casos fortes. A ponderação aprendida preservou essa recuperação e não acrescentou ganho médio no nível intermediário. A interpretação se limita às intervenções e ao dia avaliados. [Resumo antes e depois, com controles](resultados/artigo/resumo.csv) · [Resultados por realização e seed](resultados/artigo/por_realizacao_seed.csv)


### Reprodução do estudo principal

Em Linux ou WSL, use Python 3.12 e GNU Make. Coloque os dois arquivos brutos `betgraph_arestas_16set.csv` e `betgraph_nos_evento_16set.csv` em `data/raw/`. O executor confere os hashes dos dados, filtra os CSVs, monta o grafo do dia, recria as três intervenções, treina a GNN e calcula os três rankings. As dependências e os dados volumosos permanecem locais. [Hashes das entradas](resultados/entradas.json)

```bash
make ambiente
make validar
make testes
make artigo
```

Use `BRUTOS=/caminho/para/csvs` e `DESTINO=/caminho/para/execucoes` para escolher as pastas. O padrão é um processo de treino por vez. `make artigo PROCESSOS=3` permite treinar as três seeds em paralelo quando há memória disponível. Cada chamada cria uma pasta exclusiva e imprime a localização dos CSVs de resultados. O código do artigo fica em [experimentos/artigo](experimentos/artigo) e o modelo em [experimentos/gnn](experimentos/gnn).

Para reavaliar embeddings salvos de uma execução, use:

```bash
.venv/bin/python -m experimentos.artigo.ablacao PASTA_DA_EXECUCAO PASTA_DE_SAIDA
```

## O que há nas outras branches

| Branch | Conteúdo |
|---|---|
| [feat/gnn-v2-anomalias](https://github.com/Joao-santos-oliveira/betgraph-anomaly/tree/feat/gnn-v2-anomalias) | Desenvolvimento histórico do modelo principal, com treino contrastivo, score híbrido e Isolation Forest sobre embeddings. |
| [feat/gnn-v3-anomalias](https://github.com/Joao-santos-oliveira/betgraph-anomaly/tree/feat/gnn-v3-anomalias) | Experimento com atributos de aresta, incluindo lado escolhido, horário e valor, além dos controles sintéticos. |
| [feat/evidencias-artigo](https://github.com/Joao-santos-oliveira/betgraph-anomaly/tree/feat/evidencias-artigo) | Benchmark anterior na coorte de 5.000 carteiras, com tabelas comparativas, figuras, subgrafos e comprovantes de reprodução. |

A comparação anterior de escalas e o resumo da revisão descritiva podem ser consultadas no [histórico da main](https://github.com/Joao-santos-oliveira/betgraph-anomaly/tree/e976ca3/resultados).
