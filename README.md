# Betgraph Anomaly

**Validação ampliada — 26/10/2026:** [resultados e limites](docs/validacao_ampliada/resultado.md). Foram executadas 24 tentativas: 21 cenários completos e 3 injeções não concluídas, em duas amostras disjuntas de 5.000 carteiras do mesmo dia. O detector relacional recuperou 31–32/50 alvos no novo recorte com concentração de 60%, mas 1–2 com 30%; features+IF respondeu melhor ao aumento de volume. Preparadas 50 fichas reais (25 casos, 15 controles e 10 aleatórios); revisão humana independente e outros dias ainda pendentes. 28 testes passaram. Esta rodada não retreinou a GNN nem abriu o teste final de ligações.

Projeto universitário de detecção não supervisionada de carteiras anômalas em dados do Polymarket (Polygon), no recorte de 16/09/2026. Um score de anomalia não confirma fraude. A metodologia pode ser estudada em outros mercados, mas os valores não são numericamente comparáveis a casas de apostas tradicionais.


Roteiro atualizado: [próximos passos](docs/proximos_passos.md). Pilotos e validação ampliada concluídos; revisão humana independente pendente.

## Estrutura e dados

- `src/`: scripts Python, executados a partir da raiz do projeto.
- `data/raw/`: CSVs originais, intermediários e finais (não versionados).
- `data/processed/`: features, scalers, índices e grafos (não versionados).
- `mapeamento_*.json`: identificadores persistentes anteriores aos filtros.
- `queries/` e `notebooks/`: atualmente vazios. As consultas SQL ainda precisam ser recuperadas do Dune para completar a reprodução da extração.

CSVs finais confirmados por inspeção: `betgraph_wallets_16set_nos.csv` (27.039 carteiras), `betgraph_nos_evento_16set_final.csv` (16.693 mercados) e `betgraph_arestas_16set_final.csv` (358.689 arestas agregadas carteira–mercado–outcome).

## Ambiente CPU (WSL, Python 3.12)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/validar_ambiente.py
python -m pip check
```

`requirements.txt` fixa as versões do ambiente e inclui o índice de wheels CPU do PyTorch. A validação executa multiplicação de matrizes e forward/backward de SAGEConv. Treino com amostragem de vizinhos ainda não foi validado; poderá exigir dependências adicionais.

## Preparação existente

Execute a partir da raiz, nesta ordem, somente se precisar regenerar os dados:

```bash
python src/processar_graficos_arestas.py data/raw/betgraph_arestas_16set.csv
python src/processar_graficos_nos.py data/raw/betgraph_nos_evento_16set.csv
python src/filtrar_recorrentes.py data/raw/betgraph_nos_evento_16set_processado.csv data/raw/betgraph_arestas_16set_processado.csv
python src/processar_graficos_wallets.py data/raw/betgraph_arestas_16set_final.csv
```

Os scripts antigos usam caminhos relativos à raiz e atualizam os mapeamentos JSON. A remoção de mercados recorrentes deixa lacunas nos IDs; a construção do grafo deve criar índices consecutivos e preservar a correspondência com os IDs originais.

## Decisões de modelagem e limitações

- Grafo heterogêneo bipartido com nós `user` e `event`, relação `bets_on` e relação reversa. Cada linha agregada continua sendo uma aresta; um mesmo par carteira–mercado pode ter outcomes diferentes.
- Maker/taker não está disponível nem no CSV original de arestas. O primeiro modelo não utiliza essa feature e não usa volume ou frequência como substitutos. Melhoria futura: reextrair os dados com contagens por papel para calcular a proporção de trades como maker.
- Os 707 mercados sem duração válida são preservados. A preparação das features cria `duracao_ausente` e preenche a duração pela mediana das durações válidas do conjunto usado no ajuste.
- `duracao_total_segundos` é uma soma das durações de arestas, não tempo de atividade efetivo; fica fora do primeiro modelo.
- `primeira_atividade` significa **primeira atividade observada no recorte filtrado**, não idade da carteira.
- Status de resolução e fechamento podem refletir informações posteriores ao recorte; ficam fora das features iniciais. Perguntas, endereços e IDs são metadados, não medidas numéricas de comportamento.
- `n_outcomes_distintos` conta rótulos globais de outcome, não posições distintas por mercado; fica fora das features iniciais.
- Desenvolvimento em amostra reproduzível de aproximadamente 5.000 carteiras; execução completa apenas ao final.

## Plano científico

O roteiro de avaliação, controles, reprodução e submissão está em [docs/plano_pesquisa.md](docs/plano_pesquisa.md). Os resultados atuais são exploratórios; a próxima etapa é fixar o protocolo de comparação e testar o valor das mensagens no grafo.

## Status

Limpeza e agregação existentes; ambiente CPU, features e construção de HeteroData validados. Baseline IsolationForest e seu ranking implementados na amostra. Treino inicial da GNN implementado; comparação entre modelos, validação em três seeds e exportação para Gephi ainda estão pendentes.

## Features e construção do grafo

```bash
python src/preparar_features.py --sample-size 5000 --seed 42
python src/construir_hetero.py
python -m unittest discover -s tests -v
```

A amostra seleciona carteiras sem reposição após ordenação por ID, preserva todas as suas arestas e inclui seus mercados vizinhos. O modo completo (`--sample-size 0 --output data/processed/full`) preserva todos os nós de entrada, inclusive mercados isolados; execute-o apenas ao final do desenvolvimento.

Features iniciais:

| Entidade | Colunas e transformações |
| --- | --- |
| Carteira | `log1p` de `volume_total_usd`, `n_trades_total`, `n_mercados_distintos`, `ticket_medio_geral`; `preco_medio_ponderado`; StandardScaler nas cinco colunas |
| Mercado | duração imputada pela mediana, seguida de `log1p` e StandardScaler; flag binária `duracao_ausente`; categoria em one-hot |
| Aresta | `log1p(volume_usd)`, `preco_medio`, fração UTC do dia de `primeiro_trade`; StandardScaler nas três colunas |

A flag e as categorias permanecem binárias. `primeira_atividade` fica como metadado da carteira nesta versão. Categorias são as já extraídas pelo pipeline antigo a partir da primeira palavra de tags; sua qualidade semântica ainda deve ser revisada. Outcomes são preservados como metadados de arestas, sem tratar seus IDs como grandezas numéricas.

O ajuste atual é **transdutivo**, realizado no subgrafo selecionado, sem avaliação de generalização temporal. Para uma avaliação futura em dados separados, ajuste os transformadores apenas no treino e aplique-os aos outros conjuntos com `--preprocessador CAMINHO/preprocessador.joblib`. Esse arquivo inclui os três scalers, a mediana, o codificador de categorias e os nomes das features; categorias desconhecidas recebem vetor zero.

Artefatos em `data/processed/dev/`:

- `user.csv`, `event.csv`, `edge.csv`: linhas selecionadas, na mesma ordem das matrizes.
- `features.npz`: matrizes de features; `preprocessador.joblib`: transformadores ajustados.
- `features_manifest.json`: seed, contagens, mediana e nomes/ordem das features.
- `hetero.pt`: HeteroData com atributos, IDs originais e arestas reversas.
- `indices_originais.json`: ID original → índice consecutivo, separado por tipo de nó. Chaves JSON são strings.
- `grafo_manifest.json`: contagens e graus médios contando multiarestas.

As contagens do grafo são verificadas contra os CSVs selecionados. Os IDs são confrontados com os três JSONs persistentes da raiz. A construção rejeita arestas órfãs, IDs de nós duplicados e features não finitas, e verifica a relação reversa.

Execução de desenvolvimento realizada com seed 42: 5.000 carteiras, 9.189 mercados, 62.328 arestas diretas e 62.328 reversas. Foram imputadas 313 durações entre os mercados selecionados, com mediana de 2.082.742,5 segundos. Isso não descarta os 707 mercados com duração ausente da entrada: os demais não são vizinhos das carteiras amostradas. A execução completa ainda não foi realizada.

Validação: reutilização do preprocessador gerou matrizes idênticas; o grafo salvo foi recarregado e passou por forward/backward de HeteroConv em CPU. O teste de integração verifica mediana e flag, índices consecutivos, relação reversa e rejeição de arestas órfãs. Isso não é treino nem resultado de detecção de anomalias.

**Decisão da primeira versão:** SAGEConv não consome `edge_attr` diretamente. Os atributos estão preservados no grafo original, mas não são usados pelo encoder ou decoder atual. Para carregar um `HeteroData` próprio, use `torch.load(caminho, weights_only=False)`; carregue somente artefatos de confiança.

## Baseline: primeiro ranking sem GNN

O baseline responde: quais carteiras têm combinações incomuns das cinco características individuais? Ele não recebe `edge_index`, identidade dos mercados vizinhos nem embeddings. O número de mercados distintos é uma característica agregada já existente, mas não informa com quem uma carteira compartilha mercados.

```bash
python src/baseline.py --seed 42
python -m unittest discover -s tests -v
```

Leia `src/baseline.py` nesta ordem:

1. `carregar_entrada`: carrega as carteiras, a matriz e o preprocessador. Reconstrói as features a partir do CSV e compara com a matriz para evitar associar um score à linha errada.
2. `ajustar_ranking`: ajusta IsolationForest, calcula scores, ordena e seleciona o top. O ajuste usa apenas as features; endereços e IDs não entram no modelo.
3. `executar`: salva os resultados, o modelo e um manifesto com versões, parâmetros, hashes das entradas e tempos.
4. `main`: recebe argumentos do terminal e escolhe a pasta de saída.

IsolationForest separa observações por divisões aleatórias em características. Pontos isolados com caminhos mais curtos tendem a ser considerados anômalos. Escolhemos essa primeira referência por considerar combinações de variáveis; um z-score seria mais simples de interpretar por coluna, mas exigiria uma regra adicional para combinar desvios. A floresta não fornece, neste script, uma explicação causal por carteira.

Configuração inicial: 100 árvores (`--trees`), `max_samples='auto'` (256 carteiras por árvore nesta execução), seed 42 e uma thread. Reutilizamos as features com log1p e StandardScaler já preparadas, sem reajustar o preprocessador. A padronização não é uma exigência das árvores; aqui mantém a mesma entrada numérica disponível para a GNN. O log1p altera a distribuição usada para escolher os cortes.

O score salvo é `-model.score_samples(X)`: maior significa mais anômalo. Não é probabilidade de fraude. `contamination='auto'` não define nosso top; não usamos `predict` para rotular carteiras. O top é uma fila de revisão com `ceil(n * top_fraction)` posições, por padrão 1%. Empates são resolvidos por ID crescente, sem significado adicional de risco.

Saídas em `data/processed/dev/baseline_seed_42/`:

- `ranking.csv`: todas as carteiras, seus valores originais, posição, score e flag de seleção.
- `top.csv`: as primeiras 50 carteiras nesta amostra.
- `modelo.joblib`: floresta ajustada (o preprocessador continua na pasta de entrada).
- `manifest.json`: parâmetros, versão do sklearn, hashes, quantis dos scores e tempos separados de ajuste/scoring.

O ranking mantém colunas originais para consulta, inclusive colunas que NÃO foram usadas no ajuste. A lista exata das cinco features utilizadas está no manifesto.

Ajuste e scoring são realizados nas mesmas 5.000 carteiras: exploração transdutiva, sem rótulos e sem teste de generalização. Não reportamos acurácia, precisão ou recall. A comparação futura com a GNN deve usar as mesmas carteiras; estabilidade em três seeds ainda será avaliada. A repetição de uma única seed só verifica reprodutibilidade.

Execução inicial real: ajuste em aproximadamente 0,128 s e scoring em 0,025 s, excluindo inicialização, leitura, validação e escrita. Não são estimativas de tempo da GNN nem um benchmark repetido. O modelo recarregado reproduziu os scores, e um novo ajuste com seed 42 reproduziu o ranking.

Referência do algoritmo e convenção de score: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html

O ranking do baseline já foi inspecionado exploratoriamente; a comparação com a GNN ainda está pendente. A visualização do grafo permanece para depois do ranking da GNN, conforme combinado.

## GNN — etapa 1: divisao e arquitetura

```bash
python src/preparar_gnn.py --seed 42
python -m unittest discover -s tests -v
```

`src/preparar_gnn.py` le o `hetero.pt` local e cria um novo grafo para a tarefa de existencia de ligacao: cada par carteira-mercado aparece uma unica vez, independentemente do outcome. O grafo original, seus atributos de aresta e outcomes permanecem preservados. A primeira GNN nao recebe `edge_attr` nem outcomes.

Os pares unicos sao divididos com seed fixa: 10% validacao, 10% teste e o restante treino (arredondamento para baixo nos dois primeiros). Dentro do treino, 25% ficam como alvos e os demais como contexto para propagacao. Alvos e contexto sao disjuntos. Somente as conexoes de contexto recebem reversas, evitando vazamento de conectividade entre particoes. O mesmo contexto sera usado em treino, validacao e teste.

Saidas em `data/processed/dev/gnn_seed_42/`: `divisao.pt` e `divisao_manifest.json`. O arquivo PT contem `context`, `train_positive`, `val_positive`, `test_positive` e `all_positive`. Este ultimo devera ser usado para excluir todos os pares observados da futura amostragem negativa, inclusive os reservados para avaliacao. Os negativos são gerados e reservados na etapa de treinamento descrita abaixo.

Resultado real: 39.562 pares unicos a partir de 62.328 linhas, sendo 23.738 de contexto, 7.912 alvos de treino, 3.956 de validacao e 3.956 de teste. Todos os nos foram preservados. Ha 975 carteiras e 2.098 mercados sem vizinhos no contexto: GraphSAGE ainda utiliza suas features proprias, mas esses nos nao recebem mensagens de vizinhos. A avaliacao futura deve distinguir resultados com e sem contexto de vizinhanca.

As features e seus scalers continuam calculados sobre o recorte selecionado inteiro, inclusive agregados de apostas reservadas. Esta etapa evita vazamento direto de conectividade, mas NAO constitui um teste independente de todas as informacoes das arestas ocultadas. O protocolo e transdutivo, condicionado a essas features; nao representa previsao temporal. Para alegar generalizacao sobre atividade nao observada sera preciso recalcular as features com dados permitidos pelo protocolo.

`src/modelo_gnn.py` implementa `GNNBipartida`:

1. `layer1`: HeteroConv com uma SAGEConv por direcao, saida de 64 dimensoes.
2. ReLU e dropout 0,1 (dropout desativado em avaliacao).
3. `layer2`: outra HeteroConv, gerando embeddings de 32 dimensoes por no.
4. `decode`: concatena o embedding da carteira com o do mercado e usa uma rede pequena (64 unidades intermediarias) para retornar um logit por par. O treinamento usa BCEWithLogitsLoss; nao aplicar sigmoid antes dessa perda.

O decoder MLP foi escolhido para aprender compatibilidade entre os dois tipos de no sem impor apenas produto interno. Com as dimensoes da amostra, o modelo tem 20.673 parametros treinaveis. Isso e uma configuracao inicial, nao resultado de busca de hiperparametros.

O teste `tests/test_gnn.py` usa um grafo ficticio com pares repetidos para conferir particoes disjuntas, preservacao do original, reversas, reproducibilidade, dimensoes e gradientes finitos do encoder/decoder. Um forward no contexto real tambem passou: embeddings de carteiras [5000, 32], mercados [9189, 32] e 7.912 logits de treino. Esse teste inicial usou pesos aleatórios; os resultados aprendidos pertencem aos checkpoints da etapa de treinamento.

Negativos, loop de treino, parada antecipada, métricas de validação e benchmark estão implementados na etapa 2. Teste fica reservado para a avaliacao final. Optuna permanece opcional; o ranking GNN foi implementado na etapa 3 abaixo.

## GNN — etapa 2: treinamento em CPU

```bash
python src/treinar_gnn.py --epochs 100 --patience 15 --threads 2
```

A entrada padrao e `data/processed/dev/gnn_seed_42/divisao.pt`. `--seed` controla inicializacao e dropout; `--negative-seed` controla negativos de treino e `--eval-seed` controla os negativos de avaliacao (teste usa eval-seed + 1). Nenhuma dessas flags altera a divisao fornecida. Use a mesma divisao para estudar a variabilidade do treinamento; mude `--input` explicitamente para outro experimento de divisao. A saida padrao recebe timestamp UTC e seed; `--output` permite escolher outra pasta, que nao pode existir para evitar sobrescrita.

Funcoes de `src/treinar_gnn.py`:

- `validar_divisao`: confere limites, pares unicos, cobertura, particoes disjuntas e reversas.
- `amostrar_negativos`: sorteia pares carteira-mercado sem conexao observada, sem matriz densa e sem repeticoes dentro de cada amostra. Ausencia de conexao nao significa legitimidade.
- `exemplos`: combina positivos e negativos com alvos 1 e 0.
- `avaliar_validacao`: usa modo de avaliacao (dropout desligado), calcula BCE, Average Precision (AP) e ROC-AUC, incluindo grupos com ambos os endpoints conectados ao contexto ou algum endpoint isolado.
- `treinar`: executa o loop, atualiza pesos, salva o checkpoint e verifica sua recarga.

Cada epoca realiza um passo de otimizacao usando todo o grafo de contexto e todos os pares-alvo de treino (full-batch). Ha um negativo por positivo. Negativos de validacao e teste sao fixados antes do treino e excluidos do sorteio de treino; todos os positivos observados tambem sao excluidos. Negativos de treino sao renovados por epoca. Os pares de teste sao apenas reservados; suas metricas NAO sao calculadas.

Configuracao inicial: Adam, learning rate 0,001, hidden 64, embedding 32, dropout 0,1 e clipping da norma de gradiente em 5. A inicializacao e deterministica em CPU com seed 42 e duas threads. A melhor configuracao de pesos e selecionada por menor perda de validacao, exigindo reducao maior que 0,0001; interrompemos apos 15 epocas sem essa melhora ou ao atingir 100 epocas. A rede sem treino tambem e medida e pode permanecer como melhor checkpoint se nenhuma epoca melhorar.

Arquivos por execucao:

- `historico.csv`: loss de treino, loss/AP/AUC de validacao, norma de gradiente antes do clipping e tempo por epoca.
- `melhor_modelo.pt`: pesos, arquitetura, seed, epoca e metricas do checkpoint selecionado. E um checkpoint para inferencia; nao inclui estado de Adam para retomada exata do treino.
- `validacao.pt` e `teste_reservado.pt`: pares e alvos fixos.
- `treino_manifest.json`: versoes, hash da divisao, parametros, metricas iniciais/finais e tempos.

Os tempos por epoca incluem sorteio, forward, backward, atualizacao e validacao, mas nao escrita dos artefatos. O tempo do loop inclui essas escritas; nao inclui inicializacao/leitura e validacao inicial. Sao medicoes de uma execucao, nao previsao do custo no dataset completo.

AP e ROC-AUC aqui avaliam conexoes observadas contra negativos uniformes, NAO fraude. A validacao global tem proporcao 1:1; os subgrupos podem ter proporcoes diferentes, portanto suas APs nao sao diretamente equivalentes. Features agregadas incluem informacoes do recorte inteiro, e negativos uniformes podem tornar a tarefa facil. Uma melhora nao demonstra ganho causado pela estrutura: comparacoes futuras com modelos sem mensagens e controles por grau sao necessarias para essa conclusao. Nao foi feita busca Optuna nem avaliacao final de teste.

O teste de integracao verifica negativos validos, reproducibilidade do historico e checkpoint, isolamento das particoes e protecao contra sobrescrita. A comparacao controlada e o ranking estao descritos na etapa 3. A visualizacao do grafo permanece para depois do ranking.

Execucao historica com o protocolo inicial de treino (anterior aos controles da etapa 3): `data/processed/dev/gnn_seed_42/treino_20261017T230054027371Z_seed_42`. 100 epocas, melhor checkpoint na epoca 100; BCE de validacao 0.695289 -> 0.297171; AP 0.4972 -> 0.9445; ROC-AUC final 0.9461. Mediana de 0.091 s/epoca e 9.43 s no loop. Parou pelo limite de epocas, nao por convergencia ou paciencia esgotada. Uma execucao exploratoria anterior, sem negativos de teste reservados, foi preservada apenas como historico e nao deve ser usada como resultado desta versao.


## GNN — etapa 3: controles, estabilidade e rankings

Executar na raiz com a .venv ativada (WSL/Linux):

```bash
python -m unittest discover -s tests -v
python src/experimento_gnn.py
```

A pasta de saida recebe timestamp. Para escolher o destino, use `--output data/processed/dev/NOME_NOVO`; uma pasta existente e rejeitada. O experimento completo executa 12 treinos: GNN e MLP sem mensagens, duas condicoes de features e seeds 42, 43 e 44. Cada treino tem ate 100 epocas e paciencia 15, com o mesmo decoder, dimensoes e hiperparametros. A quantidade de parametros difere entre encoders e e registrada, portanto nao e uma comparacao com capacidade exatamente igual.

- `full_day`: features originais do dia inteiro; inclui agregados relacionados a conexoes ocultadas. Mantida como referencia transdutiva.
- `context_only`: agrega todos os outcomes apenas dos pares de contexto. Contagens, volume, ticket e preco das carteiras sao recalculados. Scalers, medianas e categorias sao ajustados nos nos ativos no contexto. Carteiras isoladas recebem contagens/volume/ticket zero, preco mediano dos ativos e flag `sem_contexto`. Metadados dos mercados continuam transdutivos; isto nao transforma o experimento em previsao temporal.

As duas condicoes preservam nos e particoes. A divisao existente fica fixa, assim como as seeds de avaliacao (43, teste reservado 44), negativos de treino (42) e IsolationForest (42). Apenas inicializacao e dropout variam entre as tres seeds do encoder. Os exemplos de avaliacao sao conferidos por igualdade exata entre todas as execucoes. O teste permanece reservado; nao ha metricas de teste.

Funcoes principais:

- `features_contexto.py`: `alinhar` organiza pelos IDs originais; `features_contexto` recalcula os agregados permitidos e salva a definicao dos transformadores.
- `modelo_gnn.py`: `MLPSemMensagens` codifica cada no individualmente, ignorando arestas, e usa o mesmo tipo de decoder da GNN.
- `experimento_gnn.py`: `executar` registra o protocolo antes dos treinos, roda os controles e consolida resultados; `comparar_rankings` calcula Spearman com postos medios para empates, intersecao e Jaccard nos cortes 0,5%, 1% e 2%. Correlacao de scores constantes e indefinida e fica vazia.
- `ranking_gnn.py`: `exportar` confere o hash da divisao do checkpoint, extrai embeddings no contexto fixo e preserva os IDs; `salvar_ranking` aplica IsolationForest e salva scores, top 1% e detector.

O controle de atividade usa produto dos graus dos endpoints no contexto e calcula AP/AUC de ligacao. Nao recebe BCE porque o produto nao e um logit calibrado. Rankings sao gerados para GNN, MLP e features individuais; scores de metodos diferentes nao sao probabilidades nem possuem escala diretamente comparavel.

Artefatos por experimento (em `data/`, ignorado pelo Git):

- `protocolo.json`: seeds, orcamento, condicoes, hashes de entradas e codigo.
- `preprocessador_contexto.joblib`, `features_contexto.csv`: transformadores e agregados permitidos.
- `resultados.csv`, `resumo_seeds.csv`, `controle_grau.json`: resultados individuais, media/desvio-padrao amostral e controle de grau.
- `comparacoes_rankings.csv`, `integridade.json`: estabilidade e igualdade dos exemplos.
- Em cada condicao/modelo/seed: historico, checkpoint, manifestos e pasta `ranking/` com `ranking.csv`, `top.csv`, `embeddings.npz`, `indices_originais.json` e `isolation_forest.joblib`. O baseline de features fica em `baseline_features/` de cada condicao.

Os embeddings incluem carteiras e mercados; o ranking e de carteiras. Sao usados os checkpoints selecionados por validacao, sem reajustar no grafo completo. O top 1% corresponde a 50 carteiras priorizadas para inspecao, nao a 50 fraudes. Carteiras sem contexto permanecem sinalizadas no ranking. Os testes cobrem isolamento das features, alinhamento de IDs, MLP independente das arestas, avaliacao fixa entre seeds/modelos, exportacao e rejeicao de checkpoint de outra divisao.

Resultados desta rodada e interpretacao: [relatorio do experimento](docs/resultados_gnn_v1.md). Os resultados anteriores permanecem preservados. Proximas etapas: inspecao com controles e visualizacao dos subgrafos; a ablacao com vizinhos aleatorizados e a sensibilidade a seed do IsolationForest ainda estao pendentes. Makefile, conjunto completo e avaliacao final continuam para etapas posteriores.

## Curvas ROC e novas seeds

Para repetir a comparação com as seeds escolhidas (inicialização/dropout):

```bash
python src/experimento_gnn.py --seeds 42 73 2026 --output data/processed/dev/experimento_seeds_42_73_2026
python src/avaliar_curvas.py --input data/processed/dev/experimento_seeds_42_73_2026 --output data/processed/dev/experimento_seeds_42_73_2026/curvas
```

Escolha destinos novos se essas pastas já existirem. A seed da divisão, os negativos de treino/avaliação e o IsolationForest continuam fixos. A rodada anterior (42, 43, 44) é preservada, sem selecionar só as melhores execuções.

`avaliar_curvas.py` recarrega os checkpoints selecionados por BCE, confere o hash da divisão, verifica a igualdade dos pares de validação e reproduz a AUC registrada. Salva ROC por seed/modelo e controle por grau, perdas por época, PNG/PDF e os pontos das curvas em CSV. Os logits são usados diretamente: sigmoid preservaria a ordenação da ROC. Não abre o arquivo de teste reservado. A ROC é de reconstrução de conexões, não de fraude, e usa a mesma validação que selecionou os checkpoints; não substitui avaliação final independente.

Resumo da rodada: [seeds 42, 73 e 2026](docs/resultados_seeds_42_73_2026.md).

Para gerar ROC média, faixa de ±1 desvio-padrão amostral e maior/menor AUC entre seeds:

```bash
python src/resumir_roc.py --input data/processed/dev/experimento_seeds_42_73_2026/curvas
```

Saídas `roc_resumo_CONDICAO_MODELO.png/.pdf/.csv`, separadas por condição e modelo. A média usa interpolação numa grade comum de FPR; a legenda informa média/desvio das AUCs individuais. As três seeds compartilham divisão e exemplos: não são folds nem réplicas independentes de dados. A faixa é descritiva, não intervalo de confiança; os extremos são mostrados apenas para comparação, sem selecionar uma seed vencedora.


## Piloto sintético v2 — concluído em 25/10/2026

A correção da elegibilidade permitiu completar 50 alvos com 253 trocas, preservando graus e positivos reservados. Foram executados 12 treinos e avaliados 14 rankings. Resultado: GNN + IF e MLP + IF recuperaram zero alvos no top 100 em todas as seeds; features + IF recuperou 2 no top 50 e 3 no top 100. Isso registra uma limitação da recuperação deste padrão artificial, apesar da boa validação de ligações.

Os 21 testes passaram e os scores/métricas dos 14 rankings foram recalculados na auditoria. Veja [protocolo v2](docs/protocolo_piloto_v2.md), [relatório e comandos de reprodução](docs/piloto_validacao_v2/resultado_piloto_v2.md) e [figura de recuperação](docs/piloto_validacao_v2/recuperacao_piloto_v2.png). A v1 inviável continua preservada. As realizações sintéticas seguintes e a inspeção descritiva foram concluídas; os relatórios atuais estão no início deste documento. Permanecem pendentes o teste final de ligações e a revisão humana independente.


## O que compartilhar pelo Git

O repositório contém código, testes, dependências, mapeamentos de IDs, protocolos, métricas resumidas e figuras PNG dos relatórios. Os resultados antigos são mantidos como evidência da evolução experimental, inclusive tentativas sem sucesso. PDFs das mesmas figuras são exportações regeneráveis e ficam fora do Git.

`data/`, `.venv/`, checkpoints `.pt` e modelos/scalers `.joblib` permanecem locais. Para reproduzir em outro computador, criar a própria `.venv` e disponibilizar os CSVs de entrada separadamente; as fichas reais ficam em `data/processed/validacao_ampliada_v1/casos_reais/` e podem ser compartilhadas separadamente ou regeneradas. Os manifestos históricos registram caminhos absolutos da máquina de execução: preservá-los como evidência; para uma nova execução, gerar novos manifestos em vez de adulterar os antigos.

Relatórios principais: [piloto v2](docs/piloto_validacao_v2/resultado_piloto_v2.md), [investigação](docs/investigacao_anomalias/resultado.md) e [validação ampliada](docs/validacao_ampliada/resultado.md). Cada relatório inclui comandos e limites de reprodução.
