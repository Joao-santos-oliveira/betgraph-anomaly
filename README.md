![Betgraph Anomaly: carteiras e mercados de previsão](assets/betgraph.svg)

# Betgraph Anomaly

## Estudo principal

O Betgraph Anomaly avalia uma **rede neural em grafos heterogêneos (GNN V2 híbrida)** para priorizar carteiras com coparticipação atípica na Polymarket. O recorte de 16/09/2026 contém 27.039 carteiras e 16.693 mercados após a filtragem. No grafo bipartido, cada ligação representa uma carteira que operou em um mercado. A pergunta de pesquisa é quais padrões de atuação em grupo a GNN consegue trazer ao topo da fila de investigação e como a recuperação varia com a intensidade.

### Um dia de atividade real

Contamos 229.072 pares distintos no dia, dos quais 137.444 entram no contexto da rede. A carteira 18753 ocupa a primeira posição da V2 nas seeds 42, 73 e 2026. Ela compartilhou 64 mercados com a carteira 18626, ante uma sobreposição esperada de aproximadamente 7,51 pelo modelo relacional. Mercados de uma mesma partida podem explicar parte de uma coincidência. Os dados permitem investigar essa relação, mas atribuir coordenação ou fraude exige evidência independente. [Top 100 da V2 na seed 42](resultados/artigo/top100_dia_real.csv) · [Posições das 27.039 carteiras por método e seed](resultados/artigo/posicoes_dia_real.csv)

### Controle positivo e ablação do score

Em cada realização, 50 carteiras existentes formam dez grupos de cinco. Cada grupo passa a compartilhar a mesma lista de 4, 10 ou 20 novos mercados. As trocas preservam o grau de cada carteira e mercado e os pares reservados. Cada alvo tem um controle intacto de mesmo grau. O desenho alterna as doses entre três sorteios e totaliza 50 participações-alvo por intensidade. [Protocolo](experimentos/artigo/protocolo.json)

Os três scores consideram os mesmos pares candidatos com pelo menos dois mercados em comum e agregam as três maiores contribuições por carteira, dividindo a soma por três. O relacional mede a surpresa da sobreposição. A ablação usa apenas a similaridade de cosseno dos embeddings da V2, reescalada para o intervalo de 0 a 1. A V2 híbrida multiplica os dois termos para cada par. É uma ablação do **score final**. Os embeddings continuam recebendo os atributos estruturais usados no treinamento original.

| Novos mercados comuns por grupo | V2 híbrida | Apenas similaridade V2 | Relacional direto |
|---|---:|---:|---:|
| 4 | 2 / 7 | 0 / 0 | 2 / 7 |
| 10 | 0 / 24,33 | 0 / 0 | 0 / 26 |
| 20 | 50 / 50 | 0 / 0 | 50 / 50 |

Cada célula informa **alvos no top 100 / alvos no top 541** após a intervenção, acumulados nos três sorteios. As contagens das redes são médias entre as três seeds sobre os mesmos alvos. O top 100 é o corte principal. O top 541 cobre cerca de 2% do dia. Esse corte complementar foi introduzido após a primeira análise do dia completo e constava do protocolo do teste de grupos antes de calcular seus rankings. O sinal de surpresa sozinho recuperou todos os casos fortes. A ponderação aprendida preservou essa recuperação e não acrescentou ganho médio no nível intermediário. A interpretação se limita às intervenções e ao dia avaliados. [Resumo antes e depois, com controles](resultados/artigo/resumo.csv) · [Resultados por realização e seed](resultados/artigo/por_realizacao_seed.csv)

O experimento anterior de concentração em um conjunto de dez mercados permitia que as carteiras escolhessem subconjuntos diferentes. Naquele desenho, o top 100 do dia completo não recuperou nenhum alvo estrutural pela V2 ou pelo relacional. A tabela de intensidades acima usa listas idênticas impostas a todos os integrantes de cada grupo. Os tempos da comparação de escalas foram medidos com processos concorrentes. [Comparação anterior entre 5.000 carteiras e o dia completo](resultados/artigo/comparacao_escalas.csv)

### Reprodução do estudo principal

Em Linux ou WSL, use Python 3.12 e GNU Make. Coloque os dois arquivos brutos `betgraph_arestas_16set.csv` e `betgraph_nos_evento_16set.csv` em `data/raw/`. O executor confere os hashes dos dados, filtra os CSVs, monta o grafo do dia, recria as três intervenções, treina a V2 e calcula os três rankings. As dependências e os dados volumosos permanecem locais. [Hashes das entradas](resultados/entradas.json)

```bash
make ambiente
make validar
make testes
make artigo
```

Use `BRUTOS=/caminho/para/csvs` e `DESTINO=/caminho/para/execucoes` para escolher as pastas. O padrão é um processo de treino por vez. `make artigo PROCESSOS=3` permite treinar as três seeds em paralelo quando há memória disponível. Cada chamada cria uma pasta exclusiva e imprime a localização dos CSVs de resultados. O código do artigo fica em [experimentos/artigo](experimentos/artigo) e o modelo em [experimentos/gnn_v2](experimentos/gnn_v2).

Para reavaliar embeddings salvos de uma execução, use:

```bash
.venv/bin/python -m experimentos.artigo.ablacao PASTA_DA_EXECUCAO PASTA_DE_SAIDA
```

## Estudos de desenvolvimento

Os comandos abaixo reproduzem os estudos anteriores. As branches [V2](https://github.com/Joao-santos-oliveira/betgraph-anomaly/tree/feat/gnn-v2-anomalias), [V3](https://github.com/Joao-santos-oliveira/betgraph-anomaly/tree/feat/gnn-v3-anomalias) e [evidências da comparação anterior](https://github.com/Joao-santos-oliveira/betgraph-anomaly/tree/feat/evidencias-artigo) preservam as respectivas rodadas.

<details>
<summary>Comandos históricos, comparação das versões, revisão de casos e proveniência</summary>

## Testar os modelos e as anomalias na main

Na raiz do clone, instale o ambiente com `make ambiente` e coloque os dois brutos em `data/raw/`, conforme indicado abaixo. Os comandos desta tabela usam **todas as 27.039 carteiras** por padrão. Acrescente **`ESCOPO=5000`** a qualquer linha para usar a segunda coorte de 5.000 carteiras da comparação v2/v3.

| Comando | Modelos e dados |
|---|---|
| `make baselines` | Relacional e IsolationForest sobre atributos, no contexto original |
| `make redes` | GNN inicial + IsolationForest e MLP + IsolationForest, no contexto original |
| `make comparar` | Os quatro métodos no mesmo contexto |
| `make anomalias-volume` | Os quatro métodos após multiplicar por dez o volume das arestas de contexto de 50 carteiras elegíveis |
| `make anomalias-estrutura` | Os quatro métodos após concentrar pelo menos 60% das conexões de 50 carteiras elegíveis em dez mercados, preservando graus |

Exemplos: `make anomalias-volume GRUPO=redes` treina GNN e MLP no dia completo alterado. `make anomalias-estrutura ESCOPO=5000 GRUPO=baselines` testa somente os dois baselines na amostra. Os atalhos `make comparar-5000`, `make redes-5000` e `make baselines-5000` continuam disponíveis. `BRUTOS` e `DESTINO` permitem escolher pastas. Cada chamada cria uma execução isolada.

**Protocolo.** O universo completo usa a divisão histórica com seed 42 e 137.444 pares de contexto; a amostra usa a divisão 9103. Para cada intervenção, todos os métodos recebem a mesma cópia, os mesmos alvos e as mesmas partições. Os atributos de contexto são recalculados após a intervenção. A injeção estrutural usa as restrições e o limite de tentativas do piloto v2; se for inviável, a execução grava o diagnóstico e termina. A seleção dos alvos usa `INJECAO=40101` por padrão. Para uma segunda realização predefinida, repita com `INJECAO=40201`. As redes usam seeds 42, 73 e 2026 e até 100 épocas. Para conferir a instalação, `EPOCAS=1` permite um teste curto. Registre o orçamento usado ao comparar métodos.

**Saídas.** Abra `resultados/` no caminho impresso no terminal. Cada método gera um ranking completo por seed; `coorte.json` identifica o universo e os pares de contexto, e `custos.csv` registra tempo e pico de memória por execução. Nos cenários artificiais, `rotulos.csv` identifica alvos/doadoras e `metricas.csv` reúne os resultados. Para o universo completo, consulte `recovered_100`, `recall_100`, `precision_50`, `ap` e `auc`. Os rótulos entram na avaliação após o ranking. Como o número de alvos é fixo, a prevalência muda de 1% na amostra para aproximadamente 0,185% no dia completo. Esses cenários são novas realizações de estrutura e volume da main. A rodada histórica da v3 tem também intervenções combinadas e decoys próprias.

## Reproduzir a partir dos CSVs brutos

Use **um clone Git com histórico**, Linux ou Ubuntu no WSL, GNU Make, Python 3.12 com suporte a venv e acesso à internet para instalar dependências. As revisões históricas usadas pelo executor estão no histórico Git das branches. Em um clone raso, obtenha primeiro o histórico completo. A primeira execução exige somente os **dois arquivos de entrada** abaixo. Os mapeamentos de IDs persistentes acompanham o código.

| CSV em `data/raw/` | Unidade da entrada |
|---|---|
| `betgraph_arestas_16set.csv` | Operações já agregadas por carteira, mercado e outcome, incluindo primeiro e último trade |
| `betgraph_nos_evento_16set.csv` | Mercado identificado por condition_id, pergunta, tags e metadados |

A publicação dos brutos no Zenodo está prevista. O identificador da edição está nos [hashes SHA-256](resultados/entradas.json). A etapa inicial confere ambos os arquivos. A partir da raiz do clone:

```bash
make ambiente
make validar
make reproduzir
```

Para ler os mesmos arquivos diretamente da pasta de download e escolher onde guardar as execuções:

```bash
make reproduzir BRUTOS="/caminho/para/os/csvs" DESTINO="/caminho/para/execucoes"
```

Cada chamada cria uma pasta exclusiva em `data/execucoes/` por padrão. Dentro dela, o executor recupera localmente o código do experimento, processa os IDs, deriva duração e categoria, exclui mercados recorrentes e suas arestas e agrega as carteiras. Confere os hashes dos **três CSVs tratados regenerados** antes de construir os grafos. Os hashes dos arquivos brutos, a versão e a conclusão ficam em `execucao.json`; a pasta `resultados/` da execução dá acesso às saídas calculadas. Os relatórios auditados usados na comparação permanecem na [pasta resultados do repositório](resultados/).

| Comando | Finalidade |
|---|---|
| `make ambiente` | Criar a venv CPU e instalar as versões de requirements.txt |
| `make validar` | Conferir dependências e executar uma operação PyTorch/PyG |
| `make testes` | Rodar os testes da branch |
| `make reproduzir` | Partir dos CSVs brutos e executar o benchmark histórico dos dois detectores de referência |
| `make historico` | Executar também o controle GNN/MLP de ligações e suas curvas ROC, o piloto viável, a investigação relacional e a validação ampliada com as fichas de casos |

**Saída esperada de `make reproduzir`.** O pipeline primeiro verifica a amostra de 256 carteiras (3.084 mercados, 5.228 linhas e 2.241 pares de contexto) e a ordem integral dos dois rankings. Depois mede o dia completo e confere o resumo e os SHA-256 dos dois rankings completos contra a referência. Para a reprodução em outra máquina, devolver `execucao.json` e o comprovante da amostra acessível em `resultados/` da execução.

## Ficha comparativa das três branches

Valores de recuperação indicam **quantos dos 50 alvos artificiais apareceram nas primeiras 100 posições**, em 5.000 carteiras. A avaliação técnica de escala usa todo o dia tratado. O controle de decoys mede a priorização indesejada de carteiras com coparticipação forte, lados balanceados e horários dispersos.

| Branch e pergunta | Evidência observada | Contribuição e limite |
|---|---|---|
| **main** · Que padrões os métodos simples capturam? | Na comparação pareada da rodada v3, relacional direto recuperou **32/50** com estrutura forte e atributos + IsolationForest recuperou **9/50** com volume ×10. No controle, o relacional destacou **29,5/50 decoys**. | Referências interpretáveis e de baixo custo, sensíveis a tipos diferentes de alteração. Mercados associados ao mesmo evento podem explicar coparticipação. |
| **v2** · O embedding melhora um detector estrutural? | No experimento original, o híbrido contrastivo recuperou **31,67/50**, contra **32/50** do relacional no mesmo cenário de 60%. Seus top 100 compartilharam **94 a 98** carteiras. | Isola o efeito da representação aprendida sobre um score estrutural forte. O ganho consistente exigido pelo protocolo ficou por demonstrar. |
| **v3** · Atributos de aresta, outcome e horário melhoram a priorização? | No cenário combinado S1, a GNN principal ficou em **0/50** nas seis combinações. AP média de **0,0300**, ante **0,0343** do relacional e **0,0454** do manual. | Testa diretamente a informação de aresta, com MLP sem mensagens e controle de decoys. A GNN principal também destacou zero decoys no top 100 global. |

**Comparação nas mesmas intervenções da v3.** A referência v2 foi retreinada nas realizações 30101/30201. Os 31,67 do experimento original da v2 vêm de outras realizações. A tabela abaixo usa apenas as realizações compartilhadas.

| Método | S1 combinado | S2 decoys ↓ | S3 estrutura 60% | S3 volume ×10 |
|---|---:|---:|---:|---:|
| Relacional direto | 0,50 | 29,50 | 32,00 | 0,50 |
| Atributos + IsolationForest | 0 | 2,00 | 0 | 9,00 |
| Híbrido GNN v2 | 0,50 | 29,00 | 31,83 | 0,50 |
| GNN v3, score de pares | 0 | 0 | 0 | 0 |
| GNN v3 + IsolationForest | 0,17 | 0,17 | 0 | 3,83 |

As três seeds de encoder são 42, 73 e 2026. As intervenções artificiais fornecem os rótulos para essa avaliação. A capacidade de identificar fraude em carteiras reais permanece desconhecida. [Tabela original v2](https://github.com/Joao-santos-oliveira/betgraph-anomaly/blob/c0b7390/docs/gnn_v2/resumo.csv) · [Métricas originais v3](https://github.com/Joao-santos-oliveira/betgraph-anomaly/blob/deb24e5/docs/gnn_v3/metricas.csv)

## Onde encontrar os resultados

| Etapa | Arquivo para abrir |
|---|---|
| Filtragem e resultados esperados da amostra | [Assinaturas das entradas](resultados/entradas.json) · [referência dos rankings](resultados/reproducao/referencia.json) |
| Piloto sintético concluído | [Resumo entre seeds](resultados/piloto/resumo_seeds.csv) · [métricas completas](resultados/piloto/metricas.csv) |
| Robustez por intensidade e recorte | [Tabela consolidada](resultados/validacao/resumo.csv) · [figura](resultados/validacao/robustez.png) |
| Escala no dia completo | [Rankings e sobreposição resumidos](resultados/escala/resumo.json) · [custos por estágio](resultados/escala/custos.json) |
| Revisão descritiva individual | [Denominadores por grupo](resultados/revisao/resumo.csv) |

**Piloto sintético.** A primeira tentativa parou antes do treino: uma carteira precisava alcançar dez mercados, mas só oito pares eram permitidos. A correção do gerador preservou a intensidade e as seeds e tornou viável a intervenção em 50 carteiras. Por isso a pasta `resultados/piloto/` apresenta a rodada concluída. O diagnóstico anterior continua disponível no [histórico do piloto v1](https://github.com/Joao-santos-oliveira/betgraph-anomaly/blob/19c9f5a/docs/piloto_validacao/resultado_piloto_v1.md). Essa numeração do gerador é independente das versões das redes.

**Escala.** Dos 560.985 registros brutos de arestas e 22.418 mercados iniciais, a filtragem preservou 358.689 linhas agregadas, 27.039 carteiras e 16.693 mercados. O dia contém 229.072 pares únicos; 137.444 formaram o contexto da divisão com seed 42. Os métodos compartilharam duas carteiras no top 50 e 13 no top 100. Na execução de referência, os cinco estágios do benchmark somaram 56,7 s e o pico de RSS foi 761,6 MiB. Esses tempos cobrem a preparação a partir dos tratados até os rankings, excluindo a filtragem inicial. Tempo e memória dependem da máquina. O [registro da medição original](resultados/escala/fontes_da_medicao.zip) preserva o protocolo e a versão do executor correspondentes aos hashes daquela execução.

**Revisão humana.** Um avaliador descreveu 50 fichas. Houve 25 casos, 15 controles e dez carteiras aleatórias. Das 42 fichas com parceiro, quatro pares recíprocos reduziram o total a 38 relações distintas. Em 13 dos 15 pares caso–controle, o caso compartilhou mais mercados com seu próprio parceiro selecionado. A seleção incorpora coparticipação e permite descrições de padrões, com cegamento parcial. O formulário original encerrado tem SHA-256 `d4a83d5be1623108bf8b613a7a0e3b3203e2ad6c049f323d8c31be81db9a2cf5`. A execução histórica recria as fichas, a seleção e os controles. Os julgamentos humanos são um insumo separado para quem quiser auditar a interpretação.

## Organização e proveniência

| Pasta | Função |
|---|---|
| `src/` | Processamento de CSVs, atributos e construção do grafo |
| `experimentos/` | Detectores e estudos por cenário |
| `scripts/` | Orquestração reproduzível a partir dos brutos |
| `resultados/` | Evidências e referências desta branch |
| `tests/` | Validações da implementação |
| `assets/` | Banner do projeto |
| `data/` | Dados locais e execuções geradas |

Os mapeamentos JSON versionados na raiz conservam os identificadores anteriores à filtragem. O executor reconstrói os experimentos históricos em uma cópia de trabalho isolada, usando a base `1a1d456` para o pipeline principal, `c0b7390` para a v2 e a implementação congelada em `deb24e5` sobre a base original para a v3. Essa cópia permite respeitar os caminhos, protocolos e hashes das rodadas publicadas. Resultados de desenvolvimento referem-se a um dia, e carteiras representam endereços. As informações de maker/taker exigiriam nova extração.

</details>
