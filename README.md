![Betgraph Anomaly: carteiras e mercados de previsão](assets/betgraph.svg)

# Betgraph Anomaly

Estudo de priorização de carteiras no Polymarket em **16 de setembro de 2026**. O grafo bipartido liga carteiras aos mercados em que operaram. Uma mesma partida pode originar mercados de vencedor, handicap e totais. A pesquisa investiga quanto a estrutura dessas ligações e os atributos das operações ajudam a colocar comportamentos incomuns no início da fila de revisão.

**Branch experimental GNN v2.** Consulte a [main](https://github.com/Joao-santos-oliveira/betgraph-anomaly/tree/main) para o benchmark e a ficha comparativa geral.

## Reproduzir a partir dos CSVs brutos

Use **um clone Git com histórico**, Linux ou Ubuntu no WSL, GNU Make, Python 3.12 com suporte a venv e acesso à internet para instalar dependências. As revisões históricas usadas pelo executor estão no histórico Git das branches. Em um clone raso, obtenha primeiro o histórico completo. A primeira execução exige somente os **dois arquivos de entrada** abaixo. Os mapeamentos de IDs persistentes acompanham o código.

| CSV em `data/raw/` | Unidade da entrada |
|---|---|
| `betgraph_arestas_16set.csv` | Operações já agregadas por carteira, mercado e outcome, incluindo primeiro e último trade |
| `betgraph_nos_evento_16set.csv` | Mercado identificado por condition_id, pergunta, tags e metadados |

A distribuição prevista é um depósito de dados, como Zenodo. O identificador da edição está nos [hashes SHA-256](resultados/entradas.json). A etapa inicial confere ambos os arquivos. A partir da raiz do clone:

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
| `make reproduzir` | Partir dos CSVs brutos e executar a análise principal desta branch |
| `make preparar` | Regenerar os tratados, verificar os CSVs da segunda coorte e preparar as intervenções, até a entrada dos treinos |

O comando específico desta versão é **`make hibrido`**. Para aplicar IsolationForest aos embeddings desses mesmos encoders, use **`make embeddings-if`**. Nos alvos de intervenção, `DETECTOR=if` seleciona essa variante para os encoders. As saídas ficam em diretórios independentes com o detector registrado em `execucao.json`.

## Comparação opcional no dia completo ou em 5.000 carteiras

O alvo histórico `make reproduzir` mantém a rodada publicada. Para o novo benchmark, os comandos abaixo usam as 27.039 carteiras por padrão. Escolha `ESCOPO=5000` para a mesma coorte usada na comparação experimental. Todos aceitam `BRUTOS` e `DESTINO`.

| Comando | Execução |
|---|---|
| `make redes` | GNN e controle MLP desta branch (híbrido contrastivo com features estruturais) |
| `make baselines` | Relacional direto e IsolationForest sobre os atributos de contexto |
| `make comparar` | Os quatro métodos sobre o mesmo grafo |
| `make anomalias-volume` | Volume de contexto multiplicado por dez para 50 carteiras elegíveis |
| `make anomalias-estrutura` | Coparticipação de 60% em dez mercados para 50 carteiras, preservando graus |

Para as intervenções, escolha `GRUPO=todos` (padrão), `redes` ou `baselines`. A seleção usa `INJECAO=40101` ou `40201`, como na main. As redes usam seeds 42, 73 e 2026 e até 100 épocas. Um teste de instalação pode usar `EPOCAS=1`. Compare execuções entre branches com a mesma coorte, seed de injeção e cenário. A prevalência de 50 alvos varia conforme o universo escolhido.

Cada execução produz rankings e `custos.csv` (tempo do método e pico RSS do processo). Nos cenários artificiais, consulte também `rotulos.csv`, `injecao.json` e `metricas.csv` (AP, AUC, precision@50, recall@100 e alvos recuperados). A preparação é compartilhada e antecede os tempos individuais. Os novos alvos usam os scores principais da branch; as variantes secundárias e os controles específicos da rodada original continuam em `make reproduzir`.

## Resultados da GNN v2

A execução integral prepara a segunda coorte disjunta de 5.000 carteiras, gera as intervenções, realiza 126 treinos e compara rankings em estrutura e volume. O híbrido contrastivo recuperou 31,67/50 alvos em média no top 100 com coparticipação de 60%, diante de 32/50 do detector direto. O overlap de 94–98 carteiras indica que o híbrido reproduz em grande parte a ordenação estrutural.

[Tabela de resultados](resultados/v2/resumo.csv) · [figura comparativa](resultados/v2/comparacao.png) · [dependência do relacional](resultados/v2/dependencia_hibrido.csv) · [custos](resultados/v2/custos.csv) · [auditoria](resultados/v2/auditoria.json)

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
