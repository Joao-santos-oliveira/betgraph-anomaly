![Betgraph Anomaly: carteiras e mercados de previsão](assets/betgraph.svg)

# Betgraph Anomaly

Estudo de priorização de carteiras no Polymarket em **16 de setembro de 2026**. O grafo bipartido liga carteiras aos mercados em que operaram. Uma mesma partida pode originar mercados de vencedor, handicap e totais. A pesquisa investiga quanto a estrutura dessas ligações e os atributos das operações ajudam a colocar comportamentos incomuns no início da fila de revisão.

**Branch experimental GNN v3.** Consulte a [main](https://github.com/Joao-santos-oliveira/betgraph-anomaly/tree/main) para o benchmark e a ficha comparativa geral.

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

## Resultados da GNN v3

A execução integral prepara a segunda coorte disjunta de 5.000 carteiras e oito intervenções. As três seeds de encoder, três modelos e nove grafos correspondem a 81 treinos. A GNN com atributos de aresta recuperou 0/50 no top 100 em S1 combinado nas seis combinações. Em volume ×10, sua variante secundária com IsolationForest recuperou 3,83/50 contra 9/50 do baseline de atributos. O teste de decoys e o controle MLP ajudam a interpretar o papel das mensagens.

[Tabela de resultados](resultados/v3/resumo_metricas.csv) · [recuperação incremental](resultados/v3/incremental.csv) · [decisão](resultados/v3/decisao.json) · [custos](resultados/v3/custos_treinos.csv) · [auditoria](resultados/v3/auditoria.json)

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
