# Betgraph Anomaly

Projeto universitário de detecção não supervisionada de carteiras anômalas em dados do Polymarket (Polygon), no recorte de 16/09/2026. Um score de anomalia não confirma fraude. A metodologia pode ser estudada em outros mercados, mas os valores não são numericamente comparáveis a casas de apostas tradicionais.

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

## Status

Limpeza e agregação existentes; ambiente CPU, features e construção de HeteroData validados. Baseline IsolationForest e seu ranking implementados na amostra. Treino da GNN, comparação entre modelos, validação em três seeds e exportação para Gephi ainda estão pendentes.

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

**Próxima decisão do modelo:** SAGEConv não consome `edge_attr` diretamente. Os atributos estão disponíveis no grafo, mas seu uso pelo encoder ou decoder precisa ser implementado explicitamente. Para carregar um `HeteroData` próprio, use `torch.load(caminho, weights_only=False)`; carregue somente artefatos de confiança.

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

Próximo passo didático: inspecionar o ranking e entender seus limites antes de implementar a GNN. A visualização do grafo permanece para depois do ranking da GNN, conforme combinado.
