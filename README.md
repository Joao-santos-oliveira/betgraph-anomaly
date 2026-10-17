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

Limpeza e agregação existentes; ambiente CPU validado. Features e construção de HeteroData são a próxima etapa. Baseline, treino, ranking, validação em três seeds e exportação para Gephi ainda não foram implementados.
