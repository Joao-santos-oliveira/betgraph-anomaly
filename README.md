# Betgraph anomaly

Detecção de sindicatos e padrões de fraude em mercados de apostas usando Graph Neural Networks (GNN), com dados reais on-chain do Polymarket.

## Objetivo

Em vez de prever o resultado de partidas, o projeto usa uma GNN para mapear o comportamento de usuários em um mercado de apostas e identificar padrões anômalos que indicam coordenação entre contas (sindicatos) ou lavagem de dinheiro — ex.: dezenas de carteiras aparentemente desconectadas apostando altos valores, simultaneamente, no mesmo evento de baixíssima probabilidade.

## Por que dados reais (Polymarket)

Casas de apostas tradicionais não disponibilizam dados de usuários publicamente. O Polymarket é um mercado de apostas/previsão descentralizado (rede Polygon) cujas transações ficam registradas on-chain de forma pública e auditável — carteira, evento, valor, timestamp, preço. Um estudo da Columbia University (nov/2025) já identificou indícios de wash trading nesse dataset (~25% do volume histórico, com picos próximos de 60%), o que valida o fenômeno que o projeto busca detectar.

**Limitação assumida:** os resultados não são diretamente comparáveis, numericamente, a casas de apostas tradicionais (estruturas de mercado, identificação de usuário e regulação são diferentes). O que se generaliza é a metodologia — o padrão estrutural de detecção via grafo é agnóstico à plataforma.

## Modelagem do grafo

Grafo bipartido:

- **Nó Usuário**: carteira (endereço), volume total apostado, número de trades, tamanho médio de aposta, número de mercados distintos, data da primeira atividade.
- **Nó Evento/Mercado**: `condition_id`, pergunta/questão do mercado, categoria, status de resolução.
- **Aresta (Aposta)**: valor em USD, timestamp, preço (probabilidade implícita), papel na ordem (maker/taker).

## Pipeline de dados

1. **Extração** — Dune Analytics (conta gratuita), tabela `polymarket_polygon.market_trades`.
2. **Agregação** — queries SQL separadas para nível de aresta (trades brutos) e nível de nó (features agregadas por carteira e por mercado).
3. **Exportação** — CSV de cada query.
4. **Construção do grafo** — Python, `networkx` para montagem/inspeção inicial do grafo bipartido.
5. **Conversão para GNN** — `torch_geometric.data.HeteroData` (`user`, `event`, aresta `bets_on` com `edge_attr`).
6. **Modelo** — GNN heterogênea (ex.: `HeteroConv`/`GraphSAGE` adaptado) para gerar embeddings dos nós.
7. **Detecção de anomalia** — não supervisionada (não há rótulo de "fraude confirmada" disponível): score de anomalia via reconstrução de grafo ou distância/clusterização no espaço de embeddings.

## Stack

- Extração: Dune Analytics (SQL)
- Processamento: Python, pandas
- Grafo: networkx, PyTorch Geometric (ou DGL)
- Modelo: GNN heterogênea, treino não supervisionado

## Status atual

- [ ] Definição da modelagem do grafo
- [ ] Fonte de dados definida e validada (Polymarket via Dune)
- [ ] Query inicial de extração de trades
- [ ] Query de agregação de features de nó
- [ ] Script de construção do grafo (networkx → HeteroData)
- [ ] Implementação e treino do modelo GNN
- [ ] Avaliação dos resultados / análise de casos anômalos detectados

## Próximos passos

- Rodar as queries no Dune e exportar os CSVs
- Escrever o script de ingestão e construção do grafo
- Prototipar a arquitetura da GNN
