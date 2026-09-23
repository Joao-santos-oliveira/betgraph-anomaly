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
 
1. **Extração** — Dune Analytics (conta gratuita), tabelas `polymarket_polygon.market_trades` e `polymarket_polygon.market_details`.
2. **Agregação** — queries SQL separadas para nível de aresta (trades agregados por carteira+mercado+outcome) e nível de nó (metadados do mercado).
3. **Exportação** — CSV de cada query.
4. **Construção do grafo** — Python, `networkx` para montagem/inspeção inicial do grafo bipartido.
5. **Conversão para GNN** — `torch_geometric.data.HeteroData` (`user`, `event`, aresta `bets_on` com `edge_attr`).
6. **Modelo** — GNN heterogênea (ex.: `HeteroConv`/`GraphSAGE` adaptado) para gerar embeddings dos nós.
7. **Detecção de anomalia** — não supervisionada (não há rótulo de "fraude confirmada" disponível): score de anomalia via reconstrução de grafo ou distância/clusterização no espaço de embeddings.
## Stack
 
- Extração: Dune Analytics (SQL)
- Processamento: Python, pandas, DuckDB
- Grafo: networkx, PyTorch Geometric (ou DGL)
- Modelo: GNN heterogênea, treino não supervisionado
## Recorte de dados atual
 
- **Janela:** 16/09/2026 (um dia completo), usada como protótipo antes de escalar.
- **Filtro de mercados:** excluídos os mercados "Up or Down" (curtíssimo prazo, ~60% do volume total, dominados por bots).
- **Filtro anti-bot:** descartados pares (carteira, mercado) com mais de 500 trades e ticket médio abaixo de US$ 0,50 (identificado a partir de um mercado de baixíssima liquidez onde 2 carteiras somavam ~170 mil trades de ~US$ 0,01 cada).
- **Arquivos gerados** (não versionados no Git, ver seção de setup):
  - `data/raw/betgraph_arestas_16set.csv` — 560.985 linhas (carteira, mercado, outcome agregados).
  - `data/raw/betgraph_nos_evento_16set.csv` — 22.418 linhas (metadados dos mercados: pergunta, categoria, status, datas).
- **Queries salvas no Dune:** `betgraph_arestas_16set` e `betgraph_nos_evento_16set` (ver seção de setup para reproduzir).
## Status atual
 
- [x] Definição da modelagem do grafo
- [x] Fonte de dados definida e validada (Polymarket via Dune)
- [x] Exploração do schema das tabelas (`market_trades`, `market_details`)
- [x] Dimensionamento do volume de dados e identificação de outliers (bots de market making)
- [x] Query de extração de arestas (agregada, com filtro anti-bot)
- [x] Query de nós de evento (metadados via `market_details`)
- [x] CSVs exportados para `data/raw/`
- [ ] Carregamento e limpeza dos CSVs em Python (EDA)
- [ ] Script de construção do grafo (networkx → HeteroData)
- [ ] Implementação e treino do modelo GNN
- [ ] Avaliação dos resultados / análise de casos anômalos detectados
## Próximos passos
 
- Carregar os CSVs em Python, tratar tipos e limpar dados
- Análise exploratória (distribuições de volume, trades por carteira, checagem de proxy wallets)
- Escrever o script de ingestão e construção do grafo
- Prototipar a arquitetura da GNN
---
 
## Setup do ambiente (para rodar o projeto pela primeira vez)
 
O `.venv` (ambiente virtual Python) e a pasta `data/` **não são versionados no Git** — são específicos de cada máquina e/ou pesados demais para o repositório. Depois de clonar o repositório, siga estes passos:
 
### 1. Criar e ativar o ambiente virtual
 
```bash
python3 -m venv .venv
source .venv/bin/activate      # no Windows (PowerShell): .venv\Scripts\Activate.ps1
```
 
### 2. Instalar as dependências
 
```bash
pip install -r requirements.txt
```
 
### 3. Obter os dados
 
Os CSVs usados pelo projeto (`data/raw/*.csv`) **não estão no Git**. Duas formas de conseguir:
 
- **Opção A (recomendada):** peça o link das queries salvas no Dune (`betgraph_arestas_16set` e `betgraph_nos_evento_16set`) para quem já as criou, rode-as você mesmo (é necessário ter uma conta gratuita em [dune.com](https://dune.com)) e exporte os CSVs para `data/raw/`.
- **Opção B:** peça os arquivos `.csv` já exportados diretamente (Drive, pendrive, etc.) e coloque-os em `data/raw/`.
As queries SQL usadas estão documentadas em `queries/`.
 
### 4. Conferir que está tudo certo
 
```bash
python -c "import pandas, pyarrow, duckdb, networkx; print('ok')"
ls data/raw/
```
 
Se aparecer `ok` e os arquivos `betgraph_arestas_16set.csv` e `betgraph_nos_evento_16set.csv` estiverem em `data/raw/`, o ambiente está pronto para rodar os notebooks em `notebooks/`.