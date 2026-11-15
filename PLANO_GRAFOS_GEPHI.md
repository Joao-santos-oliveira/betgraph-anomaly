# Plano de implementação — Grafos Gephi do artigo Betgraph

Fonte: `Betgraph-grafos-para-o-colega.pdf` (4 figuras). Escopo: somente a pasta `betgraph-anomaly/`,
na branch `main` atual, sem criar branches. Nada em `src/`, `experimentos/`, `resultados/` ou `data/raw`
é alterado; tudo novo fica em `visualizacao_gephi/` (código) e `figuras/` (entregas).

## 0. O que já existe e será reaproveitado

| Recurso | Onde | Uso |
|---|---|---|
| Arestas brutas do dia (358.689 linhas → 229.072 pares) | `data/raw/betgraph_arestas_16set_final.csv` | Fig. 1, 2B, 3 (cobertura total) |
| Metadados de mercado (`question`, `tags`, `categoria_principal`) | `data/raw/betgraph_nos_evento_16set_final.csv` | Modalidade/evento (Fig. 2A, 3, 4) |
| Score, posições, parceiros, observada/esperada | `resultados/artigo/extra/contribuicoes_casos.csv`, `casos_dia_real.csv`, `cobertura_total.csv` | Quadros da Fig. 3 |
| Rodada usada no experimento (rotulos, `dia/hetero.pt`, `dia/edge.csv`) | WSL: `/home/aliss/betgraph-main/data/execucoes/artigo-olt_5b_c/` | Fig. 4 |
| Gephi 0.10.1 + JRE próprio | `C:\Program Files\Gephi-0.10.1` | Layout, Preview, exportação PDF/SVG/PNG, `.gephi` |
| Padrão de verificação Java contra os jars do Gephi | `data/processed/gephi_bipartido_final_20261031/java_verificacao/` | Base do renderizador headless |

**Ponto de atenção (Fig. 4):** as cópias do teste PAREADO foram geradas num `mkdtemp` (`pareado-XXXX`)
que não existe mais. Como `preparar_doses` é determinístico (`default_rng(51011)`, divisão seed 42,
mesmos `rotulos.csv`), as cópias serão **regeneradas com o próprio código do experimento** e
validadas antes de desenhar (ver §4). Se a validação falhar, a figura não é gerada e isso é reportado.

## 1. Arquitetura

```
visualizacao_gephi/
  comum.py            # leitura dos CSVs, pares distintos, metadados → modalidade/evento, paleta, gravação GEXF
  recortes.py         # monta os 4 recortes (nós/arestas + atributos) e valida contagens do roteiro
  intervencao_wsl.py  # roda no WSL: regenera doses 4/10/20 do sorteio 51011 e exporta só o grupo 0
  modalidade.py       # regra explícita mercado → modalidade/evento (+ "indeterminado"), gera tabela de correspondência
  java/RenderGephi.java  # Gephi Toolkit (jars do Gephi instalado): importa GEXF, layout, Preview, exporta PDF/SVG/PNG/.gephi
  compor.py           # monta painéis A–D, legendas e quadros numéricos em PDF/SVG vetorial final (17,5 cm)
  renderizar.ps1      # orquestra tudo no Windows
tests/test_visualizacao_gephi.py   # asserções das contagens e invariantes do roteiro
figuras/
  gephi_global.{pdf,svg,png}  gephi_nucleo.*  gephi_casos.*  gephi_intervencao.*
  gephi/*.gephi                # projeto Gephi de cada painel (+ prévia nativa e coordenadas)
  recortes/<figura>/{nos.csv,arestas.csv,manifesto.json}
  LEGENDAS.md                 # texto de legenda pronto para o LaTeX (universo, seed, filtros, layout, codificação)
```

Fluxo: Python gera GEXF com atributos e cores → Java/Gephi Toolkit calcula layout e exporta painéis
→ Python compõe a figura final com legenda, quadros e marcações de painel (A, B, C, D).

## 2. Identidade visual (comum às 4 figuras, "nível doutorado")

- Fundo branco, largura final 17,5 cm; fonte sem serifa (Source Sans 3 / Arial), **nunca < 8 pt** impresso.
- Paleta acessível a daltonismo (Okabe–Ito): carteiras **azul #0072B2 (círculo)**, mercados **laranja #E69F00 (quadrado)**,
  foco **vermelho-vinho #A6192E**, parceiros **roxo #7B3294**, mercados compartilhados **verde #009E73**, intervenção **magenta #CC79A7**,
  "indeterminado" cinza #BDBDBD. Mesma paleta em todas as figuras.
- Tamanho = `log10(1 + grau)` declarado na legenda; escala **idêntica** entre painéis comparáveis.
- Arestas não direcionadas, sem setas, finas e transparentes; espessura só onde há peso (Fig. 2B).
- Rótulos apenas em nós de interesse (IDs internos, nunca endereço/identidade civil).
- Texto neutro: "coparticipação", "sobreposição"; nunca "fraude/suspeito".
- Cada figura com bloco de metadados em `LEGENDAS.md` e `manifesto.json`: universo nós/arestas, seed/cobertura,
  critério de seleção, filtros, layout (algoritmo, versão Gephi 0.10.1, parâmetros, seed), significado de cor/tamanho/espessura/rótulo.

## 3. Figuras

### Fig. 1 — `gephi_global.pdf` (rede bipartida completa)
1. Pares distintos carteira–mercado do CSV final; **assert 27.039 + 16.693 = 43.732 nós e 229.072 arestas**, sem duplicar direções do PyG.
2. Estatísticas no Gephi: grau, componentes conexas (registrar nº e tamanho; nada é removido).
3. Layout ForceAtlas2 (Barnes-Hut, LinLog, prevent overlap no final, seed fixa) + Noverlap; parâmetros no manifesto.
4. Arestas com opacidade ~3–5 % e espessura mínima; nós com tamanho log-grau; carteiras círculos, mercados quadrados.
5. Cinco líderes (17762, 18767, 20748, 18753, 17487): anel de destaque + rótulo com linha de chamada fora do "miolo".
6. Painel inset opcional com zoom na vizinhança das líderes (declarado como zoom, não amostra).

**Ressalva (pedida por você):** com 229 mil arestas, a vista tende a virar uma "nuvem" e o PDF vetorial pode ficar
pesado (dezenas de MB) ou lento no visualizador. Estratégia:
- tentar primeiro o PDF 100 % vetorial; se ultrapassar ~25 MB ou ficar ilegível, as **arestas viram camada raster 600 dpi**
  e nós/rótulos/legenda continuam vetoriais — isso será **declarado na legenda**;
- em qualquer caso entrega-se a exportação completa (`.gephi`, GEXF, `nos.csv`, `arestas.csv`) e a legenda diz
  que a figura é contexto e não sustenta conclusões por proximidade visual. Nenhuma amostragem silenciosa.

### Fig. 2 — `gephi_nucleo.pdf` (núcleo das cinco líderes, dois painéis)
- **A (bipartido):** 5 carteiras + mercados comuns às 5 → **assert 149 nós, 720 arestas**. Layout determinístico:
  carteiras ao centro/coluna, mercados em arcos agrupados por modalidade/evento (`modalidade.py`, regra por tags/question
  com tabela `correspondencia_mercado_evento.csv` registrada; sem correspondência → "indeterminado").
- **B (projeção):** só as 5 carteiras, 10 pares; peso = sobreposição com **todas** as participações do dia
  (**assert pesos em [153, 160]**, e explicitamente ≠ 144). Pentágono fixo, espessura ∝ peso, rótulos de ID e peso.
- Legenda: "A e B usam universos diferentes (A: interseção de 144 mercados; B: todas as participações do dia)".

### Fig. 3 — `gephi_casos.pdf` (casos e controles, quatro painéis)
- Para cada foco (17762, 1507, 16179, 53) e seus 3 parceiros do CSV: todos os mercados do foco + arestas dos 4 usuários
  para esses mercados; assert grau do foco = 171/171/41/41 e parceiros iguais aos do roteiro.
- Cores: foco, parceiros, mercados compartilhados (com ≥1 parceiro) vs exclusivos; modalidade como contorno/agrupamento,
  com "indeterminado" mantido.
- Layout: foco no centro, mercados em anel ordenado por modalidade, parceiros em posições fixas iguais nos 4 painéis;
  **mesma escala de tamanho e espessura em todos** (sem normalização por painel).
- Quadro compacto por painel: score do foco, posição contexto→total, e por parceiro sobreposição observada/esperada
  (lidos de `contribuicoes_casos.csv`/`casos_dia_real.csv`, nada digitado à mão). Nota: "grau desenhado do parceiro ≠ grau global".

### Fig. 4 — `gephi_intervencao.pdf` (sorteio 51011, grupo 0, PAREADO)
1. No WSL, regenerar as doses com `experimentos.artigo.pareado.preparar_doses` sobre `artigo-olt_5b_c` (sem treinar nada).
2. Validações antes de desenhar: alvos = 7896, 36905, 10780, 7219, 9621; graus 34, 27, 24, 31, 42 iguais em todas as doses;
   graus de todos os nós preservados (assert do próprio código); trocas aninhadas 4 ⊂ 10 ⊂ 20.
3. Universo visual = união dos mercados vizinhos das 5 carteiras nas 4 condições; **coordenadas fixas** em todos os painéis;
   aresta ausente na condição não é desenhada; mercados da intervenção em magenta; rótulos com grau global.
4. Doadoras fora do grupo não são desenhadas como controles (se aparecerem, cor neutra e nota explícita).
5. Nota na legenda: o recorte pode parecer ganhar arestas, mas o grau global de cada carteira é constante.

## 4. Entrega e verificação

- Por figura: PDF vetorial, SVG editável, PNG 600 dpi (prévia), `.gephi`, `nos.csv`/`arestas.csv`/`manifesto.json`.
- Altura: 1 painel livre; 2 painéis ~8 cm; 4 painéis 7–10 cm (grade 2×2 ou 1×4 conforme legibilidade).
- `tests/test_visualizacao_gephi.py`: contagens (43.732/229.072, 149/720, pesos 153–160, graus da Fig. 3 e 4),
  ausência de setas/arestas duplicadas, fontes ≥ 8 pt no PDF final (checagem via PyMuPDF).
- Inspeção visual de cada PNG antes de entregar; ajuste iterativo de layout/legibilidade.
- Ordem de execução = prioridade do artigo: **Fig. 3 → Fig. 2 → Fig. 4 → Fig. 1** (a global por último, por ser a mais pesada).
- Sem commit/push até você pedir.
