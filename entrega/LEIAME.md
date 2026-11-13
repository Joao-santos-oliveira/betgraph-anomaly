# Tabelas e gráficos do artigo

Tudo nesta pasta é gerado pelo Makefile a partir de `resultados/artigo/`. Nenhum número é digitado à mão.

| Pasta | Conteúdo |
|---|---|
| `tabelas/` | T1 a T6: `.csv` (dados), `.tex` (booktabs), `.png` (3×) e `.pdf`; `tabelas.tex` reúne todas |
| `graficos/` | G1 a G6 e S1 a S4: `.pdf` (vetorial), `.png` (3×), `.html` (interativo) e `.csv` (valores desenhados, por sorteio e seed); `graficos.tex` traz as figuras com legenda |
| `graficos_publicacao/` | **Versão para o artigo** das mesmas dez figuras: tamanho real de impressão, PNG 600 dpi, PDF, SVG e HTML; `graficos.tex`, `validacao.json` (conferência contra as fontes), `comparacao.html` (antes e depois) e `AUDITORIA.md` (diagnóstico, notas e limitações). Gerada por `make graficos-publicacao` |

## Como reproduzir

Linux ou WSL, Python 3.12 e GNU Make, com os dois CSVs brutos em `data/raw/`.

```bash
make ambiente            # venv do estudo (.venv)
make figuras-ambiente    # venv de tabelas e gráficos (.venv-figuras): plotly e kaleido
make artigo-original     # treina o dia real (3 seeds); para o estudo inteiro, make artigo
make dados-artigo RODADA=data/execucoes/artigo-XXXX   # extrai os dados pequenos para resultados/artigo/extra/
make benchmark REPETICOES=3                           # tempo e memória por etapa: o único passo que depende da máquina
make conteudo            # tabelas + gráficos
```

- **Depende da máquina:** só o `make benchmark` (gráfico S4). Tabelas e gráficos leem CSVs e saem iguais em qualquer computador; CSV e LaTeX são idênticos byte a byte no Windows e no Linux.
- **PNG e PDF:** o kaleido precisa de um Chrome ou Chromium. Depois do `make figuras-ambiente`, rode uma vez `.venv-figuras/bin/python -c "import kaleido; kaleido.get_chrome_sync()"` (no Linux, o Chrome pode pedir bibliotecas do sistema). Sem navegador, use `make conteudo TABELAS_OPCOES=--sem-imagens GRAFICOS_OPCOES=--sem-imagens`.
- **Benchmark padronizado:** feche outros programas, deixe o computador na tomada e use `REPETICOES=3` ou mais. Cada processo usa 2 threads (`OMP_NUM_THREADS=2`). O hardware fica em `benchmark/hardware.json`.

## No LaTeX

```latex
\usepackage{booktabs}
\usepackage{graphicx}
\graphicspath{{entrega/graficos/}{entrega/tabelas/}}
...
\input{entrega/tabelas/T3_recuperacao_alvos.tex}
\begin{figure*}[htbp]\centering
  \includegraphics[width=\textwidth]{G1_recuperacao_intensidade.pdf}
  \caption{...}
\end{figure*}
```

Cada `.tex` de tabela já traz `\caption`, `\label{tab:tN}` e a nota. Tabelas largas usam `table*` com `\resizebox`. O `graficos.tex` traz uma sugestão de `figure` com legenda e `\label{fig:gN}` para cada gráfico.

## Tabelas

| | Conteúdo | Fonte |
|---|---|---|
| T1 | Dados do dia e divisão das ligações (60/20/10/10%), cobertura, seeds e sorteios | `extra/divisao.json` |
| T2 | AUROC e AP por seed e média no teste reservado; positivos, negativos e amostragem. O MLP não entra (não foi avaliado nos mesmos pares) | `avaliacao_cobertura.json` |
| T3 | Recuperação dos alvos: misto e pareado × 4, 10, 20 mercados × 3 métodos × top 100 e 541, com contagem, denominador, % e regra de agregação | `resumo.csv`, `pareado/resumo.csv` |
| T4 | GNN híbrida antes e depois da intervenção, alvos e controles (teste misto); "antes" = quantos já estavam no corte | `resumo.csv` |
| T5 | Contexto versus cobertura total por seed: permanências, entradas, saídas, liderança, Spearman e posições dos casos | `cobertura_total.csv` |
| T6 | Fichas dos casos e controles (seed 42): score, três maiores contribuições, parceiros, sobreposição observada e esperada, eventos | `extra/casos_dia_real.csv` |

**Regra de eventos (T6).** Os dados não trazem o identificador do evento. Um mercado tem evento identificado quando a pergunta nomeia um confronto ("A vs. B" ou placar exato); a chave é o par de nomes. Os demais (um só lado, eleições, preços) contam como evento indeterminado. Implementação e testes: `experimentos/artigo/extrair.py` e `tests/test_conteudo.py`.

## Gráficos

| | Conteúdo |
|---|---|
| G1 | **Figura principal.** Recuperação por intensidade: misto e pareado (colunas), top 100 e 541 (linhas), 3 métodos com cor e marcador; pontos claros = cada ranking |
| G2 | Alvos versus controles no pareado (GNN híbrida), original, 4, 10 e 20; barras = menor e maior dos 3 sorteios |
| G3 | Curva ROC das ligações reservadas por seed, com zoom até 10% de falsos positivos |
| G4 | Comparação entre as GNNs: híbrida versus similaridade isolada (mesmos embeddings). Posição dos alvos e sobreposição com o relacional no dia real |
| G5 | Trajetória de cada alvo e controle pelas intensidades (seed 42), com a mediana |
| G6 | Entradas e saídas do top 100 do dia real (contexto → cobertura total), com os casos da T6 |
| S1 | Recuperação acumulada pela profundidade do ranking (10 e 20 mercados) |
| S2 | Curva precisão–revocação das ligações por seed |
| S3 | Perda e AUROC de validação por época |
| S4 | Tempo e pico de memória por etapa, com o hardware |

Cores: slots 1 a 3 da paleta categórica de referência (azul, laranja e verde-água), validados para daltonismo em todos os pares; cada método também tem marcador próprio.

## Reprodução conferida

`resultados/artigo/extra/verificacao.json` compara o dia real re-treinado com o publicado: top 100, top 541, líderes e posições dos casos idênticos nas três seeds; AUROC com diferença abaixo de 10⁻⁶. Cerca de 1.600 posições mudam na cauda do ranking, entre scores quase empatados (ponto flutuante).
