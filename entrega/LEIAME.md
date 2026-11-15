# Tabelas e gráficos do artigo

Dados brutos e tratados: [Zenodo — DOI 10.5281/zenodo.24230223](https://doi.org/10.5281/zenodo.24230223). O download, a conferência dos hashes e os comandos para reprodução estão no [README principal](../README.md#baixar-e-usar-o-pacote-do-zenodo). A [verificação do ZIP publicado](../resultados/zenodo_verificacao.json) confirmou os cinco CSVs e a reprodução dos três tratados a partir dos brutos.

As tabelas e gráficos são gerados pelo Makefile a partir de `resultados/artigo/` e `benchmark/`. A documentação descreve esses artefatos.

A versão oficial está em `graficos_publicacao/`. `make graficos` continua disponível para diagnóstico, mas suas exportações antigas e a comparação visual são locais e ignoradas pelo Git. `make conteudo` gera somente tabelas e gráficos de publicação.

| Pasta | Conteúdo |
|---|---|
| `tabelas/` | T1 a T6: `.csv` (dados), `.tex` (booktabs), `.png` (3×) e `.pdf`; `tabelas.tex` reúne todas |
| `graficos_publicacao/` | **Versão para o artigo** das mesmas dez figuras: tamanho real de impressão, PNG 600 dpi, PDF, SVG e HTML; `graficos.tex`, `validacao.json` (conferência contra as fontes), critérios e limitações descritos neste guia. Gerada por `make graficos-publicacao` |

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

- **Depende da máquina:** só o `make benchmark` (gráfico S4). Tabelas e gráficos reutilizam os CSVs de referência. Fontes, navegador e bibliotecas podem afetar a renderização; novas execuções de treino também podem variar entre ambientes.
- **PNG e PDF:** o kaleido precisa de um Chrome ou Chromium. Depois do `make figuras-ambiente`, rode uma vez `.venv-figuras/bin/python -c "import kaleido; kaleido.get_chrome_sync()"` (no Linux, o Chrome pode pedir bibliotecas do sistema). Sem navegador, use `make conteudo TABELAS_OPCOES=--sem-imagens GRAFICOS_OPCOES=--sem-imagens`.
- **Benchmark padronizado:** feche outros programas, deixe o computador na tomada e use `REPETICOES=3` ou mais. Cada processo usa 2 threads (`OMP_NUM_THREADS=2`). O hardware fica em `benchmark/hardware.json`.

## No LaTeX

```latex
\usepackage{booktabs}
\usepackage{graphicx}
\graphicspath{{entrega/graficos_publicacao/}{entrega/tabelas/}}
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
| T2 | AUROC e AP por seed e média no teste reservado; positivos, negativos e amostragem. | `avaliacao_cobertura.json` |
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

Na primeira repetição do benchmark local, o Ryzen 5 4600G reproduziu as posições das 27.039 carteiras em contexto e cobertura total nas seeds 42, 73 e 2026. Os scores de cobertura total e as métricas AUROC e AP coincidiram com os valores de referência. A conferência por seed está em `resultados/artigo/extra/verificacao.json`. O S4 registra os tempos e a memória das três repetições do protocolo do dia real. O CSV auxiliar de treino por seed registra a segunda repetição; sua coluna de segundos usa o cronômetro interno do treinamento. O preparo temporário das dependências de exportação coincidiu com parte da primeira filtragem. O protocolo registra o estado do Git ao fim da medição, após a escrita de resultados rastreados; o código executado foi o commit `9c1102e`.

## Qualidade e validação das figuras de publicação

As figuras oficiais usam PDF e SVG vetoriais, PNG de 600 dpi, painéis identificados por letras e cores combinadas com traços e marcadores. Foram dimensionadas para impressão: página de 7,0 polegadas, coluna de 3,5 polegadas e ROC/precisão–revocação de 4,6 polegadas. Redimensioná-las no artigo também reduz as fontes; confira a legibilidade no PDF final.

As melhorias de apresentação não alteraram as métricas oficiais. As fontes continuam em `resultados/artigo/` e `benchmark/`. `make graficos-publicacao` verifica a consistência numérica e grava o [relatório de validação](graficos_publicacao/validacao.json); a geração falha quando uma verificação não é satisfeita. As notas subjetivas atribuídas à apresentação antiga não são métricas científicas e não integram os resultados.

### Verificação de integridade

| Figura | Conferência automática | Resultado |
|---|---|---|
| G1 | Médias por ranking contra a T3 (`resumo.csv` e `pareado/resumo.csv`) | diferença máxima 0 p.p.; 168 pontos; 1, 3 ou 9 rankings por célula |
| G2 | Médias dos sorteios contra `pareado/resumo.csv` | diferença 0; 3 sorteios por ponto |
| G3 | AUROC recalculado das curvas contra o oficial | diferença máxima 3,4 × 10⁻⁷; faixa em [0, 1] |
| G4 | 450 posições por caixa; nenhuma omitida | ok |
| G5 | 150 alvos e 150 controles com as 4 intensidades | pares completos |
| G6 | 94 permaneceram, líderes 18753 → 17762 contra `avaliacao_cobertura.json` | ok |
| S1 | Curvas monótonas e terminando em 100% | ok |
| S2 | AP recalculado contra o oficial | diferença < 10⁻⁴ |
| S3 | 300 linhas (3 seeds × 100 épocas), sem suavização | ok |
| S4 | Medianas recalculadas de `etapas.csv` contra `resumo.csv` | ok |

### Transformações aplicadas só à visualização

- **G3, média e faixa:** para cada seed, TPR máximo em cada FPR distinto e interpolação linear numa grade de 1.001 FPR (0 a 1). O DP é amostral (n − 1); a faixa é limitada a [0, 1]. AUROC médio entre seeds = 0,9620 ± 0,0008 (oficial). Área da curva média interpolada = 0,96198, um valor diferente, citado na legenda.
- **G3 e S2, desenho:** vértices a menos de 0,0004 (em unidades de taxa) do último mantido são omitidos só no traço, para o PDF e o SVG não ficarem pesados. AUROC e AP usam todos os 45.814 pares. Os valores completos estão no CSV.
- **G5c:** log10(posição original / posição com 20 mercados), por carteira, derivado das trajetórias.
- **G1, G2 e G4:** deslocamento horizontal entre métodos e jitter dos pontos, para evitar sobreposição. O jitter usa semente fixa.

### Limitações de interpretação

- **G3:** as 3 seeds compartilham divisão, pares de teste e negativos. A faixa mede só a sensibilidade à seed (n = 3), não a incerteza amostral. Por isso a diferença melhor − pior (0,0015) não deve ser lida como diferença entre modelos. O maior DP da TPR é 0,011, perto da origem; no resto da curva a faixa é imperceptível, e isso está dito na legenda em vez de ampliado artificialmente.
- **G3 e S2:** não existe referência de 25 execuções: há 3. Melhor e pior são duas das três curvas.
- **S2:** não há curva média. A média vertical de precisões não corresponde a nenhum AP.
- **Dispersão:** mínimo e máximo entre repetições não são intervalos de confiança. O número de rankings depende do desenho e do método; o S4 utiliza três repetições locais.
- **S3:** a melhor época é a 100ª nas três seeds. O melhor checkpoint atingiu o teto de épocas; a convergência exigiria avaliação adicional.
- **Fonte:** Arial/Liberation Sans, presentes no Windows e no Linux. Para combinar com uma fonte serifada do LaTeX, troque `FONTE` em `graficos_publicacao.py`.
