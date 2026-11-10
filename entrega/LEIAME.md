# Entrega: evidências oficiais do artigo

Data: 10/11/2026. A branch `feat/evidencias-artigo` parte da `main` (`c0ad9ea`) e contém as evidências do método escolhido, o detector relacional direto, e dos seus comparadores.

## Conteúdo

| Item | Arquivo |
|---|---|
| Tabela comparativa (recuperação no top 100, AP, decoys e custo) | `tabela_comparativa.md` e `.csv`; apêndice em `tabela_apendice.csv` |
| Justificativa do método principal e dos comparadores | `justificativa.md` |
| Figura 1: alvos recuperados por cenário | `../figuras/fig1_recuperacao.{pdf,png,html}` |
| Figura 2: decoys em painel próprio | `../figuras/fig2_decoys.{pdf,png,html}` |
| Figura 3: subgrafos de caso e controle | `../figuras/fig3_subgrafos.{pdf,png,html}` |
| Dados desenhados | `../figuras/fig1_fig2_pontos.csv` e `../figuras/fig3_subgrafos_dados.json` |
| Critério de seleção e cortes dos subgrafos | `subgrafos_selecao.json` |
| Benchmark por etapa: hardware, seeds, tempo e pico de RAM | `../benchmark/` (`resumo.csv`, `etapas.csv`, `hardware.json`, `protocolo.json`, `rankings_sha256.json`) |
| Fontes e hashes das métricas e dos custos | `fontes.json` |
| Comprovante da execução em clone novo | `comprovante/` |
| Scripts geradores | `../scripts/evidencias/` (`tabela.py`, `graficos.py`, `subgrafos.py`, `benchmark.py`) e `../tests/test_evidencias.py` |
| Hashes de todos os arquivos desta entrega | `SHA256SUMS` |

## Como a entrega foi produzida

1. Clone novo e completo (não raso) de `https://github.com/Joao-santos-oliveira/betgraph-anomaly`, mais a branch local `feat/evidencias-artigo`. O clone ficou em `~/betgraph-limpo`, no ext4 do WSL2, com apenas os dois CSVs brutos em `data/raw/` (SHA-256 em `comprovante/clone.txt`).
2. Execução do Makefile da branch nesta ordem, com código de saída, tempo e pico de RAM de cada alvo em `comprovante/logs/resumo_alvos.tsv`:

   ```bash
   make ambiente
   make validar
   make testes
   make reproduzir
   make benchmark REPETICOES=3
   make figuras-ambiente
   make evidencias-dados
   make figuras FIGURAS_OPCOES=--sem-imagens
   ```

   Todos terminaram com código 0. O `make reproduzir` conferiu a amostra de 256 carteiras sem divergências e os SHA-256 dos dois rankings do dia completo contra a referência (`comprovante/reproduzir/`).
3. **Exportação das figuras para PDF e PNG.** O kaleido precisa de um navegador Chromium, que o WSL desta máquina não tem: instalá-lo exige `sudo`. Por isso as imagens foram exportadas no Windows, com os mesmos scripts e os mesmos dados (`figuras/*_dados.json` e `*_pontos.csv` gerados no clone limpo), usando o Edge:

   ```bash
   make figuras PYTHON_FIGURAS=<python com requirements-figuras.txt>
   ```

   Com um Chromium disponível no Linux (`plotly_get_chrome`, mais as bibliotecas de sistema), `make evidencias` faz tudo no mesmo ambiente.

**Interrupção registrada.** A primeira tentativa parou durante o `make reproduzir`, porque o disco C: do hospedeiro encheu e o WSL caiu. Depois de liberar espaço, a execução foi retomada a partir de `make testes`, no commit `601bf65`, que só muda quais execuções o benchmark guarda (`comprovante/logs/retomada.txt`). Os alvos `ambiente`, `validar` e `testes` da primeira tentativa continuam válidos e estão no mesmo resumo.

## Benchmark: hardware e protocolo

| Item | Valor |
|---|---|
| CPU | 13th Gen Intel Core i3-1315U, 8 núcleos lógicos |
| Memória | 3,7 GiB no WSL2 (7,7 GiB físicos no hospedeiro), swap de 1 GiB |
| Sistema | WSL2, kernel 6.6.87.2, disco ext4 |
| Software | Python 3.12.3, torch 2.14.1+cpu, torch-geometric 2.8.1, scikit-learn 1.9.1, numpy 2.5.3 |
| Threads | `OMP_NUM_THREADS`, `MKL_NUM_THREADS` e `OPENBLAS_NUM_THREADS` = 2 |
| Recorte | coorte de 5.000 carteiras, divisão 9103, cenário original |
| Seeds | redes 42, 73 e 2026; IsolationForest 42 |
| Épocas | GNN v3 e MLP: 60; híbrido GNN v2: 100 (os padrões de cada branch) |
| Repetições | 3, alternando a ordem das branches; valores são medianas |
| Medição | `os.wait4` por etapa (tempo de parede e ru_maxrss do processo); o tempo do método vem de `custos.json`, sem a carga do interpretador |
| Determinismo | rankings idênticos entre as 3 repetições nas duas branches |

## Limites

- As métricas de recuperação, AP e decoys vêm da rodada histórica da v3 (`resultados/v3/metricas.csv`, commit `a67bcb5`). O benchmark mede só o custo, no cenário original.
- Tempos absolutos dependem da máquina. A comparação vale entre métodos medidos aqui, nas mesmas condições.
- Os subgrafos descrevem coparticipação. Não são rótulos de fraude.
