# Resultado — GNN v3, rodada única

Decisão pelo protocolo: **negativo**. Utilidade adicional demonstrada: **não**.

As métricas abaixo medem recuperação de intervenções artificiais, não fraude. S2 mede priorização indesejada de decoys. Seeds dos encoders: 42, 73, 2026; realizações 30101/30201. IF seed 42. Nenhum ajuste após resultados.

## Leitura dos resultados e recomendação

A tentativa foi concluída: **81 treinos, oito intervenções viáveis e 1.008 linhas de métricas**. Os arquivos de origem são `custos_treinos.csv`, `data/processed/gnn_v3_v1/casos.json` e `metricas.csv`. Não houve substituição de seeds ou ajuste após resultados.

No cenário principal S1, C não colocou nenhum dos 50 alvos entre os 100 primeiros em nenhuma das seis combinações. Sua AP média foi 0,0300, abaixo de A (0,0343) e B (0,0454). O ganho incremental de C fora da união A+B também foi zero. Portanto, **não há evidência para adotar esta GNN no pipeline principal**. Recomenda-se mantê-la como ablação experimental documentada.

O fato de C ter AUC média 0,8496 em S1 não contradiz o fracasso no topo: AUC resume a ordenação global entre alvos e demais carteiras, enquanto top100 mede o conjunto pequeno que seria investigado primeiro. Uma AUC alta, isoladamente, não demonstra utilidade operacional. Fonte: `metricas.csv`, universo completo, fase injetado.

S2 trouxe um limite relevante para os métodos anteriores: A destacou, em média, 29,5 decoys, e E_v2, 29,0, no top100. Isso mostra que forte coparticipação pode ser priorizada mesmo quando o cenário foi construído com lados balanceados e horários dispersos. C e B destacaram zero, mas também recuperaram zero em S1 no universo completo; essa ausência de destaque **não demonstra, sozinha, distinção útil entre os dois padrões**. Esses decoys são controles artificiais, não casos reais comprovadamente normais.

Nas regressões, A recuperou 32/50 alvos em média na estrutura, versus 31,83 para E_v2 e zero para C. Em volume, features+IF recuperou 9/50, contra 3,83 para C_IF e zero para C. C_IF permaneceu análise secundária e não foi escolhido para substituir o score principal após esses resultados. Fontes: `resumo_metricas.csv` e `decisao.json`.

### O universo de avaliação faz diferença

Em S1, o top100 **dentro das carteiras elegíveis** recuperou, em média, 5,50 alvos com A, 11,50 com B e 6,83 com C. O universo elegível teve 752 e 750 carteiras nas realizações 30101 e 30201, respectivamente, contra 5.000 no completo. Assim, o corte 100 é proporcionalmente muito maior; esses números não são diretamente comparáveis ao top100 global. C permanece atrás de B, e sua AP média nesse universo foi 0,075, versus 0,141 de B. Retirar doadoras sem restringir aos elegíveis manteve C em zero no top100 de S1. Fontes: `metricas.csv` e `resumo_por_realizacao.csv`.

Em S2, C destacou 9,33 decoys em média no top100 elegível. Portanto, a aprovação do controle S2 na regra congelada vale especificamente para o universo completo, não para qualquer filtro de candidatos. Não mudamos o universo principal após observar essa diferença.

### O que a experiência permite concluir

A combinação definida não demonstrou recuperação forte no topo, inclusive para B. Em S1, o canal de valores isolado V teve AP média 0,079, maior que B e C, mas também zero no top100 completo. Isso limita a premissa de que combinar esses sinais necessariamente produziria um padrão fortemente separável. Não concluímos que toda GNN seja incapaz de identificar comportamento incomum.

A consolidação dos outcomes e a escolha do lado local 0 em S1 criam pistas artificiais adicionais, conforme previsto no protocolo. Os resultados não isolam uma causa única para o desempenho. Além disso, todos os 27 treinos de C e os 27 de D selecionaram a época 60, o teto fixado. A rodada não demonstra convergência até um ótimo; esse teto foi respeitado e não ampliado após métricas. Fonte: `custos_treinos.csv`.

### Encerramento e próximos passos propostos

Encerrar esta tentativa sem nova busca de hiperparâmetros. Preservar resultados negativos e usar os canais relacional e de atributos como referências, reconhecendo que o relacional não distingue sozinho os decoys aqui definidos. Para o projeto, os próximos trabalhos seriam avaliação humana dos casos reais, obtenção de metadados de evento/modalidade e outros dias, e validação em escala maior com protocolo próprio. O nulo estratificado foi apenas proposto em `agrupamento_mercados.md`; não foi executado. O teste final de ligações e as realizações 30301/30401 continuam reservados.

## Auditoria final e revisão técnica

- **40 testes aprovados**, incluindo oito testes novos: saídas em `testes_regressao.txt` e `testes.txt`.
- **81 checkpoints e scores reproduzidos**, com métricas reproduzidas e exemplos de validação idênticos entre modelos/seeds: `auditoria.json`.
- **133 arquivos anteriores, cinco entradas e 11 arquivos de código/protocolo** com hashes conferidos: `congelamento.json` e `auditoria.json`.
- **Oito intervenções reconferidas nos artefatos efetivamente usados**: graus preservados, pares reservados intactos, catálogo coerente e tickets alterados consistentes. Ver `integridade_injecoes.json`.
- O campo `checks` dos manifestos de injeção vem da etapa inicial de trocas do gerador v2; `attributes_equal_by_row` refere-se a essa etapa. S1/S2/volume alteram atributos explicitamente depois; as verificações finais estão em `integridade` e `integridade_injecoes.json`, e o antes/depois em `atributos_alterados.csv`.
- Houve uma correção técnica **antes de treino experimental e consulta de métricas de recuperação**: aceitar formatos mistos de horário. O protocolo 1, seu congelamento e a preparação parcial foram preservados; protocolo 1.1 e novo congelamento foram registrados. Não mudou o desenho experimental nem houve segunda rodada de ajuste. Ver `protocolo_v1.md`, `congelamento_v1.json` e a seção de revisão em `protocolo.md`.
- Nos CSVs, `seed_modelo=0` significa método sem encoder; o IsolationForest usa seed 42, conforme protocolo.

As saídas grandes e logs estão em `data/processed/gnn_v3_v1/`, ignorado pelo Git. `src/`, arquivos anteriores e main não foram alterados. Não foi feito commit nem push.

## Todos os métodos e cenários

Universo completo, após injeção. Média [mínimo–máximo] sobre realizações e seeds disponíveis; baselines determinísticos têm duas observações, encoders até seis. Top100 conta alvos/decoys de 50. AP e AUC em escala 0–1. Valores completos antes/depois e universos elegível/sem doadoras estão em metricas.csv; resumo_metricas.csv contém média/mín/máx para todas as métricas; resumo_por_realizacao.csv resume as três seeds separadamente por realização.

| Cenário | Método | n | top50 | top100 | recall100 | precision50 | AP | AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| S1 | A | 2 | 0.500 [0.000–1.000] | 0.500 [0.000–1.000] | 0.010 [0.000–0.020] | 0.010 [0.000–0.020] | 0.034 [0.034–0.035] | 0.866 [0.864–0.868] |
| S1 | B | 2 | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.045 [0.045–0.046] | 0.901 [0.898–0.903] |
| S1 | O | 2 | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.034 [0.034–0.035] | 0.867 [0.865–0.870] |
| S1 | T | 2 | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.041 [0.040–0.041] | 0.888 [0.886–0.890] |
| S1 | V | 2 | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.079 [0.077–0.082] | 0.943 [0.941–0.945] |
| S1 | C | 6 | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.030 [0.028–0.033] | 0.850 [0.838–0.863] |
| S1 | C_IF | 6 | 0.167 [0.000–1.000] | 0.167 [0.000–1.000] | 0.003 [0.000–0.020] | 0.003 [0.000–0.020] | 0.011 [0.010–0.012] | 0.566 [0.549–0.587] |
| S1 | D | 6 | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.034 [0.031–0.036] | 0.867 [0.854–0.875] |
| S1 | D_IF | 6 | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.027 [0.020–0.032] | 0.815 [0.759–0.852] |
| S1 | E_v2 | 6 | 0.500 [0.000–1.000] | 0.500 [0.000–1.000] | 0.010 [0.000–0.020] | 0.010 [0.000–0.020] | 0.035 [0.034–0.035] | 0.869 [0.867–0.871] |
| S1 | E_features | 2 | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.013 [0.012–0.014] | 0.609 [0.579–0.639] |
| S2 | A | 2 | 13.500 [13.000–14.000] | 29.500 [25.000–34.000] | 0.590 [0.500–0.680] | 0.270 [0.260–0.280] | 0.231 [0.205–0.258] | 0.977 [0.975–0.978] |
| S2 | B | 2 | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.050 [0.050–0.050] | 0.911 [0.911–0.911] |
| S2 | O | 2 | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.051 [0.050–0.052] | 0.910 [0.908–0.912] |
| S2 | T | 2 | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.021 [0.021–0.022] | 0.785 [0.783–0.786] |
| S2 | V | 2 | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.063 [0.059–0.067] | 0.929 [0.925–0.932] |
| S2 | C | 6 | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.036 [0.032–0.039] | 0.872 [0.856–0.883] |
| S2 | C_IF | 6 | 0.000 [0.000–0.000] | 0.167 [0.000–1.000] | 0.003 [0.000–0.020] | 0.000 [0.000–0.000] | 0.014 [0.009–0.021] | 0.623 [0.486–0.738] |
| S2 | D | 6 | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.040 [0.035–0.043] | 0.885 [0.870–0.895] |
| S2 | D_IF | 6 | 0.000 [0.000–0.000] | 0.500 [0.000–1.000] | 0.010 [0.000–0.020] | 0.000 [0.000–0.000] | 0.032 [0.022–0.044] | 0.831 [0.761–0.888] |
| S2 | E_v2 | 6 | 13.667 [13.000–14.000] | 29.000 [24.000–34.000] | 0.580 [0.480–0.680] | 0.273 [0.260–0.280] | 0.227 [0.202–0.252] | 0.976 [0.975–0.977] |
| S2 | E_features | 2 | 0.500 [0.000–1.000] | 2.000 [1.000–3.000] | 0.040 [0.020–0.060] | 0.010 [0.000–0.020] | 0.015 [0.014–0.016] | 0.625 [0.599–0.650] |
| S3_estrutura | A | 2 | 12.500 [11.000–14.000] | 32.000 [30.000–34.000] | 0.640 [0.600–0.680] | 0.250 [0.220–0.280] | 0.232 [0.220–0.243] | 0.979 [0.978–0.979] |
| S3_estrutura | B | 2 | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.083 [0.079–0.087] | 0.947 [0.944–0.950] |
| S3_estrutura | O | 2 | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.059 [0.058–0.061] | 0.924 [0.922–0.925] |
| S3_estrutura | T | 2 | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.054 [0.048–0.060] | 0.916 [0.906–0.926] |
| S3_estrutura | V | 2 | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.071 [0.070–0.071] | 0.937 [0.936–0.937] |
| S3_estrutura | C | 6 | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.046 [0.040–0.054] | 0.903 [0.888–0.917] |
| S3_estrutura | C_IF | 6 | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.013 [0.009–0.018] | 0.601 [0.490–0.721] |
| S3_estrutura | D | 6 | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.052 [0.045–0.059] | 0.914 [0.901–0.925] |
| S3_estrutura | D_IF | 6 | 0.000 [0.000–0.000] | 0.333 [0.000–1.000] | 0.007 [0.000–0.020] | 0.000 [0.000–0.000] | 0.046 [0.039–0.056] | 0.889 [0.865–0.917] |
| S3_estrutura | E_v2 | 6 | 13.333 [12.000–14.000] | 31.833 [29.000–34.000] | 0.637 [0.580–0.680] | 0.267 [0.240–0.280] | 0.232 [0.220–0.246] | 0.979 [0.978–0.980] |
| S3_estrutura | E_features | 2 | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.013 [0.012–0.014] | 0.618 [0.601–0.635] |
| S3_volume | A | 2 | 0.000 [0.000–0.000] | 0.500 [0.000–1.000] | 0.010 [0.000–0.020] | 0.000 [0.000–0.000] | 0.023 [0.022–0.024] | 0.783 [0.779–0.787] |
| S3_volume | B | 2 | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.021 [0.021–0.021] | 0.775 [0.775–0.775] |
| S3_volume | O | 2 | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.021 [0.021–0.021] | 0.773 [0.766–0.779] |
| S3_volume | T | 2 | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.021 [0.021–0.022] | 0.778 [0.778–0.779] |
| S3_volume | V | 2 | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.020 [0.020–0.020] | 0.768 [0.767–0.770] |
| S3_volume | C | 6 | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.025 [0.023–0.027] | 0.801 [0.788–0.817] |
| S3_volume | C_IF | 6 | 2.000 [0.000–4.000] | 3.833 [1.000–6.000] | 0.077 [0.020–0.120] | 0.040 [0.000–0.080] | 0.041 [0.014–0.070] | 0.685 [0.622–0.737] |
| S3_volume | D | 6 | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.026 [0.024–0.029] | 0.811 [0.794–0.825] |
| S3_volume | D_IF | 6 | 1.500 [0.000–3.000] | 3.833 [0.000–9.000] | 0.077 [0.000–0.180] | 0.030 [0.000–0.060] | 0.055 [0.029–0.091] | 0.858 [0.823–0.893] |
| S3_volume | E_v2 | 6 | 0.000 [0.000–0.000] | 0.500 [0.000–1.000] | 0.010 [0.000–0.020] | 0.000 [0.000–0.000] | 0.023 [0.022–0.023] | 0.783 [0.780–0.786] |
| S3_volume | E_features | 2 | 4.000 [0.000–8.000] | 9.000 [6.000–12.000] | 0.180 [0.120–0.240] | 0.080 [0.000–0.160] | 0.074 [0.031–0.118] | 0.747 [0.733–0.761] |

A/B são os comparadores primários; O/T/V são canais isolados pré-especificados. C usa similaridade aprendida; C_IF é secundário e não substitui C na decisão. D recebe os mesmos agregados sem mensagens. E_v2 é a referência contrastiva híbrida anterior; E_features é features+IsolationForest.

## Verificação da regra

| Condição | Passou | Falhou | Não concluída |
|---|---:|---:|---:|
| S1_supera_A_B | 0 | 6 | 0 |
| S2_zero_decoys_top100 | 6 | 0 | 0 |
| S3_estrutura_sem_piora_relevante | 0 | 6 | 0 |
| S3_volume_sem_piora_relevante | 0 | 6 | 0 |

A regra exige superar A e B em recall100 e AP em todas as seis combinações de S1, zero decoys no top100 de S2 e limites de regressão em S3. Consulte protocolo.md e decisao.json para o limiar e cada combinação.

## Ganho incremental e overlap

| Cenário | Método | Novos fora A+B (média/mín/máx) | Overlap A (média) | Overlap B (média) |
|---|---|---:|---:|---:|
| S1 | C | 0.00/0/0 | 16.00 | 78.67 |
| S1 | C_IF | 0.17/0/1 | 6.83 | 0.00 |
| S2 | C | 0.00/0/0 | 2.00 | 79.17 |
| S2 | C_IF | 0.00/0/0 | 7.50 | 0.00 |
| S3_estrutura | C | 0.00/0/0 | 2.00 | 79.50 |
| S3_estrutura | C_IF | 0.00/0/0 | 7.50 | 0.00 |
| S3_volume | C | 0.00/0/0 | 6.67 | 75.50 |
| S3_volume | C_IF | 3.83/1/6 | 7.17 | 0.00 |

Ganhos separados contra A/B, por seed e universo, estão em incremental.csv. Novos alvos podem coexistir com perda de outros; isso não prova superioridade global.

## Execução e custos

- C: 27 treinos, parâmetros 33890–33890; soma dos tempos de treino 1840.5s; média 68.2s. Inclui original e intervenções; até três processos CPU em paralelo. IF e scores adicionais têm custos separados nos artefatos.
- D: 27 treinos, parâmetros 6210–6210; soma dos tempos de treino 276.5s; média 10.2s. Inclui original e intervenções; até três processos CPU em paralelo. IF e scores adicionais têm custos separados nos artefatos.
- E_v2: 27 treinos, parâmetros 22723–22723; soma dos tempos de treino 1284.1s; média 47.6s. Inclui original e intervenções; até três processos CPU em paralelo. IF e scores adicionais têm custos separados nos artefatos.

Casos/treinos não concluídos: 0. Detalhes: casos.json e treinos.json nos artefatos; não houve substituição de seeds.

## Limites e interpretação

- Um único dia, 5.000 carteiras. Novas realizações de injeção não são novos dias nem validação externa.
- Atributos descrevem arestas agregadas: primeiro horário e volume total, não sequência de apostas.
- Dois lados locais por mercado. O nome de uma equipe não estabelece correspondência entre mercados.
- S2 balanceia lados; não torna 50 carteiras mutuamente opostas. A regra de zero decoys é deliberadamente estrita, definida antes dos resultados.
- Seleção de checkpoint auto-supervisionada transdutiva; loss de ligação/auxiliar não é qualidade do ranking. O auxiliar de validação permanece no contexto de mensagens.
- Semelhança alta entre embeddings pode refletir comportamento comum. O score principal foi fixado antes dos resultados e não foi trocado pelo IF.
- A hipótese de sinais individualmente fracos deve ser confrontada com O/T/V; canais isolados fortes enfraquecem essa premissa.
- Resultado negativo rejeita utilidade desta configuração sob este protocolo; não prova impossibilidade de toda GNN.
- Teste final de ligações e realizações 30301/30401 permanecem reservados. Agrupamento e nulo estratificado foram apenas propostos.

## Reprodução e arquivos de origem

Na raiz do repositório, com a .venv e os mesmos arquivos de entrada:

```bash
.venv/bin/python -B -m unittest discover -s tests -p "test_gnn_v3.py" -v
.venv/bin/python -B -m experimentos.gnn_v3.executar congelar
.venv/bin/python -B -m experimentos.gnn_v3.executar preparar
.venv/bin/python -B -m experimentos.gnn_v3.executar treinar
.venv/bin/python -B -m experimentos.gnn_v3.executar avaliar
.venv/bin/python -B -m experimentos.gnn_v3.executar auditar
```

Congelar/preparar recusam sobrescrever uma rodada existente. Para reprodução integral, usar outra cópia limpa da base c0b7390 com as mesmas dependências e entradas. Copiar experimentos/gnn_v3/, tests/test_gnn_v3.py, protocolo.md, protocolo_v1.md, congelamento_v1.json e agrupamento_mercados.md para seus caminhos correspondentes. Não copiar o congelamento.json atual nem a pasta de saídas para essa cópia: o comando congelar criará novos registros. Nunca remover a rodada original para substituí-la. Avaliar/auditar apenas recalculam a partir dos artefatos existentes e regeneram a tabela em resultado.md. A interpretação editorial também está preservada separadamente em interpretacao.md.

- Entradas: data/processed/validacao_ampliada_v1/novo/{user,event,edge}.csv e divisao.pt.
- Execução: data/processed/gnn_v3_v1/<cenário>_<realização>/; original/ para comparações antes da injeção.
- Scores: ranking_A/B/O/T/V/E_features.csv e C/D/E_v2_<seed>/ranking.csv; variantes IF em ranking_IF.csv.
- Custos: manifest.json de cada treino, custo_IF.json e custos_baselines.json; resumo custos_treinos.csv.
- Decisão/métricas: metricas.csv, resumo_metricas.csv, incremental.csv, decisao.json nesta pasta.
- Congelamento: congelamento.json registra SHA-256 das entradas, código novo, protocolo e arquivos anteriores. Auditoria final: auditoria.json. Saída de testes: testes.txt.
- Implementação da convolução: documentação oficial https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.TransformerConv.html.

Nenhum commit ou push foi realizado nesta tarefa.
