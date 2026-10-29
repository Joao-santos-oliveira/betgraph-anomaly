# Interpretação da rodada GNN v3

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
