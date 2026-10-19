# Retomada do projeto — 18/10/2026

Este documento prepara a continuidade; não inicia nem declara concluídos os experimentos abaixo. Não fazer commit automaticamente: o usuário fará isso posteriormente. Explicar objetivo, funções e resultados de cada etapa em linguagem simples.

## Ponto de partida

- Rodada atual: `data/processed/dev/experimento_seeds_42_73_2026/`; seeds do encoder 42, 73 e 2026. Rodada anterior preservada.
- Comparação GNN/MLP, duas condições, 12 treinos e 14 rankings concluídos. Oito testes passaram na última execução. ROC, curvas de perda e ROC média disponíveis em `curvas/`.
- GNN full_day: AUC média 0,94605; AP 0,94468. GNN context_only: AUC 0,93721; AP 0,93867. São métricas de ligação, não de fraude.
- Ambas passam mensagens no mesmo contexto. full_day usa agregados do dia inteiro; context_only restringe os agregados das carteiras ao contexto. Metadados de mercado continuam transdutivos; não alegar previsão temporal.
- Amostra: 5.000 carteiras, 9.189 mercados, 62.328 linhas agregadas e 39.562 pares únicos. Contexto: 23.738 pares; 975 carteiras isoladas preservadas.
- Completo tratado: 27.039 carteiras + 16.693 mercados = 43.732 nós; 358.689 linhas agregadas, 229.072 pares únicos. Antes do filtro: 22.418 mercados e 560.985 linhas. Carteiras não equivalem necessariamente a pessoas.
- Visualização didática já mostrada para U65: três mercados de contexto. Dados e regra de seleção em `data/processed/dev/visualizacao_grafo/subgrafo.json`. Não foi inspeção de suspeitos nem validação de anomalias.
- Nenhuma injeção artificial executada. Teste final de conexões não avaliado. Conjunto completo ainda não treinado.

## 1. Próxima sessão: fechar e implementar o piloto sintético

Objetivo: avaliar se os rankings recuperam um padrão artificial previamente definido, comparando GNN + IsolationForest, MLP + IsolationForest e IsolationForest em features.

Proposta conversada, ainda não protocolo fechado: começar com 50 carteiras (1% da amostra) e um padrão de coparticipação concentrada. Alterar conexões numa cópia dos dados, tentando preservar graus por trocas de arestas. Não usar apenas volume extremo como anomalia fácil. A condição context_only será o controle principal proposto, com full_day identificado separadamente se incluído.

Antes de executar, registrar em um protocolo:

- Seleção das carteiras, mercados-alvo, intensidade da alteração e seed de injeção independente das seeds dos encoders e do detector.
- Algoritmo de troca, limites de tentativas, preservação de graus, rejeição de duplicatas, proibição de inserir pares positivos reservados e tratamento de carteiras isoladas. Se não houver trocas suficientes, reportar inviabilidade, não relaxar silenciosamente o cenário.
- Se a intervenção será apenas no contexto fixo ou em uma cópia do grafo antes de uma divisão própria. A escolha define a pergunta experimental e precisa preceder a execução. Não misturar protocolos nem abrir o teste original.
- Tratamento coerente dos atributos e outcomes das arestas alteradas. Recalcular as features permitidas e salvar transformadores; nenhuma flag de injeção entra no modelo.
- Carteiras-alvo e todos os nós/arestas colateralmente alterados devem ser registrados. Não chamar os colateralmente afetados de controles intactos; explicitar universo e rótulos de cada métrica.
- Forma de ajuste: retreinar cada método na cópia alterada ou aplicar modelos congelados. São perguntas distintas. Proposta inicial: retreinar de modo não supervisionado no cenário alterado, comparando também o ranking anterior; rótulos de injeção usados somente na avaliação.
- Novas realizações sintéticas reservadas para avaliação após desenvolvimento; não selecionar apenas seeds ou cenários favoráveis.

Entregáveis propostos: gerador reproduzível, manifesto de mudanças, testes de integridade, tabela de recuperação top-50/top-100, AP, ROC-AUC e mudança de posição antes/depois. Comparações na mesma realização sintética; detector com seed fixa inicialmente. Rótulo positivo significa intervenção artificial, não fraude. O restante do conjunto pode conter anomalias reais desconhecidas.

Critério para ampliar: injeção coerente, rastreável e sem rótulos nas entradas; execução de todos os controles; resultados favoráveis ou desfavoráveis reportados. Depois variar intensidade, seeds de injeção e proporções propostas de 1%, 3% e 5%, sem tratar esses valores como taxa real de fraude.

## 2. Inspeção dos casos reais e controles

Usar os dados originais, sem injeção. Antes de abrir casos, congelar seleção e formulário:

- Proposta: união dos 20 primeiros de cada método, controles com volume/grau semelhantes e pequena amostra aleatória. Fixar condição, regra entre seeds, pareamento, quantidade e seeds antes da seleção. Não escolher a seed com casos mais interessantes.
- Ficha por carteira: atividade, mercados, popularidade dos mercados, coparticipação, posição/estabilidade nos rankings e subgrafo.
- Registrar limites do subgrafo e omissões; distinguir arestas de contexto de outras conexões caso sejam mostradas. Distância no desenho não é distância no embedding nem evidência de coordenação.
- Se possível, dois revisores avaliam sem método/score visível inicialmente. Registrar padrão observado, explicações alternativas e evidência insuficiente; depois comparar avaliações.

Entregável: fichas de casos e controles, imagens e síntese incluindo exemplos sem interpretação convincente. Não atribuir rótulo de fraude sem evidência independente.

## 3. Controles adicionais

Ablação do contexto com vizinhos aleatorizados preservando graus, sem introduzir positivos reservados; documentar algoritmo e inviabilidade se ocorrer. Medir sensibilidade do IsolationForest separadamente da variação do encoder. Considerar comparação com capacidade mais próxima entre encoders, pois GNN e MLP atuais têm números diferentes de parâmetros. Optuna não é prioridade.

## 4. Conjunto completo

Após a primeira avaliação de qualidade na amostra, preparar em pasta nova com sample-size 0; construir grafo, divisão e features permitidas próprios. Um benchmark técnico curto pode anteceder essa avaliação sem substituir a análise de qualidade.

Medir tempo por época e pico de RAM antes de executar a bateria completa com seeds 42, 73 e 2026. As estimativas anteriores (30–90 s por treino GNN; 10–25 min para a rodada) NÃO foram medidas no completo. Não reutilizar inadvertidamente scalers da amostra nem sobrescrever seus artefatos.

Repetir controles e análise de estabilidade no universo maior. O completo ainda é um dia tratado, não toda a plataforma; novos dias exigem protocolo próprio.

## 5. Fechamento experimental e reprodução

- Congelar arquitetura, protocolo e escolhas usando desenvolvimento/validação; só então avaliar o teste reservado de conexões. Não confundir esse teste com avaliação sintética de anomalias.
- Consolidar comandos em Makefile, dependências, manifestos e documentação. Reproduzir no computador do colega; data/ e modelos ignorados pelo Git precisam ser compartilhados separadamente ou regenerados.
- Recuperar consultas e proveniência da extração; auditar disponibilidade temporal de metadados antes de alegações prospectivas.
- Escrever resultados e limitações, confrontar literatura, revisar com orientador e adequar à chamada escolhida. Não prometer aprovação.

## Leituras para retomar

- `docs/plano_pesquisa.md`: perguntas científicas e plano geral.
- `docs/resultados_seeds_42_73_2026.md`: resultados atuais e limites.
- `src/experimento_gnn.py`, `src/features_contexto.py`, `src/ranking_gnn.py`: execução, controle de informação e ranking.
- `README.md`: comandos existentes. Os scripts do piloto sintético e da revisão humana ainda não foram implementados.
