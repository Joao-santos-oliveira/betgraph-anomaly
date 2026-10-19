# Plano de pesquisa e submissão — Betgraph Anomaly

Versão inicial: 17/10/2026. Atualização: primeira rodada controlada concluída; ver [resultados](resultados_gnn_v1.md). Documento de planejamento, não declaração de resultados nem garantia de aprovação. O desenvolvimento será feito em etapas pequenas, explicadas e verificáveis. Alterações de protocolo devem ser registradas antes da execução correspondente, sem apagar resultados anteriores.

## 1. Pergunta e contribuição pretendida

Pergunta central: em que medida a informação relacional do grafo bipartido carteira–mercado acrescenta evidência útil à priorização de comportamentos incomuns, comparada a características individuais?

Contribuição pretendida: estudo experimental reproduzível de representações individuais e relacionais em negociações do Polymarket, com controles, análise de estabilidade e caracterização dos casos priorizados. Não alegamos novo algoritmo, primeira aplicação de grafos ao domínio, detecção confirmada de fraude ou superioridade da GNN.

Hipóteses de trabalho (ainda não demonstradas):

- H1: a estrutura real das conexões altera os resultados além do efeito das features, atividade das carteiras e popularidade dos mercados.
- H2: padrões dos casos priorizados são estáveis sob diferentes inicializações e não apenas artefatos de uma execução.
- H3: parte dos casos acrescentados pela representação relacional apresenta características de coparticipação não destacadas pelo baseline individual.

Resultados nulos ou contrários devem ser reportados. Uma diferença de ranking não demonstra qualidade superior. Estabilidade não demonstra fraude. A hipótese não será redefinida apenas para acomodar resultados favoráveis.

## 2. Estado e limites atuais

Já existem: preparação, HeteroData, baseline IsolationForest, divisão de pares, GraphSAGE heterogêneo e treinamento inicial em CPU. Os resultados atuais são exploratórios. A primeira comparação controlada e os rankings exploratórios foram executados em três seeds (ver relatório). Ainda não existe validação da qualidade das anomalias nem avaliação final de teste.

- Um dia de dados: 16/09/2026, com filtros já aplicados. O conjunto completo significa todo esse dia tratado, não toda a história da plataforma.
- Amostra de desenvolvimento: 5.000 carteiras. O experimento final usará o recorte completo após validação do protocolo e dimensionamento.
- Arestas agregadas não preservam toda a sequência de trades. O grafo representa participação em mercados, não contrapartes negociando entre si.
- Maker/taker ausente. Não inferir papel por volume ou frequência.
- As features atuais usam o recorte inteiro, inclusive informações agregadas de conexões ocultadas. A validação atual é reconstrução transdutiva condicionada a essas features; não é previsão temporal.
- Metadados de mercados, inclusive datas/duração, precisam de auditoria de disponibilidade temporal antes de qualquer alegação prospectiva.
- Negativos uniformes podem facilitar a tarefa de ligação. AP/AUC de ligação não avaliam fraude ou qualidade do ranking de anomalias.
- Carteiras sem vizinhos no contexto não recebem mensagens. Devem ser analisadas separadamente.
- Os resultados exploratórios já foram vistos; este plano não é um pré-registro retrospectivo.

## 3. Próxima implementação: protocolo comparável

Antes de novos rankings, produzir um protocolo versionado com exemplos de avaliação fixos e um relatório de integridade.

1. Separar seed da divisão, seed dos exemplos de avaliação, seed do treino e seed do IsolationForest. Implementado: `--seed` varia inicialização/dropout; `--eval-seed` e `--negative-seed` mantêm avaliação e negativos de treino independentes.
2. Reutilizar exatamente os mesmos positivos e negativos de validação/teste entre modelos e seeds do mesmo protocolo. Reservar ambos os conjuntos fora do treino, incluindo reversas e pares com múltiplos outcomes.
3. Decidir e registrar duas condições, sem misturá-las: (a) reconstrução transdutiva com features do dia inteiro; (b) controle com agregados das carteiras calculados somente das arestas de contexto e preprocessamento ajustado nas entradas permitidas. Declarar como tratar carteiras sem histórico no contexto. Nenhuma condição deve ser chamada de previsão futura sem uma separação temporal apropriada.
4. Preservar os artefatos e métricas atuais como exploração. Gerar novas pastas para as condições revisadas.
5. Definir orçamento de treinamento, critério de seleção e regra de parada comuns; registrar diferenças inevitáveis de capacidade e número de parâmetros.
6. Não usar o teste para ajustar a arquitetura, selecionar casos, escolher cortes ou decidir hiperparâmetros. Congelar as decisões usando desenvolvimento/validação antes da avaliação final.

Critério de conclusão: testes verificam ausência de sobreposição direta entre partições, negativos válidos, alinhamento de IDs e invariância dos exemplos de avaliação ao mudar apenas a seed do modelo.

## 4. Matriz mínima de comparações

| Experimento | Papel | Comparação legítima |
| --- | --- | --- |
| IsolationForest nas features individuais | Baseline de ranking já disponível | Rankings e caracterização; não comparar sua saída com AP de ligação |
| Encoder MLP por tipo de nó + mesmo decoder de pares | Controle treinado sem mensagens | Métricas de ligação e ranking dos embeddings, sob o mesmo protocolo |
| GraphSAGE + decoder atual | Método com mensagens | Mesmas métricas, nós e exemplos |
| Score simples por graus dos endpoints | Controle de atividade/popularidade | Métricas de ligação, sem interpretação como detector de fraude |
| GraphSAGE com contexto aleatorizado preservando graus | Ablação estrutural desejável | Verificar dependência da identidade dos vizinhos |

A ablação por aleatorização deve especificar o algoritmo, preservar graus, rejeitar duplicatas e evitar inserir pares positivos reservados. Se for inviável, documentar a restrição; não substituir por embaralhamento de linhas, que mantém o mesmo grafo.

Para rankings dos embeddings, manter o mesmo procedimento do IsolationForest e identificadores alinhados. Primeira análise: usar embeddings no contexto fixo do experimento. Eventual reajuste sobre todo o grafo para exploração deve ser identificado como outro experimento, separado da avaliação de ligações ocultadas.

Critério de conclusão: tabela de resultados com todas as variantes executadas, tempos, parâmetros e referências aos arquivos de origem, incluindo resultados desfavoráveis.

## 5. Avaliação de anomalias sem rótulos

Não usar apenas score alto como validação do próprio score. Não chamar o top 1% de taxa de fraude.

### Rankings e estabilidade

- Seeds iniciais propostas: 42, 43 e 44; divisão e avaliação fixas. Três seeds fornecem uma primeira análise descritiva, não forte evidência estatística sobre todos os cenários.
- Medir interseção/Jaccard entre top-k e correlação de postos; explicitar desempates.
- Cortes previamente definidos: 0,5%, 1% e 2%, com 1% como fila principal de revisão.
- Separar variabilidade do encoder e do IsolationForest: inicialmente fixar a seed do detector ao variar a GNN; depois medir a sensibilidade do detector separadamente.
- Não comparar valores absolutos de scores de modelos diferentes como se fossem calibrados na mesma escala.

### Inspeção dos casos

Planejar uma amostra de revisão antes de abri-la: top 20 de cada método (união com duplicatas removidas), mais controles de volume e grau semelhantes e controles aleatórios. Registrar algoritmo de pareamento, quantidade de controles, seed e critérios antes de selecionar os casos.

Usar formulário com: ID, métricas observadas, mercados compartilhados, tamanho/popularidade dos mercados, explicações alternativas, limitações e evidências disponíveis. Diferenciar ausência de evidência de evidência de ausência. Se possível, os dois integrantes avaliam independentemente sem visualizar o método/score, reconciliando divergências depois. Isso gera uma análise humana documentada, não rótulos confirmados de fraude.

Medidas candidatas de coparticipação: sobreposição de mercados, Jaccard de vizinhança, concentração e participação em mercados menos frequentes. Comparar com controles de atividade e popularidade. Essas medidas são diagnósticos estruturais, não verdade de referência independente; escolher e documentar antes de olhar os casos da GNN.

Se forem usados padrões artificiais injetados, criar experimento separado com cenários definidos antes da avaliação. Reportar desempenho somente nesses cenários sintéticos, sem extrapolar para fraude real.

Critério de conclusão: dossiês também para controles e casos sem interpretação convincente, sem selecionar apenas exemplos vistosos.

## 6. Dados, reprodução e escala

- Recuperar e versionar as consultas Dune, IDs/links, parâmetros e datas de extração; verificar a interpretação de wallet, volume e agregação. Os CSVs sozinhos não tornam a extração reproduzível.
- Registrar hashes, schema, contagens antes/depois dos filtros e justificativas. Não chamar todos os registros filtrados de bots confirmados.
- Preservar .venv fora do Git; registrar versões e comandos. Criar Makefile após estabilizar as etapas.
- Executar a reprodução no computador do colega e registrar divergências.
- Planejar dados adicionais de outros dias, se disponíveis, antes de selecionar quais produzem os melhores resultados. Sem novos dias, restringir a contribuição a estudo de caso.
- Executar o recorte completo após os controles, verificando memória, tempo e resultados separados da amostra.
- Antes de publicar dados, conferir condições de redistribuição e documentar proveniência; usar identificadores pseudônimos nos casos do artigo quando adequado, sem imputar conduta ilícita a titulares.

## 7. Artigo e submissão

Até 22/11/2026: entrega universitária prevista, com escopo compatível com os experimentos efetivamente concluídos. Essa data não é prazo confirmado de congresso.

Com o orientador, escolher uma chamada futura de escopo compatível (por exemplo, área pertinente do CLEI ou evento do CSBC). Verificar edição, prazo, trilha, tipo de artigo, limite de páginas, anonimização, template e condições de apresentação. O modelo IEEE do trabalho da disciplina não vale automaticamente para todo evento. Não assumir que uma chamada anterior será repetida.

Antes da submissão:

- [ ] Pergunta e contribuição específicas, confrontadas com trabalhos relacionados realmente lidos.
- [ ] Tabela de comparação e ablações suficientes para as alegações feitas.
- [ ] Análise de estabilidade e limites de amostragem/tempo.
- [ ] Casos e controles documentados; nenhuma afirmação de fraude sem evidência externa.
- [ ] Resultados de teste obtidos somente após congelamento das decisões.
- [ ] Fluxograma e tabelas refletem a implementação final, não apenas a proposta.
- [ ] Scripts/artefatos necessários reproduzidos em outro ambiente.
- [ ] Todos os números do texto rastreáveis a uma execução e versão.
- [ ] Regras da chamada, autoria, referências e declarações exigidas conferidas com o orientador.

Após pareceres: classificar críticas em método, evidência, escopo e redação; responder cada ponto e registrar correções. Rejeição pode exigir revisão ou outro veículo; não prometer aprovação nem alterar conclusões para agradar revisores.

## 8. Ordem das entregas e situação atual

1. Concluído na rodada 1: protocolo fixo e duas condições de features, registrados antes da execução.
2. Concluído na rodada 1: controle sem mensagens e baseline de grau; comparação em validação.
3. Três seeds concluídas; ablação estrutural ainda pendente.
4. Rankings GNN/MLP e comparação com baseline concluídos; qualidade dos casos ainda não validada.
5. Inspeção com controles e visualização dos subgrafos.
6. Recorte completo e, conforme acesso, novos dias.
7. Makefile, reprodução externa, artigo e revisão com orientador.
8. Adequação à chamada, submissão pelo autor responsável e tratamento dos pareceres.

Optuna e arquiteturas mais complexas são opcionais posteriores. Prioridade: evidências e clareza, não quantidade de ferramentas.


Retomada em 18/10/2026: [roteiro operacional](proximos_passos.md). Priorizar piloto sintético e inspeção real na amostra antes da bateria completa; a visualização didática já foi apresentada, mas a qualidade das anomalias ainda não foi validada.
