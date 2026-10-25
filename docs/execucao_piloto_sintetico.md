# Piloto sintético: protocolo e instruções de implementação

Data: 25/10/2026. Estado: plano de execução; nenhum gerador ou experimento foi executado por este documento.

Atualização de execução: módulos implementados e 17 testes aprovados. Após a ausência inicial das entradas, os CSVs foram localizados e a amostra foi reconstruída com os scripts intactos. A realização v1 atingiu o limite de propostas e foi declarada inviável por conexões reservadas de um alvo; consulte [o resultado e o diagnóstico](piloto_validacao/resultado_piloto_v1.md). As etapas abaixo registram o protocolo congelado da v1; não usar a intervenção parcial para treino.

Objetivo: medir a recuperação de 50 carteiras submetidas a coparticipação artificial concentrada, comparando GNN + IsolationForest, MLP + IsolationForest e IsolationForest nas features. O rótulo significa participação na intervenção, não fraude. O experimento avalia recuperação de um cenário definido, sem validar anomalias reais.

## 1. Regra obrigatória: preservar os algoritmos existentes

- Não editar nenhum arquivo existente em `src/`, `tests/`, os mapeamentos JSON ou `requirements.txt`.
- Criar somente módulos novos em `experimentos/piloto_sintetico/` e testes novos em `tests/`, além da documentação e das saídas isoladas.
- Importar as funções públicas existentes. Não copiar implementações dos modelos, modificar seus objetos em tempo de execução, substituir funções ou alterar parâmetros internos.
- Não escrever em `data/raw/`, nas pastas das rodadas anteriores ou nos artefatos originais. Todo processamento usa cópias independentes; destinos existentes devem ser rejeitados.
- Registrar SHA-256 dos arquivos protegidos e das entradas antes/depois. Qualquer mudança interrompe a execução e invalida a rodada. Não restaurar ou apagar arquivos automaticamente.
- Arquitetura, perda, otimizador, amostragem negativa, checkpoint e detector permanecem os já implementados. As mudanças intencionais atingem apenas a cópia das conexões e as features dela derivadas. Pesos e rankings podem mudar: isso é o efeito experimental esperado.

Reutilizar `validar_divisao` e `treinar` de `src/treinar_gnn.py`, `features_contexto` de `src/features_contexto.py`, `exportar` e `salvar_ranking` de `src/ranking_gnn.py`. Não chamar diretamente o executor `experimento_gnn.executar`: ele executa também `full_day`, enquanto este piloto usa exclusivamente `context_only`.

## 2. Etapa A — verificar entradas e congelar o protocolo

### A1. Verificação inicial

1. Registrar commit, diferenças locais, versões e hashes dos arquivos protegidos.
2. Confirmar a existência de `user.csv`, `event.csv`, `edge.csv` e da divisão original, normalmente em `data/processed/dev/gnn_seed_42/divisao.pt`.
3. Carregar e validar a divisão. Conferir IDs, cobertura dos pares pelo CSV, valores finitos e contagens reais. Não assumir que as contagens documentadas correspondem à entrada recebida.
4. Executar a suíte existente no ambiente de pesquisa. Se faltar ambiente ou dados, registrar o bloqueio; não instalar dependências ou regenerar dados silenciosamente.

### A2. Configuração inicial proposta

Salvar os parâmetros em `experimentos/piloto_sintetico/protocolo_v1.json` antes de gerar rankings. A implementação deverá validar esse arquivo. Os valores abaixo são decisões propostas para o primeiro piloto, não valores já validados quanto à viabilidade.

| Item | Regra v1 |
| --- | --- |
| Universo | Mesmas carteiras e mercados da divisão de desenvolvimento |
| Intervenção | Somente nas arestas do contexto copiado |
| Condição | `context_only` |
| Carteiras-alvo | 50, sorteadas sem reposição entre carteiras com grau de contexto >= 5 |
| Mercados-alvo | 10, sorteados entre mercados com grau >= 25 no contexto original |
| Ordenação | Ordenar IDs originais antes de cada sorteio |
| Seed de seleção | 8101 |
| Seed das trocas | 8102 |
| Intensidade | Cada alvo termina com pelo menos `ceil(0.60 * grau_original)` vizinhos entre os 10 mercados-alvo |
| Elegibilidade adicional | O alvo precisa estar abaixo dessa concentração inicialmente; o número exigido não pode exceder 10 |
| Limite | 1.000.000 propostas de troca no total; nenhuma nova seleção ao falhar |
| Encoder | Seeds 42, 73 e 2026 |
| Negativos de treino / avaliação | Seeds 42 / 43, conforme API existente |
| IsolationForest | Seed 42; configuração existente |
| Treino | Até 100 épocas; demais parâmetros iguais aos padrões atuais, explicitados no protocolo |
| Ajuste | Retreinar sem supervisão no cenário original e no alterado |
| Cortes | Top-50 e top-100 |

Selecionar primeiro os mercados; depois calcular a elegibilidade das carteiras usando o conjunto selecionado. Se houver menos de 10 mercados ou 50 carteiras elegíveis, declarar inviabilidade. Registrar o viés de seleção para carteiras com atividade suficiente. Carteiras de grau zero permanecem no universo e não são alvos nem doadoras.

Guardar configuração e seu hash antes da execução. Alterar regra, intensidade, seed ou limite exige nova versão e nova pasta, preservando inclusive os cenários inviáveis. Não ajustar o cenário com base na vantagem de um método.

**Saída da etapa:** protocolo validado, inventário e hashes. A etapa termina com regras explícitas; resultados favoráveis não são requisito de conclusão.

## 3. Etapa B — implementar o gerador isolado

Criar `experimentos/piloto_sintetico/gerar.py`, sem alterar o código existente.

### B1. Troca bipartida com preservação de graus

Para um alvo `u` abaixo da concentração exigida:

1. Escolher uma conexão de contexto `(u, a)` com `a` fora dos mercados-alvo.
2. Escolher uma conexão `(v, b)`, com `b` nos mercados-alvo e `v` fora das 50 carteiras-alvo.
3. Propor substituir `(u, a), (v, b)` por `(u, b), (v, a)`.
4. Rejeitar se os novos pares já existem no contexto corrente ou entre quaisquer positivos originais de treino, validação ou teste. Rejeitar também qualquer par positivo original fora do contexto, mesmo que não seja alvo da avaliação atual.
5. Aceitar apenas se a concentração de `u` aumentar e todas as invariantes forem satisfeitas.
6. Atualizar o contexto corrente e registrar a troca. Usar ordem e gerador aleatório determinísticos.

Cada troca mantém o grau de ambas as carteiras e dos dois mercados, além do número de pares. Um mesmo doador pode participar de várias trocas; registrar todas. Se o limite for atingido antes de todos os alvos cumprirem a intensidade, salvar diagnóstico de inviabilidade e não treinar uma intervenção parcial como se fosse o cenário v1.

### B2. Atributos e múltiplos outcomes

Tratar todas as linhas agregadas de um par como um bloco indivisível. Na troca, manter o mercado e todas as suas linhas, outcomes, volumes, preços, contagens e timestamps; mudar apenas a identidade da carteira:

- Bloco de `(u, a)` passa a `(v, a)`.
- Bloco de `(v, b)` passa a `(u, b)`.

Atualizar juntos `wallet_id` e `wallet`, usando a correspondência original. IDs de mercado e outcome permanecem válidos. Essa regra evita transportar outcomes de um mercado para outro. As linhas são registros sintéticos, não novos trades observados.

Preservam-se o número de linhas, os valores de cada bloco e os totais por mercado. **Não se preservam necessariamente volume, preço médio, número de trades ou quantidade de linhas por carteira.** Recalcular as features de contexto e reportar essas mudanças; não atribuir todo ganho exclusivamente à topologia. O baseline individual ajuda a medir esse efeito. Preservar também esses agregados exigiria outro protocolo, sem relaxar a v1 silenciosamente.

### B3. Compatibilidade com a divisão existente

- Fazer cópia profunda da divisão original.
- Substituir apenas `context[FORWARD].edge_index` e a reversa correspondente.
- Preservar exatamente `train_positive`, `val_positive`, `test_positive` e todos os IDs/ordens dos nós.
- Reconstruir `all_positive` como a união do novo contexto com as três partições preservadas. Manter os pares únicos. Isso é necessário para o validador e o amostrador existentes continuarem corretos.
- Pares removidos do contexto deixam de ser positivos no cenário sintético. Podem ser sorteados como negativos pelo algoritmo existente; registrar essa consequência no relatório.
- Não calcular métricas de teste. Os endpoints positivos de teste já presentes na divisão só podem ser usados para as verificações de exclusão e integridade.

**Saídas:** cópia do CSV alterado, divisão alterada, `trocas.csv`, `rotulos.csv` separado das entradas, `manifesto_injecao.json` e diagnóstico de concentração antes/depois.

## 4. Etapa C — testar integridade antes de treinar

Criar `tests/test_piloto_sintetico.py` com fixtures pequenas, sem editar testes anteriores.

Verificar obrigatoriamente:

- Mesmo protocolo e mesmas entradas produzem as mesmas trocas, IDs, tensores e valores de saída. Comparar conteúdo canônico; arquivos serializados podem carregar metadados variáveis.
- Número de nós e pares preservado; graus por nó idênticos antes/depois; ausência de duplicatas e arestas órfãs; reversas corretas.
- Nenhum positivo reservado entra no contexto; alvos de treino/validação/teste permanecem idênticos; `validar_divisao` aprova ambas as cópias.
- Cada bloco conserva mercado, outcomes e atributos; identidade de carteira consistente; totais por mercado preservados.
- Exatamente 50 alvos cumprem a intensidade, ou a realização é marcada como inviável e o treino não inicia.
- Carteiras e mercados colateralmente afetados estão integralmente registrados. Distinguir alteração de endpoints de alteração de features.
- Arquivos protegidos e entradas originais mantêm seus hashes; destino existente é rejeitado.
- Rótulos, indicador de alvo, lista de mercados-alvo e dados de avaliação sintética não chegam às funções de treino ou ranking.
- Recalcular features via função existente produz valores finitos e alinhados. Na condição sintética, nenhuma aresta oculta contribui para agregados de carteira.
- Um ensaio curto confirma a compatibilidade dos três métodos sem modificar suas implementações.

Fazer também um ensaio sem intervenção: as duas cópias idênticas devem produzir features, exemplos de avaliação e scores iguais no mesmo ambiente determinístico. Esse controle detecta efeitos acidentais do novo executor. Não misturar o ensaio curto com os resultados finais de 100 épocas.

**Critério de avanço:** todos os testes relevantes passam; `integridade.json` registra cada verificação. Um simples campo `true` sem comparação efetiva não serve como evidência.

## 5. Etapa D — executar os três métodos

Criar `experimentos/piloto_sintetico/executar.py` como orquestrador novo.

1. Preparar dois cenários em pastas novas: `original/` e `injetado/`.
2. Chamar `features_contexto` separadamente em cada cenário com seu CSV coerente. Salvar divisão com features recalculadas, tabela bruta e preprocessador próprios. Nunca reutilizar features antigas no cenário alterado.
3. Confirmar que graus e nós ativos dos mercados permanecem iguais; seus metadados e transformadores devem permanecer iguais sob essa intervenção.
4. Gerar um ranking de features por cenário com `salvar_ranking`, mantendo `wallet_id`, `wallet`, `grau_contexto` e `sem_contexto` como esperado pela API.
5. Para GNN e MLP, chamar `treinar` nas seeds 42, 73 e 2026, com `feature_condition='context_only'`. Explicitar todos os parâmetros do protocolo nas chamadas.
6. Chamar `exportar` com a divisão exata que originou cada checkpoint, respeitando o hash exigido. Ele gera embeddings e aplica o mesmo IsolationForest.
7. Conferir igualdade exata dos pares/rótulos de validação e teste **entre modelos e seeds dentro de cada cenário**. Não computar métricas de teste.
8. Salvar relatórios e verificar novamente os hashes dos arquivos protegidos.

Total final: 12 treinos, 12 rankings de embeddings e 2 rankings de features, considerando os dois cenários. O baseline de features não precisa ser repetido por seed do encoder.

**Atenção à comparação antes/depois:** `all_positive` muda com a intervenção. O amostrador existente pode gerar negativos diferentes entre os cenários, mesmo com seeds iguais. Não prometer igualdade de negativos antes/depois e não alterar o amostrador para forçá-la. Registrar hashes e interseções dos negativos; a comparação principal é recuperação sintética dos mesmos alvos pelos três métodos na mesma realização. A mudança de posição antes/depois descreve o efeito do retreino completo, incluindo essa variabilidade. Métricas de ligação são diagnósticos separados.

Novas conexões também podem coincidir com negativos de avaliações históricas. Isso não contamina o teste original, pois seus modelos/arquivos ficam intocados e nenhuma avaliação histórica é reaplicada ao cenário alterado. O treino sintético gera suas próprias reservas válidas. Não reutilizar checkpoints antigos para essa comparação pareada.

## 6. Etapa E — avaliar recuperação sintética

Criar `experimentos/piloto_sintetico/avaliar.py`. Somente este módulo pode juntar os rótulos aos rankings, por `wallet_id`, validando unicidade e cobertura completa.

### E1. Universos e rótulos

- Principal: todas as carteiras; 50 alvos recebem 1, demais recebem 0 para a tarefa de recuperação dos alvos designados. Zero não significa normalidade real nem ausência de intervenção colateral.
- Identificar separadamente doadoras, carteiras intactas e isoladas. Mercados afetados também ficam no manifesto.
- Complementar: alvos versus carteiras não doadoras, com AP/AUC próprias e prevalência informada. Para top-50/top-100 principal, manter o universo inteiro; se houver ranking filtrado complementar, identificá-lo explicitamente.
- Relatar o grupo isolado e a população elegível. Não interpretar um bom resultado contra isoladas como evidência suficiente de recuperação estrutural.

### E2. Métricas

| Métrica | Definição |
| --- | --- |
| Recuperados@k | Quantos dos 50 alvos aparecem nas primeiras k posições |
| Recall@k | Recuperados@k / 50 |
| Precision@k | Recuperados@k / k |
| AP | `average_precision_score` dos rótulos sintéticos e `score_anomalia` |
| ROC-AUC | `roc_auc_score` com os mesmos rótulos/scores |
| Mudança de posição | `posicao_original - posicao_injetada`; positivo significa subida |

Usar k=50 e k=100. Score maior significa mais anômalo, conforme implementação atual. Preservar desempate por ID; reportar empates no corte. AP/AUC usam scores, não a flag de seleção top-1%. A AP de referência depende da prevalência; no universo de 5.000 carteiras com 50 alvos, a prevalência é 1%.

Calcular as métricas nos rankings original e injetado usando os mesmos 50 alvos. Comparar mudanças por método e seed correspondente; exportar posições individuais e resumo por grupo. Apresentar resultados das três seeds, média e desvio-padrão amostral, sem tratá-las como três amostras independentes de dados ou escolher a vencedora.

**Saídas:** `metricas.csv`, `posicoes_antes_depois.csv`, `resumo_seeds.csv`, `diagnostico_colaterais.csv` e `relatorio.md`. Relatar resultados desfavoráveis e o efeito sobre features individuais. Uma injeção viável sem recuperação é um resultado válido.

## 7. Organização e sequência de execução

Estrutura implementada; a pasta de resultados depende de uma intervenção viável:

```text
experimentos/piloto_sintetico/
  protocolo_v1.json
  gerar.py
  executar.py
  avaliar.py
tests/
  test_piloto_sintetico.py
data/processed/sintetico/piloto_v1_8101_8102/
  protocolo.json
  inventario_hashes.json
  manifesto_injecao.json
  trocas.csv
  rotulos.csv
  original/
  injetado/
  avaliacao/
  integridade.json
```

CLI implementada, executada na raiz e no ambiente de pesquisa com as dependências instaladas. Os comandos exigem os dados de entrada; destinos já existentes serão rejeitados:

```bash
# Antes da implementação e novamente após ela:
python -B -m unittest discover -s tests -v

# Após implementar os módulos e passar pelos testes:
python -B -m experimentos.piloto_sintetico.gerar --protocolo experimentos/piloto_sintetico/protocolo_v1.json --input data/processed/dev --split data/processed/dev/gnn_seed_42/divisao.pt --output data/processed/sintetico/piloto_v1_8101_8102
python -B -m experimentos.piloto_sintetico.executar --run data/processed/sintetico/piloto_v1_8101_8102
python -B -m experimentos.piloto_sintetico.avaliar --run data/processed/sintetico/piloto_v1_8101_8102

# Revisar o escopo das alterações antes de qualquer commit:
git diff --name-only
git status --short
```

O protocolo deve conter os parâmetros da seção A2; `gerar` congela a cópia usada pela rodada. `executar` lê apenas essa cópia e rejeita saídas de treino existentes. `avaliar` rejeita saída de avaliação existente. Definir essas validações antes de implementar a CLI. Incluir arquivos `__init__.py` novos se necessários à organização dos módulos.

`data/` é ignorado pelo Git: compartilhar manifestos e resultados explicitamente ou gerar um pacote separado. Não mudar `.gitignore` para versionar dados/modelos durante esta tarefa. Não fazer commit automaticamente.

## 8. Condições de conclusão e continuação

- Protocolo congelado antes dos resultados; gerador determinístico; intervenção viável e rastreável.
- Código dos algoritmos e entradas originais intactos, comprovados por hashes.
- Três métodos executados sob a mesma intervenção e orçamento; teste original sem métricas calculadas.
- Manifesto, testes, recuperação top-50/top-100, AP, ROC-AUC e mudança de posição entregues.
- Limitações sobre atributos, seleção de alvos, negativos e doadoras explícitas.

Após desenvolver a v1, reservar novas realizações com pares de seeds de seleção/trocas `(8201, 8202)` e `(8301, 8302)`, mantendo as regras congeladas. Executá-las somente após encerrar as escolhas do desenvolvimento; preservar e reportar inviabilidades. Uma nova versão que aproveite seus resultados deixa de tratá-las como avaliação reservada. Variações de intensidade ou proporção ficam para outro protocolo.
