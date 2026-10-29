# GNN v3 — tentativa única, protocolo 1.1

26/10/2026. Branch feat/gnn-v2-anomalias, base c0b7390. Não alterar main/src/arquivos anteriores; sem commit/push. Encerrar até 02/11/2026. Uma configuração, sem busca, ajuste ou substituição após métricas. Hashes SHA-256 serão escritos em congelamento.json antes da execução. Testes em fixtures não são avaliação experimental.

## Pergunta e dados

Quando coparticipação, outcome, tempo e valores são combinados, a GNN de arestas acrescenta algo aos canais manuais? Entrada fixa: data/processed/validacao_ampliada_v1/novo/{user,event,edge}.csv e divisao.pt, amostra já usada de 5.000 carteiras do dia 16/09/2026. As realizações são novas, não o dia. O horário é primeiro_trade de uma aresta agregada; não representa uma sequência de trades nem sincronia comprovada. Valor = volume_usd agregado, não valor de uma aposta individual.

Outcomes são identificados DENTRO de cada mercado, não entre mercados distintos. Há no máximo dois outcomes observados por mercado no contexto. Ordenar os token_outcome_id originais por mercado e salvar catálogo; relações bets_0/bets_1 e reversas indicam os lados locais. Não inferir um segundo outcome não observado. Nomes de equipes não viram milhares de convoluções independentes. Catálogo e scalers usam somente contexto. Outros IDs não entram nas features.

## Realizações

Desenvolvimento: seleção 30101 e 30201; trocas usam seleção+1, atributos seleção+50. Seeds reservadas 30301/30401 NÃO são geradas, abertas ou avaliadas nesta tarefa. Não avaliar o teste final de ligações; positivos reservados servem exclusivamente a exclusão de pares/integridade.

50 alvos por cenário. Trocas bipartidas de blocos preservam graus de carteiras e mercados, rejeitam duplicatas e positivos reservados; até 1.000.000 de propostas, sem trocar seed nem relaxar regras. S1/S2 escolhem 10 mercados com grau>=25 e dois outcomes observados no contexto; os critérios de elegibilidade v2 permanecem. Artefatos incluem trocas, alvos, doadoras, mercados afetados e antes/depois de atributos. Falha recebe status não concluída, sem métricas de uma injeção parcial.

- **S1 combinado fraco:** concentração mínima ceil(0.30*grau) em 10 mercados, grau>=5. Nas participações dos alvos nesses mercados, consolidar outcomes do par numa única linha do lado local0. Volume por par = mediana original do volume por par naquele mercado * Uniforme(0.8,1.2). Horário = mediana original dos primeiros horários do mercado + Uniforme(-1800,+1800)s, com centro restrito a [1800,84600]. Mesmos valores/outcome/horário se referem a cada mercado, não a uma equipe global. Preço = mediana observada desse outcome no mercado. n_trades = soma das linhas do par; ticket=volume/n_trades; duração=0 para representar o lote artificial num mesmo instante. A duração não entra no modelo. Os sinais devem ser interpretados como hipótese de fracos individualmente; serão reportados os canais isolados para conferir, sem ajustar a intensidade.
- **S2 decoys:** concentração60%; nos mercados-alvo, dividir os participantes de cada mercado de modo balanceado entre seus dois outcomes, com permutação independente por mercado. Não é possível tornar todos os 50 participantes mutuamente opostos num mercado binário. Distribuir seus horários uniformemente ao longo de 2%–98% do dia em ordem permutada. Manter volume total por par, usar preço mediano do outcome, somar trades e atualizar ticket/duração. Decoys ainda podem compartilhar lados por acaso e mercados relacionados; reportar esse limite. Esperado: baixo destaque nos métodos que realmente usam os canais combinados.
- **S3 regressão:** dois subcenários separados, compatíveis com publicações anteriores: S3_estrutura (coparticipação60% com atributos originais dos blocos, gerador v2) e S3_volume (volume10x apenas nas arestas de contexto de50carteiras com grau>=5 e volume>0; topologia inalterada, ticket recalculado). Não confundir com uma intervenção conjunta inédita.

Toda alteração ocorre em cópia. Preservar bytes das entradas originais; pares reservados e suas linhas ficam intactos. Recalcular features de contexto e scalers em cada cópia. Doadoras também mudam; atributos agregados e mensagens podem afetar outros nós. Mesmos grupos e exemplos para todos os métodos em cada realização. Alteração do número de linhas por consolidação de outcomes é permitida e registrada; graus de pares únicos devem permanecer iguais.

## Métodos e fórmulas congeladas

Definir C_uv = mercados comuns, d_u = grau, s_uv = surpresa Poisson aproximada existente para |C_uv|>=2 (não p-valor). Para cada mercado, outcome dominante de uma carteira = linha com maior volume, empate token_outcome_id menor; t é seu primeiro horário em segundos e v seu volume. Pares com menos de dois mercados comuns não entram. Resumo top3(x_uv) = soma dos três maiores /3, completando com zeros.

**A:** relacional direto existente: top3(s_uv).

**B:** quatro canais: A; O_u=top3(sum_m 1[outcome_um=outcome_vm]/sqrt(d_u*d_v)); T_u=top3(sum_m exp(-abs(t_um-t_vm)/1800)/sqrt(d_u*d_v)); V_u=top3(sum_m exp(-abs(log1p(v_um)-log1p(v_vm))/0.25)/sqrt(d_u*d_v)). Cada soma percorre mercados comuns, e cada top3 é calculado separadamente. Posto normalizado=(rank_average-1)/(N-1), crescente com score. B=média dos quatro postos. Sem pesos aprendidos ou uso de rótulos. Reportar também os quatro canais isolados, como diagnósticos pré-especificados.

**C principal:** GNN heterogênea, duas camadas TransformerConv com edge_dim=3, quatro relações (dois lados e reversas), hidden32, embedding16, heads1, concatFalse, dropout0.1, agregação soma das relações. Edge_attr = StandardScaler([log1p(volume),preco_medio,fração do dia]), ajustado nas linhas de contexto. Nós recebem features context_only originais mais, para cada lado, log1p(contagem), médias e desvios dos três atributos de aresta (14 agregados por nó). Scaler desses agregados ajustado nos nós ativos. Não incluir A/B nem flags de injeção.

Decoder de ligação: concat(z_user,z_event)->Linear32->ReLU->Linear1. Cabeça auxiliar: concat(z_user,z_event,edge_attr,onehot_lado)->Linear32->ReLU->Linear1. Loss = BCE_ligação +0.25*BCE_real_vs_aresta_corrompida. Corrupção: permutar triplas de atributos entre arestas e inverter lado local em mercados com dois outcomes; arestas migram de relação e reversas são atualizadas, endpoints mantidos. Isso corrompe atributos/tipos de aresta, não features de nó nem necessariamente a topologia simples. Encoder também roda no grafo corrompido. Features de nó permanecem as do contexto limpo em ambos os métodos, como condição transdutiva fixa.

Separar20%das linhas de contexto para o termo auxiliar de validação, seed55; restante para auxiliar de treino. Validação auxiliar fixa (corrupção seed56), treino corrupção seed57+época, ambos independentes do encoder. As linhas auxiliares de validação continuam observáveis no grafo de contexto: é seleção auto-supervisionada transdutiva, não validação temporal ou topológica independente. Checkpoint pela loss composta de validação, sem rótulos de injeção.

Score principal de C = top3((1+cos(z_u,z_v))/2) entre carteiras com>=2mercados comuns; nenhum fator A/B. Ele mede semelhança aprendida, não probabilidade de fraude. Reportar **C_IF** = IsolationForest nos embeddings, mas não trocar o score principal após resultados.

**D/D_IF:** MLP por tipo, Linear(input,32)->ReLU->Dropout0.1->Linear(32,16), mesmas entradas agregadas, decoders, objetivo, exemplos e score de pares/IF. Sem mensagens. Parâmetros podem diferir; registrar.

**E:** GNN v2 contrastiva+híbrido intacta, com configuração publicada (100épocas,hidden64,embedding32,patience15) e features+IF originais de contexto. Reajustar nas mesmas realizações, não comparar scores treinados em outra injeção. Exemplos de ligação de validação: exatamente a amostragem v2 (seed43; exclusão de todos positivos), iguais entre C,D,E e todas as seeds por realização. Negativos de teste seed44 apenas reservados/excluídos; nunca avaliados. Seeds dos encoders42,73,2026; IF100árvores seed42.

C/D: Adam0.001, máximo60épocas,patience10,min_delta0.0001,clipnorm5, CPU2threads. No máximo3processos independentes. Uma configuração única; custos e parâmetros registrados. Nove grafos previstos (original +8intervenções),81treinos de encoders se todos viáveis. Original treinado uma vez, rótulos de cada realização somente no avaliador posterior.

## Métricas e decisão

Top50/100,recall@100,precision@50,AP,ROC-AUC por realização/seed, média/mín/máx. Computar também nos universos elegível e sem doadoras, com cortes aplicados dentro de cada universo e identificados como tais. Para C e C_IF, ganho incremental = alvos no top100 do método que não estão no top100 de A/B, separadamente e fora da união. Overlap = quantidade de carteiras comuns entre os top100. No S2, rótulo1 significa decoy e recuperados/50 é taxa de priorização dos decoys; AP/AUC são descritivas com direção oposta à desejada, não evidência positiva. Não agregar S1/S2 num único placar de eficácia.

**Regra de decisão (literal):** a GNN só é declarada útil se, em S1 e nas duas realizações, superar A e B em recall@100 e AP em todas as seeds, sem piora relevante em S3 e sem priorizar os decoys de S2. Caso contrário, o resultado é negativo e deve ser reportado como tal.

Operacionalização anterior aos resultados: regra aplicada exclusivamente a C principal, universo completo; C_IF secundário. Em S2 exigir zero decoys no top100 em ambas as realizações/todas as seeds. Em S3, não perder mais de5alvos no top100 nem ter AP inferior a90%da referência contemporânea: E_v2 na estrutura (mesma seed) e features+IF no volume. Valores históricos entram como contexto, não como comparações pareadas de realizações diferentes. Uma realização necessária não concluída impede declarar utilidade (resultado inconclusivo nessa condição, decisão geral não demonstrada). Nenhuma realização reservada será aberta após a decisão nesta tarefa.

## Integridade, auditoria e limites

Testes: graus/partições/duplicatas, catálogo/outcomes/ticket/horários, linhas reservadas intactas, sensibilidade do encoder a edge_attr e outcome, MLP independente de mensagens, corrupção reproduzível, exemplos idênticos e whitelist sem flags. Auditoria reexecuta scores/checkpoints e métricas sem retreinar ou ajustar. Arquivos protegidos em c0b7390 e todas as entradas terão hashes conferidos no fim. Registrar saídas de testes. Qualquer mudança após métricas exige novo protocolo preservando o anterior; esta tarefa não autoriza uma segunda rodada de tuning.

Ler apenas schema/exemplos dos CSVs para propor agrupamento por evento/modalidade em agrupamento_mercados.md. Não executar agrupamento ou nulo estratificado. Fonte técnica da convolução: documentação oficial PyG TransformerConv; assinatura instalada conferida. Não inventar referências. Todos os resultados são de intervenções artificiais em um único dia, sem fraude confirmada.

### Detalhes de reprodução fixados antes da rodada

Os14agregados usam os atributos **antes** do scaler de arestas: log1p(volume), preço e fração do dia; desvio populacional, zero em grupos vazios. A ordem é contagem, três médias, três desvios para cada lado. Scalers por tipo usam nós ativos. Os CSVs de cada cópia são salvos e relidos antes da preparação para que auditoria e execução usem a mesma representação numérica. O checkpoint inicial (época0) participa da seleção; não se força melhora de loss.

Consolidar outcomes e usar sempre o lado local0 em S1 cria pistas artificiais adicionais. Os modelos podem responder a essas pistas em vez de aprender uma combinação geral de sinais; eventual sucesso ficaria restrito a essa intervenção. A redução de linhas é registrada e os graus de pares únicos permanecem fixos. Isso não será corrigido após resultados nem omitido do relatório.


### Revisão técnica1.1, antes de qualquer treino experimental ou consulta às métricas de recuperação

A preparação1 foi interrompida ao ler uma coluna com horários originais separados por espaço e horários sintéticos ISO8601 com T/fração de segundo. Correção: `pd.to_datetime(..., utc=True, format="mixed")`, com teste unitário específico. Não muda valores, cenários, seeds, fórmula, hiperparâmetros ou regra de decisão. Trata-se da mesma rodada; não houve seleção por resultados nem treino a repetir.

Versão anterior preservada em `protocolo_v1.md` e `congelamento_v1.json`. Código, teste e preparação parcial anteriores preservados em `data/processed/gnn_v3_v1_preparacao_interrompida/`, incluindo `codigo_v1/` e `erro_preparacao.txt`. A revisão é congelada novamente antes de retomar a preparação em `data/processed/gnn_v3_v1/`. Rankings originais calculados durante a preparação não foram avaliados contra rótulos nem consultados para decidir esta correção.
