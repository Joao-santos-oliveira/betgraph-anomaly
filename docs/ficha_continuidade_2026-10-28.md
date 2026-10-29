# Ficha de continuidade — Betgraph Anomaly

**Atualizada em 28/10/2026. Destinatário: colega que dará sequência à pesquisa.**

Ponto de partida confirmado: commit `19c9f5a214f9268f78f4f6bb5568aaf25cae38cd` (`feat: piloto sintetico`), última etapa realizada pelo colega. Esta ficha explica tudo o que avançou depois dele, incluindo a correção do piloto, a investigação do detector e as GNNs v2/v3. Os caminhos abaixo são relativos à raiz do repositório, para funcionar no computador de ambos.

## 1. Objetivo e situação atual

O projeto prioriza carteiras com comportamento incomum em um grafo carteira–mercado do Polymarket, usando dados de um dia: **16/09/2026**. Uma carteira pode participar de vários mercados; carteira não equivale necessariamente a uma pessoa. Uma aresta agregada pode representar vários trades de um mesmo outcome naquele mercado.

**Ainda não temos rótulos de fraude real.** Nos experimentos, “alvo” significa uma carteira alterada artificialmente. Os demais participantes também não são normais confirmados.

Temos um pipeline funcional e experimentos reproduzíveis. A principal pendência científica é avaliar a interpretação dos casos reais e controlar explicações alternativas, como mercados da mesma partida. A GNN ainda não demonstrou vantagem suficiente para ser o detector principal.

## 2. O que já havia e o que fizemos depois

| Etapa desde o commit do colega | O que fizemos | O que aprendemos |
|---|---|---|
| Ponto de partida: piloto sintético v1 (`19c9f5a`) | Gerador, protocolo e diagnóstico da injeção | Realização inviável: uma carteira precisava de dez mercados-alvo, mas só oito eram permitidos; não houve treino sobre a cópia parcial |
| Piloto sintético v2 (`1a1d456`) | Corrigimos a elegibilidade antes do sorteio, respeitando positivos reservados; completamos 50 alvos, 253 trocas e 12 treinos | A injeção passou a funcionar, mas GNN/MLP recuperaram zero alvos no top100; features+IF recuperou três |
| Investigação e validação ampliada (`1a1d456`) | Implementamos a surpresa de coparticipação e variamos estrutura/volume em duas amostras disjuntas de 5.000 carteiras; preparamos fichas reais | 24 tentativas, 21 concluídas e três inviáveis; relacional responde à coparticipação forte, features+IF ao volume; revisão humana ainda pendente |
| GNN v2 (`c0b7390`) | Acrescentamos features estruturais, objetivo contrastivo e ranking híbrido; 126 treinos | Híbrido contrastivo: 31,67/50 no top100 contra 32/50 do relacional em coparticipação60%; ganho adicional não demonstrado |
| GNN v3 (`deb24e5`) | Usamos atributos de aresta e relações por outcome, corrupção de arestas no treino e combinador manual como controle; 81 treinos | Resultado negativo pela regra congelada; preservar como comparação experimental, sem adotar como detector principal |

**Duas versões diferentes usam o nome “v2”:** o piloto v2 corrigiu o gerador de dados artificiais; a GNN v2 alterou o modelo e o ranking. Corrigir a injeção não significou que a GNN passou a encontrar os alvos.

Em linguagem simples, a sequência foi: **fazer a injeção completar → medir a recuperação → investigar por que o ranking não destacava os alvos → testar detectores explícitos e novas redes → preservar o resultado negativo e avaliar casos reais.**

As médias da v2 e da v3 vêm de realizações diferentes. A comparação justa da rodada nova usa também a v2 retreinada nas mesmas cópias da v3.

Decisões de preparação que continuam válidas: maker/taker não está disponível e não é substituído por volume/frequência; duração de mercado ausente recebe flag e imputação pela mediana; `duracao_total_segundos` foi excluída do primeiro modelo; “primeira atividade” significa primeira atividade observada no recorte filtrado. Transformadores e mapeamentos de IDs são preservados.

Fontes: `docs/piloto_validacao/resultado_piloto_v1.md`, `docs/piloto_validacao_v2/resultado_piloto_v2.md`, `docs/investigacao_anomalias/resultado.md`, `docs/validacao_ampliada/resultado.md` e `docs/gnn_v2/resultado.md`. `docs/proximos_passos.md` é um roteiro histórico anterior à v3; usar esta ficha para a sequência atual.

## 3. O que mudou tecnicamente na v3

- **GNN com arestas:** duas camadas TransformerConv usam volume, preço médio e fração do dia; relações distinguem os dois lados locais observados de cada mercado.
- **Treinamento:** previsão de ligações mais um termo auxiliar que distingue arestas observadas de arestas com atributos permutados/lados invertidos. O checkpoint usa loss sem rótulos de intervenção.
- **Ranking principal:** similaridade entre embeddings de carteiras que compartilham mercados. IsolationForest nos embeddings é uma análise secundária previamente definida.
- **Controles:** relacional direto (A), média dos postos de quatro canais manuais (B), MLP com os mesmos agregados sem mensagens (D), GNN v2 híbrida e features+IF.
- **Integridade:** alterações em cópias, graus preservados, pares reservados excluídos das inserções, features recalculadas e rótulos artificiais usados somente na avaliação.

Código novo: `experimentos/gnn_v3/`. Explicação dos módulos: `docs/gnn_v3/README.md`.

A rodada usa a segunda amostra de **5.000 carteiras**, em `data/processed/validacao_ampliada_v1/novo/`. Não é o conjunto completo nem outro dia. Seeds dos modelos: **42, 73, 2026**; injeções de desenvolvimento: **30101, 30201**; IF: **42**.

## 4. Cenários e resultados

- **S1:** 50 alvos com cerca de 30% de coparticipação em dez mercados, mesmo outcome, horários próximos e valores semelhantes.
- **S2:** 50 decoys com 60% de coparticipação, lados balanceados e horários dispersos. Aqui, destacar mais decoys é indesejado. Em mercado binário, não é possível tornar os 50 participantes mutuamente opostos.
- **S3:** duas regressões separadas: coparticipação de 60% e volume multiplicado por dez.

Média de alvos/decoys no **top100 do universo completo de 5.000 carteiras**, entre 50 alterados por realização. Números fracionários são médias, não frações de pessoas.

| Método | S1: combinado | S2: decoys ↓ | S3: estrutura | S3: volume |
|---|---:|---:|---:|---:|
| Relacional direto (A) | 0,50 | 29,50 | 32,00 | 0,50 |
| Combinador manual (B) | 0 | 0 | 0 | 0 |
| GNN v3 principal (C) | 0 | 0 | 0 | 0 |
| GNN v3 + IF | 0,17 | 0,17 | 0 | 3,83 |
| MLP sem mensagens (D) | 0 | 0 | 0 | 0 |
| MLP + IF | 0 | 0,50 | 0,33 | 3,83 |
| GNN v2 híbrida | 0,50 | 29,00 | 31,83 | 0,50 |
| Features + IF | 0 | 2,00 | 0 | 9,00 |

Em S1, AP média: **A=0,0343; B=0,0454; C=0,0300**. A GNN principal falhou em superar A/B nas seis combinações de realização e seed. Também falhou nas regressões; passou apenas no controle S2 do universo completo.

**Nuance:** restringindo o ranking às carteiras elegíveis, C recuperou 6,83 alvos em S1, contra 11,50 de B. Esse universo contém 752/750 carteiras, então top100 representa uma parcela muito maior do conjunto. Não podemos trocar o universo principal depois de ver os resultados. Em S2, C destacou 9,33 decoys nesse universo restrito.

Fontes: `docs/gnn_v3/metricas.csv`, `resumo_por_realizacao.csv`, `decisao.json` e `resultado.md`. Esses arquivos também mostram top50, recall, precisão, AP, AUC, mínimos/máximos, antes/depois, universo sem doadoras e ganho incremental.

## 5. O que podemos concluir — e o que não podemos

**Conclusão defensável:** nesta configuração, amostra e protocolo, a GNN com atributos de aresta não acrescentou eficácia consistente aos comparadores. Sua implementação e seus resultados devem ser preservados como ablação, isto é, uma comparação que ajuda a entender quais componentes agregam valor.

O relacional consegue recuperar coparticipação forte, mas também priorizou decoys. Logo, conexão compartilhada não basta para concluir alinhamento de apostas ou coordenação suspeita. Features+IF respondeu melhor ao aumento artificial de volume, ainda recuperando apenas parte dos alvos.

AUC de aproximadamente 0,85 para C em S1 não compensa zero alvos no top100: ordenação global e utilidade dos primeiros casos são avaliações diferentes. Evitar decoys enquanto também se perde os alvos não demonstra um bom detector.

**Não podemos afirmar:** fraude detectada, eficácia geral em outros dias, impossibilidade de qualquer GNN, ou que a combinação de sinais proposta ficou necessariamente forte. Os canais combinados também tiveram dificuldade no topo. Todos os treinos C/D escolheram a última época permitida; convergência não foi demonstrada. Nenhum desses limites autoriza omitir o resultado negativo.

## 6. Sequência prática para o colega

| Ordem | Ação | Entrega esperada / critério de término |
|---|---|---|
| 1 | Receber arquivos, ler esta ficha e conferir ambiente | Código e entradas identificados; testes locais funcionando; sem retreino inicial |
| 2 — prioridade | Fazer a revisão humana independente das fichas existentes | Dois formulários preenchidos antes de revelar chave/método; síntese de concordâncias, divergências e evidência insuficiente |
| 3 | Verificar como agrupar mercados da mesma partida/evento | Mapeamento com origem, cobertura e ambiguidades; protocolo de controle congelado antes de executar |
| 4 | Avaliar controles por evento/modalidade, após preparar os metadados | Novo experimento separado, preservando graus e pares reservados; comparação com o relacional global |
| 5 | Medir custo no conjunto completo do dia e obter outros dias | Benchmark de RAM/tempo; dados e protocolo próprios; resultados separados por recorte |
| 6 | Fechar reprodução e texto científico | Makefile ou comandos equivalentes, execução no segundo computador, tabelas completas, limitações e revisão com orientador |

Não iniciar outra busca de GNN/Optuna como continuação automática desta rodada. A tentativa única está encerrada. Uma hipótese diferente precisaria de decisão explícita e novo protocolo; os resultados atuais continuariam preservados.

### Como realizar a revisão humana

Já existem **50 fichas: 25 casos, 15 controles pareados e dez aleatórios**. Dez casos ficaram sem controle dentro do limite definido; não relaxar o pareamento retrospectivamente.

Pasta: `data/processed/validacao_ampliada_v1/casos_reais/`.

1. Um responsável distribui somente `F*.md`, imagens associadas e um dos arquivos `formulario_revisor_1.csv` / `formulario_revisor_2.csv` para cada revisor. Conferir as imagens referenciadas pelas fichas.
2. **Não distribuir inicialmente** `chave_nao_mostrar_revisores.csv` nem `pareamento.csv`.
3. Revisores preenchem separadamente os campos existentes: padrão observado, explicação legítima, evidência insuficiente, confiança e observações. Antes de começar, combinar uma escala comum para confiança e registrar a instrução, sem alterar a seleção dos casos.
4. Descrever evidências: mercados relacionados, lados escolhidos, compatibilidade dos horários, concentração de atividade e explicações alternativas. “Não foi possível concluir” é resposta válida. Não preencher como fraude/normal confirmado.
5. Só depois comparar respostas e abrir a chave. Preservar formulários originais; registrar consenso e desacordos em uma síntese separada. Proposta de saída: `docs/revisao_humana/resultado.md`.

O cegamento é parcial: as fichas usam o parceiro de maior surpresa para todos os participantes. Subgrafos mostram até dez mercados compartilhados com um parceiro; não representam toda a atividade. Registrar esse limite na análise.

### Agrupamento de mercados

O CSV possui `question`, `tags` e `categoria_principal`, mas **não possui slug nem ID explícito de evento-pai**. `condition_id` identifica mercado, não partida. Preferir metadados reextraídos; regras por texto precisam de validação e grupo “desconhecido” para ambiguidades.

A proposta já está em `docs/gnn_v3/agrupamento_mercados.md`. O nulo estratificado — comparação com grafos aleatorizados dentro de grupos comparáveis — foi **apenas proposto**, não executado. Não usar o próprio ranking para definir os grupos.

## 7. Como receber o projeto e começar no outro computador

Estado atualizado: **v2 na branch `feat/gnn-v2-anomalias`, commit `c0b7390`; v3 na branch `feat/gnn-v3-anomalias`, com o experimento no commit `deb24e5`**. A branch v3 deriva da v2, preservando suas dependências e comparadores. Esta ficha e a nota de revisão acompanham a v3 em um commit documental separado. Main permanece em `1a1d456`.

**Os commits da v3 são locais; ainda não foi feito push.** O responsável deve publicar a branch `feat/gnn-v3-anomalias` e informar seu hash ao colega, ou transferir uma cópia identificada. Depois do envio, o colega deve buscar e selecionar essa branch; permanecer em main ou na branch v2 não traz a v3. Dados e modelos continuam precisando de transferência separada.

Compartilhar separadamente, mantendo os caminhos relativos:

- Para começar a revisão humana: somente o pacote de fichas/imagens e um formulário, sem chave/pareamento para os revisores.
- Para consultar os experimentos: `data/processed/gnn_v3_v1/` (aproximadamente 496 MB nesta execução).
- Para reproduzir a preparação: `data/processed/validacao_ampliada_v1/novo/{user,event,edge}.csv`, `divisao.pt` e `data/raw/betgraph_nos_evento_16set_final.csv`, referenciados no congelamento.
- Para preservar a trilha técnica completa: `data/processed/gnn_v3_v1_preparacao_interrompida/`. Houve correção de leitura de horários mistos antes dos treinos; a versão anterior foi arquivada.

Dados/modelos são ignorados pelo Git. Não compartilhar a `.venv` como substituto da instalação: recriar o ambiente. Referência da execução: Python 3.12, versões exatas em `requirements.txt` e `docs/gnn_v3/congelamento.json`. Em Linux/WSL, na raiz:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -B -m unittest discover -s tests -v
```

Os comandos acima instalam/testam; não repetem a bateria experimental. Se houver incompatibilidade, registrar o erro antes de trocar versões. Para comparar arquivos transferidos, usar os SHA-256 de `congelamento.json` e `data/processed/gnn_v3_v1/artefatos_sha256.json`.

**Atenção à proveniência:** o executor congelado verifica branch e HEAD c0b7390. Depois de um novo commit, a auditoria pode recusar esse checkout. Não remover essa checagem nem editar hashes para fazê-la passar. O relatório explica a reprodução em uma cópia separada da base original com os arquivos novos copiados. Não é necessário executar a auditoria completa para preencher as fichas humanas.

## 8. Proteções e checklist de passagem

- [ ] Colega recebeu a branch v3 ou um pacote identificado, e conferiu o hash usado.
- [ ] Fichas e imagens abrem; chave e pareamento separados dos revisores.
- [ ] Revisão independente feita antes da discussão conjunta.
- [ ] Teste final de ligações e realizações **30301/30401** continuam reservados; abrir só com protocolo final acordado.
- [ ] Nenhum arquivo congelado foi sobrescrito; trabalhos novos ficam em pastas próprias.
- [ ] Relatórios mantêm resultados desfavoráveis e distinguem ligação, intervenção artificial e caso real.
- [ ] Conjunto completo não é confundido com toda a plataforma: o tratado tem 27.039 carteiras + 16.693 mercados, 358.689 linhas agregadas e 229.072 pares únicos, segundo `docs/proximos_passos.md`. Seu benchmark ainda é pendente.

A rodada v3 encerrou com **40 testes aprovados, 81 checkpoints/scores reproduzidos e 133 arquivos anteriores intactos**, conforme `docs/gnn_v3/auditoria.json` e saídas dos testes. `src/` e main não foram alterados nessa rodada.

**Primeira entrega sugerida ao colega:** revisão independente das fichas, seguida de uma síntese das explicações alternativas e dos casos sem evidência suficiente. Essa etapa reduz uma incerteza científica que mais ajustes de rede, por si só, não resolvem.

## 9. Leitura curta para retomar sem acompanhar todos os treinos

1. Ler as seções2–5 desta ficha para entender a evolução e as conclusões.
2. Conferir `docs/piloto_validacao_v2/resultado_piloto_v2.md` para separar a correção da injeção da qualidade do ranking.
3. Ler a conclusão de `docs/gnn_v2/resultado.md` e `docs/gnn_v3/interpretacao.md`; os CSVs completos ficam disponíveis para conferência, sem precisar começar por eles.
4. Seguir a revisão humana descrita na seção6.
5. Consultar `docs/revisao_versionamento_2026-10-28.md` para saber o que já está commitado, o que falta e o que foi limpo.
