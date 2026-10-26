# GNN v2 — protocolo congelado antes da avaliação

26/10/2026. Branch feat/gnn-v2-anomalias, base 1a1d456. Não modificar a main nem algoritmos v1. Esta é avaliação sintética transdutiva no mesmo dia, não detecção confirmada de fraude.

## Hipóteses e controles

1. Features estruturais ajudam a recuperar coparticipação. Acrescentar log1p de grau, contagem média dos três maiores compartilhamentos, surpresa média dos três maiores, maior compartilhamento com parceiro de maior surpresa, sobreposição esperada desse parceiro. StandardScaler ajustado nos nós com contexto; IDs/parceiros/rótulos não entram. São descritores do próprio contexto, inclusive um score já conhecido; ganho não demonstra aprendizado independente desse score.
2. Objetivo adicional contrastivo ajuda a usar estrutura. Link BCE + 0.1 * discriminação normal/corrompido inspirada em Deep Graph Infomax (https://arxiv.org/abs/1809.10341), adaptada a dois tipos de nó: features permutadas entre nós ativos do mesmo tipo, arestas fixas, resumo sigmoid da média dos embeddings ativos, discriminator bilinear por tipo. Não é reprodução exata do artigo. Corrupção não é rótulo de anomalia real e não usa intervenção sintética. MLP equivalente controla benefício sem mensagens.
3. Pares podem ser mais úteis que outliers individuais. Ranking híbrido: por par com >=2 mercados em comum, surpresa Poisson aproximada * (1+cosine(embedding_u,embedding_v))/2; carteira recebe média das três maiores contribuições, com zeros se necessário. Sem ajuste de pesos por resultados. Comparar com surpresa pura e híbrido com features sem encoder. Não alegar ganho da GNN se MLP ou descritor direto forem iguais/melhores.

## Configuração fixa

Seis encoders: gnn_base,mlp_base,gnn_struct,mlp_struct,gnn_contrast,mlp_contrast. GraphSAGE e MLP originais, hidden64,embedding32,dropout0.1. Os contrastivos usam features estruturais. Seeds42,73,2026; Adam0.001,até100épocas,patience15,min_delta0.0001,CPU2threads. Mesmo orçamento e negativos para controles; contraste adiciona custo e parâmetros, registrado. Seleção de checkpoint por val_loss de ligação em todos os modelos, sem olhar alvos artificiais; escolha conservadora que pode limitar ganho do contraste. Seeds de avaliação43/44,negativos42,permutação de corrupção seed+10000 em gerador separado para não mudar dropout. Teste de ligação não avaliado.

Detectores: IF100árvores seed42 nos embeddings de todos os seis modelos; híbrido nos quatro encoders enriquecidos; IF nas features originais e enriquecidas; surpresa direta; híbrido nas features enriquecidas sem encoder. Métrica principal: recuperados@100; secundárias@50,AP,ROC-AUC,precisão/recall@100,universos completo/elegível/sem doadoras. Empates ID crescente. Não selecionar melhor seed. Comparações pareadas por realização, reportando todas.

## Dados e realizações novas

Usar o recorte novo de5000carteiras da validação ampliada (disjunto da amostra inicial, mas já usado em outros testes; não é população nem dia independente). Entrada data/processed/validacao_ampliada_v1/novo/. Gerar novas realizações antes de qualquer avaliação, seeds seleção/trocas20101/20102 e20201/20202. Três cenários por realização: coparticipação30%,coparticipação60% e volume10x,50alvos cada. Mesmos geradores preservados da main; nova elegibilidade por intensidade. Volume conserva topologia; comparação antes/depois necessária. Se inviável, registrar e não substituir. Nenhuma busca Optuna/tuning ou escolha de nova seed por desempenho.

Treinar uma vez no original e uma vez em cada cópia injetada; reusar rankings originais para comparar os grupos designados. Seis cenários x18treinos +18originais =126treinos quando todos viáveis. Corrupções e validação de ligações não acessam os rótulos sintéticos. Rótulos somente no avaliador após rankings.

## Critério de interpretação

Melhora sobre GNN v1 não basta para substituir detector simples. Manter como candidato principal apenas se ganhar de maneira consistente no top100/AP frente à GNN base e controles MLP/relacional nas duas realizações de coparticipação60%, sem ocultar falhas em30%/volume. Com duas realizações e um dia, não alegar significância populacional, generalização temporal ou fraude. Caso não ganhe, manter v2 como implementação experimental e documentar resultado negativo. Não buscar nova configuração nesta rodada após olhar resultados.

## Integridade e entrega

Salvar hashes antes de avaliar, inputs/IDs em JSON, scalers,checkpoints,históricos,embeddings,rankings,tempos,contagem de parâmetros,configurações e métricas. Testar invariantes de features sem ligações reservadas, MLP independente de arestas, contraste reproduzível e gradientes, limites do híbrido, checkpoints e ordenação de CSV com round_trip. Registrar código na branch, sem merge nem push automático. Relatório comparativo e próximos passos incluem revisão humana,mercados relacionados,outros dias,benchmark do completo,teste final de ligações e reprodução no computador do colega.
