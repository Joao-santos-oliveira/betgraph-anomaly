# Agrupamento de mercados: inspeção e proposta, não executada

CSV inspecionado: `data/raw/betgraph_nos_evento_16set_final.csv`. Colunas: `condition_node_id`, `condition_id`, `question`, `tags`, `closed`, `active`, `archived`, `outcome_status`, `categoria_principal`, `eh_mercado_recorrente`, `duracao_mercado_segundos`, `duracao_mercado_valida`.

Há categoria e tags; não há slug nem identificador explícito de evento-pai/partida. `condition_id` identifica o mercado, não agrupa mercados da mesma partida. A categoria Sports não identifica, sozinha, modalidade ou jogo. `question` e `tags` podem ajudar a construir uma taxonomia, mas nomes semelhantes não bastam para garantir o vínculo. Não foi executado agrupamento, reextração ou novo nulo nesta rodada.

## Protocolo proposto para uma etapa futura

1. Preferir reextração de metadados com ID do evento-pai, slug, modalidade/competição, participantes e data/hora. Manter origem e cobertura do vínculo mercado→evento. Se não houver reextração, propor regras determinísticas sobre question/tags e validar manualmente uma amostra antes de usar; ambiguidades ficam em grupo desconhecido.
2. Congelar o mapeamento antes de olhar rankings ou rótulos de intervenções. Não agrupar por semelhança de carteiras, pois isso reutilizaria o sinal que se quer avaliar.
3. Definir níveis: categoria/modalidade/competição/data e evento-pai. Registrar estratos pequenos sem trocas possíveis; não relaxar silenciosamente.
4. Construir nulos por trocas bipartidas restritas a mercados do mesmo estrato, preservando graus de carteiras e mercados, ausência de duplicatas e exclusão de pares reservados. No nível evento-pai, preservar também o número de participações de cada carteira no evento. Decidir antecipadamente se atributos viajam com blocos ou se há permutação condicional de outcome/tempo/valor — são hipóteses nulas diferentes.
5. Fixar, em novo protocolo, sementes, número de réplicas, estatística de comparação e correção por múltiplas comparações. Comparar a surpresa observada à distribuição nula condicional; reportar também mistura das cadeias, trocas aceitas e estratos congelados.
6. Comparar o nulo global atual ao estratificado: compartilhar vários mercados da mesma partida pode ser comum e reduzir o destaque após condicionamento. Isso melhora a interpretação de comportamento incomum; não estabelece fraude.

A presente GNN v3 não usa esse agrupamento nem incorpora o resultado de qualquer nulo estratificado.
