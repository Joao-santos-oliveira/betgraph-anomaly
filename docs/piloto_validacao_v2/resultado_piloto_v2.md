# Resultado do piloto sintético v2 — 25/10/2026

**Piloto de desenvolvimento concluído: intervenção viável, 12 treinos e 14 rankings avaliados. Recuperação baixa: GNN e MLP não recuperaram nenhum dos 50 alvos no top 100 em nenhuma das três seeds. O IsolationForest nas features recuperou 2 no top 50 e 3 no top 100.** Os resultados desfavoráveis foram preservados; não houve ajuste de intensidade ou hiperparâmetros após observar as métricas.

## Correção e geração

A v1 falhou por selecionar uma carteira que precisava de dez mercados-alvo mas só tinha oito permitidos. Na v2, o filtro de elegibilidade desconta os mercados-alvo que possuem positivos reservados da carteira, antes do sorteio. Mantidos 50 alvos, dez mercados, concentração 60%, grau mínimo 5, seeds 8101/8102 e todos os parâmetros de treino/detector.

A regra identificou quatro carteiras inviáveis por capacidade individual (IDs 12694, 12935, 22391 e 31184). O conjunto elegível passou de 659 para 655; a nova seleção foi sorteada com a mesma seed. O gerador v1 mantém a regra original quando version=v1; protocolo e relatório anteriores continuam preservados.

| Verificação | Resultado |
| --- | ---: |
| Carteiras / mercados na amostra | 5.000 / 9.189 |
| Pares de contexto / linhas agregadas no CSV completo da amostra | 23.738 / 62.328 |
| Carteiras-alvo com intensidade atingida | 50 de 50 |
| Propostas / trocas aceitas | 512 / 253 |
| Carteiras doadoras alteradas | 228 |
| Mercados afetados pelas trocas | 230 |
| Graus dos alvos no contexto | 5–16 |

Os graus de todas as carteiras e mercados foram preservados, assim como IDs, partições de alvos e atributos dos blocos por mercado. Nenhum positivo reservado foi inserido no contexto. A intervenção usa somente cópias e recalcula as features de contexto.

## Recuperação dos alvos após a intervenção

Universo principal: 5.000 carteiras, 50 alvos (prevalência 1%). GNN/MLP são embeddings seguidos de IsolationForest. Para esses dois métodos, AP/AUC são média ± desvio-padrão amostral entre seeds 42, 73 e 2026; IF nas features é uma execução com seed 42. O top 50/100 foi zero em cada seed dos encoders, sem esconder variação por média.

| Método | Alvos no top 50 | Alvos no top 100 | AP | ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| GNN + IF | 0/50 | 0/50 | 0,01204 ± 0,00114 | 0,60788 ± 0,03216 |
| MLP + IF | 0/50 | 0/50 | 0,02303 ± 0,00372 | 0,77057 ± 0,04507 |
| Features + IF | 2/50 | 3/50 | 0,01755 | 0,61024 |

No baseline de features, recall@50 = 4%, recall@100 = 6%; precision@50 = 4%, precision@100 = 3%. Um ranking aleatório teria expectativa de 0,5 alvo no top 50 e 1 no top 100. Esses valores esperados não são resultados de um teste de significância; três alvos nesta realização não demonstram superioridade geral.

A ROC-AUC do MLP pode ser relativamente alta e coexistir com top 100 vazio: AUC mede a ordenação global, enquanto os cortes medem a fila curta de revisão. A AP deve ser interpretada com prevalência 1%; os resultados são baixos em termos de priorização dos alvos.

### Comparação antes/depois

Os mesmos 50 alvos designados foram usados para avaliar os rankings antes da alteração. Antes, não eram rótulos de anomalias conhecidas; a comparação mede como os alvos selecionados já se posicionavam.

| Método | AP antes → depois | AUC antes → depois | Alvos top 100 antes → depois |
| --- | --- | --- | --- |
| GNN + IF, média das seeds | 0,01648 → 0,01204 | 0,70085 → 0,60788 | 0 → 0 |
| MLP + IF, média das seeds | 0,02290 → 0,02303 | 0,78837 → 0,77057 | 0,67 → 0 |
| Features + IF | 0,01408 → 0,01755 | 0,60859 → 0,61024 | 1 → 3 |

Para GNN, a posição média dos alvos piorou em 549,12 posições na seed 42, 364,78 na seed 73 e 466,74 na seed 2026. Esses números incluem efeitos do retreino completo; não isolam a causa da mudança. O CSV mudancas_por_grupo.csv registra também doadoras e demais grupos.

### Controles de seleção e efeitos colaterais

No universo das 655 carteiras elegíveis, a prevalência é 50/655 = 7,63%. AUC média após a injeção: GNN 0,36112, MLP 0,47827 e features 0,47815. AP: 0,05728, 0,09508 e 0,08748 respectivamente. A comparação nesse universo também não sustenta boa separação dos alvos pela GNN. Carteiras elegíveis ainda variam em grau/atividade; isto não substitui pareamento exato.

As 228 doadoras também sofreram intervenção e estão marcadas. O avaliador oferece um universo complementar que as exclui. As 50 carteiras-alvo e 228 doadoras tiveram mudanças nos agregados brutos; as 3.747 carteiras com endpoints intactos não tiveram alterações nesses agregados. As 975 isoladas tiveram mudança no preço imputado, pois a mediana dos ativos mudou. Após reajustar o scaler, as features transformadas mudaram nas 5.000 carteiras. Portanto, preservar graus não torna esta intervenção exclusivamente estrutural.

## Aprendizado de ligações versus anomalias

A GNN manteve boa validação de ligações: ROC-AUC média de 0,93721 antes e 0,93632 depois; AP de 0,93867 para 0,93782. Esses resultados coexistem com recuperação zero dos alvos no top 100. A tarefa de aprender conexões não garante que IsolationForest nos embeddings detectará a concentração artificial escolhida.

Uma hipótese para investigar é que concentração em mercados compartilhados possa resultar em representações comuns, em vez de outliers individuais, sob o agregador e detector atuais. Esta rodada não demonstra essa causa. Não declarar a arquitetura inteira inválida nem mudar o cenário para produzir um resultado favorável.

## Verificação, tempo e arquivos

- 21 testes passaram (17 existentes + 4 novos), em 8,107 s reportados pelo unittest. Incluem o caso de capacidade 8 contra exigência 10, aceitação na igualdade e reprodução/invariantes da v2.
- Auditoria posterior recarregou os 14 detectores, reproduziu seus scores e recalculou AP/AUC e recuperados@50/@100. Os 26 arquivos protegidos já versionados em src/, testes anteriores, mapeamentos e requirements permanecem byte a byte iguais ao HEAD inicial. As entradas e artefatos congelados tiveram seus hashes reconferidos.
- Os exemplos de validação/teste são iguais entre modelos/seeds de cada cenário. Antes/depois, os positivos reservados permanecem iguais; os negativos de validação compartilham 3.955 dos 3.956 pares, e os 3.956 negativos de teste são iguais. Não foram calculadas métricas de teste.
- Os 12 loops de treino somaram 103,64 segundos. Esse valor exclui geração, preparação de features, exportação, ajuste do detector e auditoria.
- Artefatos completos: data/processed/sintetico/piloto_v2_8101_8102/. Há entradas original/injetada, trocas, rótulos, checkpoints, embeddings, detectores, rankings, posições individuais e avaliações.
- Nesta pasta docs/piloto_validacao_v2 ficam cópias pequenas das métricas, protocolo, manifesto, auditoria e figura para compartilhar via Git. Dados e modelos continuam ignorados; requerem compartilhamento separado ou reprodução.

## Reprodução

Na raiz, com a .venv ativada. Os destinos precisam ser novos; a pasta v2_8101_8102 já está ocupada.

```bash
python -B -m unittest discover -s tests -v
python -B -m experimentos.piloto_sintetico.gerar --protocolo experimentos/piloto_sintetico/protocolo_v2.json --input data/processed/dev --split data/processed/dev/gnn_seed_42/divisao.pt --output data/processed/sintetico/piloto_v2_reproducao
python -B -m experimentos.piloto_sintetico.executar --run data/processed/sintetico/piloto_v2_reproducao
python -B -m experimentos.piloto_sintetico.avaliar --run data/processed/sintetico/piloto_v2_reproducao
python docs/piloto_validacao_v2/plotar_resumo.py
```

O último comando redesenha a figura a partir da cópia de metricas.csv desta pasta de documentação. O arquivo resultado_piloto_v2.md contém a interpretação desta execução; o relatório automático de cada reprodução fica em avaliacao/relatorio.md.

## Conclusão e continuidade

O piloto de desenvolvimento foi concluído com resultado desfavorável à hipótese de boa recuperação deste padrão pela GNN + IsolationForest atual. Integridade e execução passaram; eficácia na tarefa sintética não foi demonstrada.

Não se executaram as realizações reservadas 8201/8202 e 8301/8302, nem o teste final de ligações. A próxima etapa é revisar o resultado e inspecionar casos reais/controles e exemplos sintéticos para formular hipóteses explicativas. Registrar qualquer novo método antes de avaliá-lo e distinguir desenvolvimento de validação reservada. Depois seguem os controles estruturais, sensibilidade do detector, benchmark/conjunto completo e reprodução/documentação. A avaliação de anomalias reais ainda está pendente.
