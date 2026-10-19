# Resultados — seeds 42, 73 e 2026

Rodada: `data/processed/dev/experimento_seeds_42_73_2026`. As execuções anteriores permanecem preservadas. Foram repetidos os 12 treinos (GNN/MLP × duas condições × três seeds), com o mesmo orçamento e exemplos de validação. Só variam inicialização e dropout. Os 14 rankings foram gerados novamente. Teste final reservado, sem avaliação.

## Métricas de validação

Média ± desvio-padrão amostral de três seeds:

| Features | Modelo | BCE | AP | ROC-AUC |
| --- | --- | --- | --- | --- |
| Dia inteiro | GNN | 0,29715 ± 0,00018 | 0,94468 ± 0,00033 | 0,94605 ± 0,00019 |
| Dia inteiro | MLP | 0,41337 ± 0,00059 | 0,90204 ± 0,00042 | 0,88823 ± 0,00058 |
| Contexto | GNN | 0,31996 ± 0,00165 | 0,93867 ± 0,00051 | 0,93721 ± 0,00044 |
| Contexto | MLP | 0,43197 ± 0,00078 | 0,89343 ± 0,00049 | 0,87673 ± 0,00100 |

Controle por produto dos graus: AP 0,85222 e ROC-AUC 0,78763. A GNN supera os controles nesta tarefa; diferenças de capacidade entre GNN/MLP, negativos uniformes e ausência de ablação estrutural ainda limitam a atribuição do ganho à identidade dos vizinhos.

Todos os treinos atingiram o orçamento de 100 épocas. Os checkpoints selecionados ficaram nas épocas 99 ou 100. As perdas caem ao longo do treinamento, sem divergência sustentada da validação nas curvas; isso não prova ausência de sobreajuste nem convergência. A loss de treino inclui dropout e negativos renovados, e é medida antes do passo; a de validação usa pesos atualizados e dropout desligado. Por isso, validação abaixo de treino não é necessariamente erro.

Loops de treino: 99,84 segundos somados, sem preparação e ranking. GNN de contexto: média 10,57 s/treino. Oito testes passaram. A geração das curvas reproduziu as 12 AUCs dos manifestos e conferiu igualdade exata dos exemplos de validação.

## Como ler a ROC

Arquivos em `curvas/`: `roc_validacao.png/.pdf`, `curvas_aprendizado.png/.pdf`, pontos de cada ROC em CSV, AUCs conferidas e manifesto. A ROC mostra, ao variar o limiar, a fração de conexões observadas recuperadas (TPR) versus a fração de pares negativos classificados como conexão (FPR). Mais perto do canto superior esquerdo é melhor; diagonal tem AUC 0,5. AUC não é porcentagem de acertos.

As linhas de cada modelo quase se sobrepõem porque as seeds apresentam desempenhos próximos. As figuras usam checkpoints escolhidos na própria validação: são diagnóstico de desenvolvimento, não avaliação final independente. Não são curvas de fraude, pois os rótulos representam conexão/ausência de conexão. As features do dia inteiro incluem agregados de conexões ocultadas; a condição de contexto remove essa informação dos agregados de carteiras, mas metadados de mercados continuam transdutivos.

## Estabilidade e ranking

Na GNN de contexto, a interseção entre tops de 50 é 45–47 carteiras, Jaccard 0,8182–0,8868 e Spearman 0,9323–0,9526. Na GNN com features do dia inteiro, são 40–44 carteiras, Jaccard 0,6667–0,7857 e Spearman 0,8028–0,8132. Esses intervalos são mínimos/máximos das três comparações entre seeds, não intervalos de confiança.

Estabilidade não valida anomalias. Os casos ainda precisam de inspeção com controles e visualização dos subgrafos. As 975 carteiras sem contexto continuam no universo; no controle de contexto recebem entradas iguais e não podem ser diferenciadas pelos embeddings disponíveis.

## Escala completa

Contagens conferidas nos CSVs locais: 27.039 carteiras, 16.693 mercados, 358.689 linhas agregadas de arestas e 229.072 pares carteira–mercado únicos. A amostra tem 5.000 carteiras, 9.189 mercados, 62.328 linhas e 39.562 pares únicos. Portanto, o completo tem aproximadamente 5,4 vezes as carteiras e 5,8 vezes os pares. Não multiplicar esses fatores para estimar custo; não usamos matriz densa.

A expansão parece viável, mas ainda não foi executada nesta rodada. Havia aproximadamente 5,52 GiB disponíveis no WSL na consulta; isso varia e não constitui medição do pico da GNN. Próximo procedimento: preparar em pasta separada com `--sample-size 0`, construir grafo e divisão novos com o mesmo protocolo, recalcular features de contexto, medir tempo/memória em uma execução curta e então repetir GNN/MLP com seeds 42, 73 e 2026. Reajustar os preprocessadores conforme os nós permitidos; não reutilizar inadvertidamente os da amostra.

O completo continua sendo o recorte tratado de um dia, não todo o histórico do Polymarket. Expandir aumenta a cobertura e testa robustez à escala, mas não demonstra generalização para outros dias. O teste reservado continua fechado até congelar as decisões experimentais.
