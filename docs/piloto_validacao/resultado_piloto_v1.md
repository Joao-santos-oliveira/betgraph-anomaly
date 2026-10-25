# Resultado da execução do piloto sintético v1

Execução em 25/10/2026. Resultado: **realização inviável antes do treinamento**, conforme o critério de parada do protocolo. A limitação foi demonstrada pelas conexões reservadas de uma das carteiras selecionadas. Nenhum algoritmo existente foi alterado, nenhum modelo foi treinado sobre a intervenção parcial e nenhuma métrica de teste foi calculada.

## Entradas recuperadas e reconstrução

Os CSVs tratados foram localizados em `C:\Users\aliss\Documents\betgraph-anomaly\data\raw`. Os três mapeamentos persistentes dessa cópia têm os mesmos hashes dos mapeamentos deste repositório. Os arquivos originais foram apenas lidos.

Como os artefatos processados anteriores não foram encontrados, as APIs existentes `preparar_features.preparar`, `construir_hetero.construir` e `preparar_gnn.executar` foram executadas sem modificações, usando seed 42 e pasta nova: `data/processed/piloto_entrada_seed42_20261025/`.

Trata-se de uma nova reconstrução determinística, não de recuperação de um checkpoint histórico. As contagens documentadas conferiram:

| Item | Contagem verificada |
| --- | ---: |
| Carteiras de entrada | 27.039 |
| Mercados de entrada | 16.693 |
| Linhas agregadas de entrada | 358.689 |
| Carteiras da amostra | 5.000 |
| Mercados da amostra | 9.189 |
| Linhas agregadas da amostra | 62.328 |
| Pares únicos | 39.562 |
| Pares de contexto | 23.738 |
| Alvos de treino | 7.912 |
| Alvos de validação / teste | 3.956 / 3.956 |
| Carteiras / mercados sem contexto | 975 / 2.098 |

Proveniência, hashes e conferências estão em `reconstrucao_entrada.json`; a saída da preparação está em `reconstrucao_entrada.log`.

## Resultado da geração

O protocolo v1 foi mantido: 50 carteiras, 10 mercados, concentração mínima de 60%, seed de seleção 8101, seed de trocas 8102 e limite de 1.000.000 de propostas. O arquivo `protocolo_v1.json` não foi modificado.

| Diagnóstico | Resultado |
| --- | ---: |
| Mercados candidatos | 133 |
| Carteiras elegíveis pela regra v1 | 659 |
| Carteiras selecionadas | 50 |
| Propostas de troca | 1.000.000 |
| Trocas aceitas antes da interrupção | 126 |
| Carteiras que atingiram a intensidade | 23 de 50 |
| Carteiras com impossibilidade individual comprovada | 1 |

A soma dos graus dos dez mercados-alvo era 473, enquanto os alvos exigiam 263 conexões com esses mercados ao final. A capacidade agregada, portanto, não basta para validar a seleção: há restrições por carteira.

### Prova da inviabilidade

A carteira de ID original `12694` foi selecionada com grau de contexto 16. Pela regra de intensidade, precisa de `ceil(0,60 × 16) = 10` vizinhos entre os dez mercados-alvo.

Entretanto, os pares dessa carteira com os mercados `13` e `441` pertencem aos positivos reservados, fora do contexto. A v1 proíbe inseri-los. Assim, seu limite individual é `10 - 2 = 8` vizinhos permitidos, inferior aos dez exigidos. Ela começou com zero e chegou a oito, atingindo esse limite. Foi a primeira carteira não concluída pelo gerador.

O aumento do número de tentativas não resolve essa incompatibilidade. O diagnóstico completo de todos os 50 alvos está em `diagnostico_inviabilidade_v1.json`.

## Consequências para treino e avaliação

- A tentativa foi registrada como `inviavel`, preservando protocolo, inventário e log das 126 trocas parciais no manifesto.
- Não houve nova seleção de carteiras, mudança de seed, redução de intensidade ou inserção de conexões reservadas.
- Não foram produzidos rankings do cenário parcial. Recuperação top-50/top-100, AP, ROC-AUC e mudança de posição **não se aplicam a esta realização interrompida**; não devem ser preenchidas com zero.
- Os módulos de treino e avaliação permanecem implementados e testados, mas não foram executados no piloto real porque a geração não satisfez os critérios obrigatórios.
- As realizações reservadas 8201/8202 e 8301/8302 permanecem sem execução.

Artefatos desta tentativa: `data/processed/sintetico/piloto_v1_8101_8102_02/`. O primeiro diretório `piloto_v1_8101_8102/`, que registrou a ausência inicial das entradas, permanece preservado. Como `data/` é ignorado pelo Git, este relatório e o diagnóstico resumido mantêm o resultado rastreável no repositório.

## Verificação dos códigos e ambiente

Os 17 testes passaram novamente, em 28,934 segundos reportados pelo unittest. A saída está em `testes_retomada.log`. O ambiente Ubuntu WSL/Python 3.12 foi instalado isoladamente em `/home/aliss/.cache/betgraph-piloto-v1/.venv`, sem modificar o ambiente da outra cópia do projeto. `pip check` não identificou dependências inconsistentes; as versões instaladas estão em `ambiente_retomada.txt`.

Os 25 arquivos protegidos continuam byte a byte iguais ao inventário inicial: `preservacao_retomada.json`. Os hashes das entradas originais também foram reconferidos. O status do Git capturado no WSL sinaliza diferenças de finais de linha em arquivos existentes; a comparação ignorando finais de linha retorna zero, o status do Git no Windows não apresenta modificações nesses arquivos e os hashes confirmam que seus bytes não foram alterados durante o trabalho.

## Correção proposta para uma v2

Antes do sorteio de carteiras, acrescentar à elegibilidade a condição necessária:

```text
ceil(concentracao * grau_contexto)
    <= numero_de_mercados_alvo
       - numero_de_mercados_alvo_com_positivo_reservado_da_carteira
```

Manter 50 carteiras, 10 mercados, 60% de concentração, preservação de graus, proibição dos positivos reservados e seeds previamente declaradas. A seleção das carteiras mudará porque seu conjunto elegível será corrigido; registrar isso como **protocolo v2**, com nova pasta e referência explícita ao resultado inviável da v1. Essa condição não garante viabilidade conjunta: doadoras, graus dos mercados e duplicatas ainda podem impedir as trocas.

Essa alteração não foi executada nesta rodada. A v1 permanece congelada e seu resultado desfavorável foi preservado. A correção afeta apenas a elegibilidade do experimento sintético, sem necessidade de alterar GNN, MLP, IsolationForest ou seus códigos existentes.
