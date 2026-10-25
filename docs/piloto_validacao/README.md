# Execução do protocolo — 25/10/2026

Estado atual: implementação validada e piloto v1 executado até a geração. Os dados foram localizados e as entradas reconstruídas; a realização foi declarada **inviável por uma restrição comprovada de conexões reservadas**. Nenhuma métrica de recuperação do piloto foi calculada e o teste reservado não foi avaliado. Consulte [o resultado da execução v1](resultado_piloto_v1.md) antes de retomar. As seções sobre entradas ausentes abaixo preservam o registro da primeira tentativa.

## Implementação entregue

- `experimentos/piloto_sintetico/protocolo_v1.json`: configuração congelada de 50 carteiras, 10 mercados, concentração de 60% e seeds 8101/8102.
- `gerar.py`: cópias independentes, trocas que preservam graus, blocos de múltiplos outcomes mantidos por mercado, manifesto e diagnóstico de inviabilidade.
- `executar.py`: chama diretamente as funções existentes de preparação, treinamento e ranking, exclusivamente na condição de contexto. Registra alterações das features brutas e transformadas e comparação dos negativos antes/depois.
- `avaliar.py`: recuperação top-50/top-100, precisão/recall, AP, ROC-AUC, empates e posições antes/depois. Separa os universos completo, sem doadoras e elegível, com diagnósticos por grupo.
- `comum.py`: validação do protocolo, hashes, proteção dos originais e verificação dos artefatos congelados.
- `tests/test_piloto_sintetico.py`: nove testes novos, sem modificar os testes anteriores.

O código dos algoritmos em `src/`, os cinco arquivos de testes anteriores, os mapeamentos JSON e `requirements.txt` permanecem intactos. O inventário inicial está em `../piloto_hashes_originais.json`; a comparação final está em `preservacao.json`.

## Verificação executada

Comando: `python -B -m unittest discover -s tests -v`.

Resultado: **17 testes aprovados**, sendo oito anteriores e nove novos. Tempo reportado pelo unittest: 17,845 segundos, excluindo preparação do ambiente e importações iniciais. Saída completa em `testes.log`.

Os novos testes verificam regras e seeds, detecção de alteração dos arquivos, bloqueio por entradas ausentes, rejeição de destinos existentes, reprodutibilidade das trocas, graus e partições, preservação dos atributos e dos originais, inviabilidade sem relaxamento, CSVs inconsistentes, features sem arestas ocultadas, métricas e empates, execução dos três métodos e controle sem intervenção com igualdade exata dos rankings e exemplos de avaliação.

A integração utilizou exclusivamente fixtures pequenas e duas épocas. Esses números não representam resultados do piloto de 50 carteiras. A configuração de produção permanece com 100 épocas e seeds 42, 73 e 2026.

Os testes foram executados no Ubuntu WSL/Python 3.12 com um ambiente temporário isolado. Foram instalados PyTorch 2.14.1+cpu, PyG 2.8.1, NumPy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1 e joblib 1.6.0, com as dependências transitivas limitadas pelo `requirements.txt` existente. Esse ambiente temporário não ficou disponível após a execução; a retomada precisa de um ambiente de pesquisa com essas dependências. Uma tentativa anterior de instalação dentro do OneDrive foi interrompida por lentidão e não deve ser usada como ambiente validado.

## Tentativa com os dados previstos pelo protocolo

Foi executado o gerador com a entrada `data/processed/dev/` e a divisão `gnn_seed_42/divisao.pt`. O gerador rejeitou a execução porque faltam:

1. `data/processed/dev/user.csv`;
2. `data/processed/dev/event.csv`;
3. `data/processed/dev/edge.csv`;
4. `data/processed/dev/gnn_seed_42/divisao.pt`.

Registro preservado em `data/processed/sintetico/piloto_v1_8101_8102/manifesto_injecao.json`, com `status=bloqueado` e `teste_avaliado=false`. Protocolo e inventário de hashes da tentativa estão na mesma pasta. Como `data/` é ignorado pelo Git, esse registro precisa ser compartilhado separadamente; esta documentação mantém o motivo do bloqueio no repositório.

Não foram inventados dados substitutos, regenerada uma divisão diferente ou treinados modelos reais sem as entradas exigidas.

## Retomada

A sequência abaixo documenta a retomada inicialmente planejada. A pasta `_02` já foi usada com entradas reconstruídas e contém a realização inviável; não repetir os comandos com esse destino. Para uma próxima realização, registrar antes um protocolo novo ou uma realização adicional explicitamente justificada. A correção proposta da elegibilidade está no relatório v1.

Comandos históricos de referência:

```bash
python -B -m unittest discover -s tests -v
python -B -m experimentos.piloto_sintetico.gerar --protocolo experimentos/piloto_sintetico/protocolo_v1.json --input data/processed/dev --split data/processed/dev/gnn_seed_42/divisao.pt --output data/processed/sintetico/piloto_v1_8101_8102_02
python -B -m experimentos.piloto_sintetico.executar --run data/processed/sintetico/piloto_v1_8101_8102_02
python -B -m experimentos.piloto_sintetico.avaliar --run data/processed/sintetico/piloto_v1_8101_8102_02
```

Se os arquivos estiverem em outra pasta, ajustar somente `--input` e `--split`, mantendo a configuração científica. O gerador exige correspondência completa entre CSVs e divisão. Se a seleção ou as trocas forem inviáveis, registrar esse resultado e não iniciar os treinos nem relaxar o cenário.

O treino recebe apenas representações e conexões. Os rótulos sintéticos são lidos pelo avaliador depois da geração dos rankings. As realizações reservadas 8201/8202 e 8301/8302 não foram executadas.
