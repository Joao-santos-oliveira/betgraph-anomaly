# Revisão de versionamento e limpeza — 28/10/2026

## Resultado da revisão

Comparação feita desde `19c9f5a214f9268f78f4f6bb5568aaf25cae38cd`, último commit do colega. Após a revisão, o usuário autorizou os commits e a separação das branches: v2 preservada em `feat/gnn-v2-anomalias` (`c0b7390`); v3 em `feat/gnn-v3-anomalias`, criada a partir da v2, com experimento commitado em `deb24e5`. A ficha e esta nota entram em um segundo commit documental. Sem push ou merge; main preservada em `1a1d456`.

O trabalho posterior ao colega está dividido em:

| Estado | Conteúdo | Ação |
|---|---|---|
| Já commitado: `1a1d456` | Correção do piloto, investigação relacional, validação ampliada e preparação da revisão humana | Manter; o colega precisa atualizar sua cópia para conhecer essa etapa |
| Já commitado: `c0b7390` | GNN v2 e comparação experimental | Manter como ablação, mesmo com resultado negativo |
| Commitado: `deb24e5` | Código, testes, documentação e resultados pequenos da GNN v3 | 25 arquivos na branch exclusiva da v3 |
| Commit documental que contém esta nota | Ficha de continuidade atualizada e esta nota | Separado do código experimental, na branch v3 |
| Local e ignorado | Dados, checkpoints, scalers, rankings completos e preparação interrompida | Manter fora do Git; compartilhar ou preservar em armazenamento separado |

## Conteúdo selecionado para os commits

**Bloco experimental v3:**

- `experimentos/gnn_v3/`: seis módulos Python, incluindo o inicializador. Código necessário para reproduzir a experiência.
- `tests/test_gnn_v3.py`: testes específicos da nova implementação.
- `docs/gnn_v3/`: protocolo, congelamentos, resultados, interpretação, métricas completas/resumidas, custos, decisão, auditorias, saída dos testes e proposta de agrupamento.

Não remover `protocolo_v1.md` e `congelamento_v1.json`: documentam a revisão técnica de leitura dos horários e são entradas explícitas do congelamento atual. Não confundir preparação interrompida com uma rodada adicional de escolha de hiperparâmetros.

**Bloco de passagem:**

- `docs/ficha_continuidade_2026-10-28.md`.
- `docs/revisao_versionamento_2026-10-28.md` (esta nota).

Organização adotada:

1. `deb24e5` — `feat: registra tentativa controlada da GNN com atributos de aresta`: 25 arquivos, sem dados/modelos grandes.
2. Commit documental que contém esta nota — `docs: prepara continuidade desde o piloto sintetico`: ficha e revisão de versionamento.

Os arquivos do primeiro commit passaram por `git diff --cached --check`; os hashes do código congelado e entradas foram reconferidos antes do commit. A branch v2 não recebeu nenhum desses arquivos. A branch v3 inclui o histórico da v2 porque depende dela como referência experimental; não é uma segunda cópia de código no mesmo diretório.

O envio ao GitHub permanece pendente. O colega só conseguirá buscar a branch nova remotamente depois do push pelo responsável.

## O que foi removido e o que deve ficar

Removido somente `experimentos/gnn_v3/__pycache__/`, cache regenerável não versionado: 228 bytes. Não havia volume relevante de lixo entre os arquivos examinados. Nenhum arquivo versionado foi removido ou alterado nesta revisão; a ficha já era um arquivo novo pendente.

**Manter no repositório:** resultados negativos, protocolos antigos, diagnósticos de inviabilidade, testes, gráficos científicos, métricas e código de reprodução. Algumas duplicações são deliberadas: resumo para leitura, valores completos para conferência. Os scripts em `docs/gnn_v2/` ainda são referenciados por comandos de consolidação/auditoria; movê-los agora exigiria uma migração explícita de caminhos e proveniência.

`docs/gnn_v2/progresso.json`, apesar do nome, registra as execuções e é copiado pela consolidação. Também está coberto pelo congelamento da v3. Não é um cache que possa ser apagado sem consequência.

**Manter fora do Git:** `data/`, `.venv/`, `__pycache__/`, arquivos `.pyc`, `.pt` e `.joblib`. A `.gitignore` já cobre esses itens; não foi necessário modificá-la. A revisão conferiu o status do Git: os pendentes são código, testes e documentos, não checkpoints/datasets.

Tamanhos locais observados, arredondados pelo comando `du`:

| Pasta | Espaço local aproximado | Observação |
|---|---:|---|
| `docs/gnn_v2/` | 860 KB | Documentação e resultados versionados |
| `docs/gnn_v3/` | 428 KB | Documentação e resultados pequenos a versionar |
| `data/processed/gnn_v2_v1/` | 778 MB | Artefatos grandes, já ignorados |
| `data/processed/gnn_v3_v1/` | 496 MB | Artefatos grandes, já ignorados |
| `data/processed/gnn_v3_v1_preparacao_interrompida/` | 44 MB | Histórico da correção técnica, já ignorado |

A maior parte do espaço está em artefatos locais, não no conteúdo enviado ao GitHub. Não apagamos esses artefatos: ainda são úteis para conferir e reproduzir os resultados. Se houver necessidade futura de liberar disco, preparar um arquivo de backup, conferir seus hashes e documentar como restaurar os caminhos antes de remover a cópia de trabalho.

## Verificações e limite da revisão

Foram reconferidos os SHA-256 do congelamento v3: **133 arquivos anteriores, 11 arquivos de código/protocolo e cinco entradas continuam intactos**. Nenhum treino foi repetido e nenhum critério experimental foi modificado.

Os **40 testes aprovados** e a reprodução dos **81 checkpoints/scores** são evidências da rodada anterior, consultadas nesta revisão; não foram reexecutados para uma alteração apenas documental e remoção de cache. Esta revisão de organização/versionamento não equivale a provar eficácia científica do detector.

O executor v3 congelado exige a branch original `feat/gnn-v2-anomalias` e o HEAD `c0b7390`. **Um novo commit fará sua checagem de HEAD recusar o novo checkout**, mesmo se os arquivos originais permanecerem iguais. Isso é uma restrição de proveniência conhecida, não um motivo para editar os hashes ou apagar o histórico. Para reproduzir a rodada, usar uma cópia separada da base original e copiar os arquivos experimentais conforme `docs/gnn_v3/resultado.md`. Melhorar essa interface de reprodução futuramente exige mudança documentada, preservando a versão usada no experimento.

Não alteramos README/roteiros antigos cobertos pelos hashes apenas para inserir um atalho novo. O documento de entrada atualizado é `docs/ficha_continuidade_2026-10-28.md`; ele informa que `docs/proximos_passos.md` antecede a v3.

## O que mudou na ficha do colega

Agora ela começa exatamente no commit `19c9f5a`, explica a inviabilidade do piloto v1, a correção no piloto v2, o detector relacional, a validação ampliada e as duas tentativas de aprimorar a GNN. Distingue **piloto v2** de **GNN v2**, que são etapas diferentes.

O próximo passo recomendado continua sendo a revisão humana independente das fichas, seguida do controle por mercados relacionados. Os resultados não justificam adotar a GNN v3 como detector principal, nem iniciar automaticamente outra busca de parâmetros.
