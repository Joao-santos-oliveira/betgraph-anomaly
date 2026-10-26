# Protocolo do piloto sintetico v2 — 25/10/2026

Registrado antes da geracao e dos resultados. A v1 permanece preservada no commit 19c9f5a e em docs/piloto_validacao/resultado_piloto_v1.md. Esta e uma realizacao de desenvolvimento, nao a avaliacao sintetica reservada.

A unica mudanca cientifica em relacao a v1 e a elegibilidade individual, antes do sorteio dos alvos: ceil(0,60 * grau_contexto) deve ser menor ou igual ao numero de mercados-alvo sem positivo reservado para aquela carteira. Conexoes reservadas de treino, validacao e teste continuam proibidas na intervencao. A carteira 12694 e um caso de regressao: exige 10 mercados e tem capacidade 8 na selecao da v1.

Permanecem 50 carteiras, 10 mercados com grau >=25, grau minimo das carteiras 5, concentracao 60%, selecao 8101, trocas 8102 e limite total de 1.000.000 propostas. Seeds do encoder 42/73/2026, IF 42, negativos e orcamento de treino conforme protocolo_v2.json. A selecao muda por corrigir o conjunto elegivel; nao se escolhem novas seeds por resultado. A capacidade individual nao garante trocas conjuntamente viaveis.

Entrada local: data/processed/dev/{user,event,edge}.csv e data/processed/dev/gnn_seed_42/divisao.pt. Saida nova: data/processed/sintetico/piloto_v2_8101_8102/. Nesta maquina as entradas historicas existem; nao e necessario reconstruir a amostra usada no ambiente do colega.

O gerador segue usando trocas bipartidas com preservacao de graus e blocos de outcomes por mercado; os agregados individuais podem mudar e sao recalculados exclusivamente no contexto. Nao e uma intervencao exclusivamente topologica. Carteiras doadoras e nos afetados ficam identificados para a avaliacao complementar.

A v1 continua seguindo sua regra original quando version=v1. A v2 usa elegibilidade_v2.py; versoes desconhecidas sao rejeitadas. Arquivos src/, testes anteriores, mapeamentos, dependencias e dados originais permanecem intactos. Inventarios incluem o protocolo e os testes novos. Nao alterar codigo congelado durante a execucao.

Avanco: testes aprovados; 50/50 alvos atendem a intensidade; graus, IDs, atributos por linha e particoes conferidos. Se a geracao falhar, registrar sem treinar a intervencao parcial. Se passar, executar os tres metodos antes/depois: 12 treinos e 14 rankings; avaliar recuperacao@50/@100, AP, AUC e posicoes, com universos completo, sem doadoras e elegivel. O rótulo indica alvo artificial, sem garantia de normalidade dos demais.

Criterio de conclusao do piloto: resultado rastreavel, integridade verificada e relatorio de todos os metodos, mesmo se a GNN nao recuperar os alvos. Nao escolher hiperparametros nem intensidades depois de observar as metricas desta rodada. As realizacoes reservadas 8201/8202 e 8301/8302, a revisao real e a expansao de escala permanecem etapas posteriores.
