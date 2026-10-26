# Investigação da recuperação — protocolo de desenvolvimento

25/10/2026. Registrado antes de executar os diagnósticos adicionais. Entrada: piloto_v2_8101_8102, preservado. Nenhum código original do encoder/detector será alterado. Novos experimentos ficam em data/processed/investigacao_anomalias_v1/.

Perguntas: (1) o top é dominado por atividade individual? (2) o encoder preserva sinal da intervenção com pesos congelados? (3) trocar o detector ou o score recupera alvos? (4) um score explícito de coparticipação é mais adequado ao cenário coletivo?

Desenvolvimento, sem alegação confirmatória: comparar ranks com graus/volume; distâncias e mudança de embedding com pesos fixos, separando features e topologia; aplicar LOF (20 vizinhos, features padronizadas) nos embeddings existentes; avaliar erro médio de reconstrução das arestas de contexto (diagnóstico in-sample, não teste independente); medir compartilhamento de mercados diretamente.

Score relacional proposto antes dos resultados: contar mercados compartilhados por par de carteiras; para pares com pelo menos dois mercados em comum, usar surpresa -log10 da cauda Poisson com lambda = d_u*d_v*sum_m[d_m*(d_m-1)]/[E*(E-1)], aproximação do modelo de configuração bipartido. Score da carteira: média das três maiores surpresas, completando com zeros se houver menos de três parceiros. O corte de suporte 2 exige coparticipação repetida. Usar todos os nós; isolados têm score zero. Não receber rótulos nem IDs de mercados-alvo. A aproximação não produz p-valores calibrados nem prova de coordenação; comparação de ranks é o objetivo.

Controle estrutural simples: mesmo resumo das três maiores contagens compartilhadas, sem corrigir atividade/popularidade. Comparar original/injetado, top-50/100, AP/AUC, universo elegível e doadoras. Persistir todos os resultados, inclusive negativos. Não buscar cortes nem seeds de injeção para favorecer uma variante.

Após desenvolvimento, congelar a variante justificada, parâmetros e fontes antes de gerar/avaliar as realizações 8201/8202 e 8301/8302. Essas realizações usam a mesma família de intervenção e não são novos dias independentes. Relatar inviabilidades sem mudar seeds. Comparar a variante com o pipeline original GNN/MLP/IF nessas realizações. Se não houver recuperação útil, registrar a limitação e reformular as hipóteses experimentais, sem garantir solução geral. O teste de ligações original permanece reservado.
