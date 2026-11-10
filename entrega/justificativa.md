# Método principal e comparadores

O método principal do artigo é o **detector relacional direto** da branch main. Nas intervenções compartilhadas da v3, ele recuperou em média 32 dos 50 alvos no top 100 com a concentração estrutural de 60% (de 30 a 34 entre as realizações 30101 e 30201; AP 0,232). O custo é de 0,32 s em 5.000 carteiras, sem treinamento, e cada escore pode ser explicado pela coparticipação entre carteiras.

Três comparadores delimitam o que esse resultado significa:

- **Atributos + IsolationForest** é o único método que captura a alteração de volume (9 de 50, de 6 a 12; AP 0,075), com custo equivalente ao do relacional. Isso mostra que estrutura e atributos detectam padrões diferentes.
- **Híbrido GNN v2** mede o que um embedding aprendido acrescenta ao score estrutural: praticamente nada (31,83 de 50 e os mesmos decoys). O custo é cerca de 78 vezes maior (24,9 s por treino, contra 0,32 s).
- **GNN v3 com atributos de aresta** responde diretamente à pergunta sobre lado, horário e valor: zero alvos recuperados em todos os cenários, com o maior custo (30,1 s por treino). O MLP sem mensagens, com resultado igual, mostra que a troca de mensagens não compensou.

O controle S2 deixa explícito o principal limite do método principal. O relacional coloca em média 29,5 dos 50 decoys no top 100, porque coparticipação forte entre mercados do mesmo evento não é, por si só, anomalia.

As variantes com IsolationForest sobre embeddings, o controle manual e os canais isolados de lado, horário e valor ficam no apêndice. O canal de valor tem a maior AP no cenário combinado (0,079), mas não coloca nenhum alvo no top 100.

Valores e fontes: `tabela_comparativa.md` e `fontes.json`. Os custos são medianas de 3 repetições do benchmark por etapa (`benchmark/resumo.csv`), todas na mesma máquina e na mesma coorte.
