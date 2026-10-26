# Inspeção preliminar dos casos reais

Esta é uma leitura descritiva assistida, não revisão cega independente nem confirmação de fraude. A seleção foi congelada antes de abrir fichas: união top10 de quatro métodos, com mediana das posições das seeds para GNN/MLP. Foram selecionados25casos,15controles e10aleatórios;10casos não encontraram controle dentro do limite definido. Não tratar controles ou aleatórios como normais confirmados.

## Evidências observadas

| Caso | Mercados / volume USD | Controle | Mercados / volume USD | Mercados compartilhados com o parceiro de maior surpresa (caso / controle) |
| --- | --- | --- | --- | --- |
|13537|71 /1689,83|13826|71 /1851,34|39 /7|
|11612|43 /4370,42|5788|42 /4099,42|28 /16|
|18852|68 /543,64|21579|69 /434,37|34 /14|

Cada carteira é comparada ao seu próprio parceiro de maior surpresa; não necessariamente o par de maior contagem. Os pares de controles foram escolhidos por grau e volume, não por compartilhamento ou score.

- **13537 e13379:**39mercados compartilhados. Os10mostrados incluem Counter-Strike e Dota2, com partidas e mapas. Evidência de participação sobreposta; especialização em e-sports é explicação alternativa plausível. Não há demonstração de sincronização ou mesma pessoa.
- **11612 e1440:**28mercados compartilhados. Os10mostrados se referem a tênis em diferentes torneios. Especialização na modalidade ou estratégia semelhante pode explicar parte do padrão. O controle também apresenta16compartilhamentos, mostrando que o comportamento não é exclusivo dos casos destacados.
- **18852 e13537:**34mercados compartilhados; os10exibidos incluem mercados da partida e de jogos/mapas de um mesmo confronto. Mercados distintos podem ser economicamente relacionados: contá-los como evidências independentes superestima a força da interpretação. Esse caso e o primeiro compartilham uma carteira, portanto não são observações independentes.

Nos15pares disponíveis, as medianas foram: grau68vs54, volumeUS$6002,51vs5909,44 e compartilhamento27vs8. O pareamento é aproximado, não perfeito. Não comparamos a mediana dos25casos diretamente à dos15controles, pois os10sem controle mudariam a composição. A diferença de compartilhamento é parcialmente esperada pela seleção dos casos usando o próprio score: não é validação independente nem teste de significância.

Todas as50fichas possuem dados descritivos e subgrafo Mermaid limitado a10mercados compartilhados. A seleção do parceiro é guiada pelo score relacional em todas as fichas, mesmo quando outro método selecionou o caso; esse viés de apresentação está explicitado. As fichas não contêm o tipo de seleção ou scores, mas essa ocultação não torna a interpretação completamente cega.

## O que falta para concluir

1. Dois revisores preencherem separadamente os formulários sem acessar chave/pareamento ou este resumo antes da revisão. Quem já viu os casos aqui não pode ser considerado cego para esses exemplos.
2. Comparar observações, registrar desacordos e a proporção de evidência insuficiente. Não fabricar consenso nem substituir os revisores por este relatório.
3. Agrupar mercados relacionados por evento/modalidade e testar nulos estratificados em um protocolo futuro. A categoria ampla Sports não controla diferenças entre tênis, futebol e e-sports.
4. Obter outros dias e, se a extração permitir, sequência temporal e informação adicional que distinga especialização legítima de coordenação. Os dados atuais não sustentam rótulos de fraude.

Fichas, formulários, chave e balanceamento: `data/processed/validacao_ampliada_v1/casos_reais/`. Arquivos de dados não são versionados. A leitura descritiva está concluída; a revisão humana independente permanece pendente.
