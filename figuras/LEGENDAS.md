# Legendas das figuras de grafos (prontas para o LaTeX)

Os nomes dos arquivos são os esperados pelo manuscrito. Todas as figuras têm 17,5 cm de largura, fundo
branco, texto ≥ 8 pt na impressão, ligações sem setas (coparticipação) e identificadores internos (sem
endereço ou identificação civil). Os parâmetros completos de cada figura estão em
`recortes/<figura>/manifesto.json` e `recortes/<figura>/figura.json`.

Ordem de prioridade de interpretação: casos e controles → núcleo das líderes → intervenção → vista global.

---

## `gephi_casos.pdf` — Casos e controles (quatro painéis, 17,5 × 10 cm)

**Legenda proposta.** Vizinhança de quatro carteiras com o mesmo número de mercados, no grafo de cobertura
total do dia e com o score híbrido da seed 42. (A) caso estável 17762 e (B) seu controle 1507, ambos com
171 mercados; (C) caso entrante 16179 e (D) seu controle 53, ambos com 41 mercados. Em cada painel aparecem
o foco (centro, vermelho), seus três parceiros de maior contribuição ao score (roxo), todos os mercados do
foco (anel) e as arestas desses quatro usuários para esses mercados. Arestas dos parceiros para outros
mercados ficam fora da vista; por isso o grau desenhado do parceiro não é seu grau global, informado no
quadro. Mercados verdes são compartilhados com ao menos um parceiro; brancos, só do foco. A faixa externa
indica a modalidade (regra por tags; ver `recortes/gephi_nucleo/correspondencia_mercado_evento.csv` para a
regra de evento). O quadro traz o score do foco, a posição no ranking (contexto → cobertura total) e, por
parceiro, a sobreposição observada/esperada (`resultados/artigo/extra/contribuicoes_casos.csv`). Codificação
e escala são as mesmas nos quatro painéis: o tamanho das carteiras segue log10(1 + grau global) e não é
renormalizado por painel.

| Painel | Foco | Score | Posição ctx → total | Compartilhados ≥1 / com os 3 | Arestas desenhadas |
|---|---:|---:|---|---:|---:|
| A caso estável | 17762 | 84,14 | 4 → 1 | 169 / 148 | 651 |
| B controle | 1507 | 15,77 | 3.599 → 698 | 135 / 0 | 336 |
| C caso entrante | 16179 | 44,89 | 123 → 85 | 41 / 39 | 161 |
| D controle | 53 | 5,64 | 6.864 → 6.063 | 16 / 0 | 60 |

**Leitura.** Com o mesmo número de mercados, os casos têm parceiros que repetem quase todo o seu portfólio
(160 mercados observados contra cerca de 20 esperados em A; 40 contra cerca de 1,2 em C). Nos controles a
sobreposição é parcial e nenhum mercado é comum aos três parceiros.

---

## `gephi_nucleo.pdf` — Núcleo das cinco líderes (dois painéis, 17,5 × 10 cm)

**Legenda proposta.** (A) Sub-rede bipartida das carteiras 17762, 18767, 20748, 18753 e 17487 com os 144
mercados comuns às cinco (149 nós, 720 arestas; cobertura total, seed 42). Os mercados estão agrupados por
modalidade (faixa externa) e, dentro dela, por evento, com uma pequena lacuna entre eventos. A
correspondência mercado → evento foi registrada em `recortes/gephi_nucleo/correspondencia_mercado_evento.csv`:
uma pergunta “A vs B” define o evento; uma pergunta que cita uma equipe de um único evento com as mesmas tags é
atribuída a ele; nos demais casos o evento fica indeterminado (quadrados vazados), o que não significa ausência
de evento. (B) Projeção nas cinco carteiras: o peso de cada um dos dez pares é o número de mercados em comum
calculado com todas as participações do dia (153–160); a espessura é linear nesse intervalo. **Os dois painéis
usam universos diferentes:** restrito à interseção de A, todo par teria peso 144.

Composição de A por modalidade (mercados; eventos identificados; indeterminados): Futebol 60; 30; 19 —
CS2 42; 25; 0 — Dota 2 16; 5; 0 — LoL 12; 5; 0 — Valorant 6; 5; 0 — Rocket League 3; 3; 0 —
Tênis 2; 2; 0 — Luta (MMA) 2; 2; 0 — Outros esports 1; 1; 0.

Pesos de B: 17762–18767 160 · 17762–20748 160 · 17762–17487 160 · 17762–18753 155 · 18767–20748 158 ·
20748–18753 156 · 20748–17487 156 · 18767–18753 154 · 18767–17487 154 · 18753–17487 153.

---

## `gephi_intervencao.pdf` — Intervenção artificial (quatro condições, 17,5 × 10 cm)

**Legenda proposta.** Regra de seleção fixa: sorteio 51011, grupo de ID 0 do teste pareado (alvos 7896,
36905, 10780, 7219 e 9621). (A) grafo original; (B–D) cópias com 4, 10 e 20 novos mercados comuns, as mesmas
usadas no experimento (regeneradas com `experimentos.artigo.pareado.preparar_doses`, mesma seed, mesma divisão
42 e mesmos rótulos; ver `recortes/gephi_intervencao/fonte/manifesto_intervencao.json`). As coordenadas são
fixas nos quatro painéis. O universo visual é a união dos mercados vizinhos das cinco carteiras nas quatro
condições (165 mercados); em cada painel só se desenham as conexões existentes naquela condição. Os mercados da
intervenção ficam no arco magenta do topo, na ordem em que foram adicionados (vazados enquanto ainda não
ligados). Cada troca leva uma aresta do alvo para um mercado novo e devolve outra aresta a uma carteira doadora
fora do grupo (não desenhada; doadoras não são controles). Assim, o recorte tem 158 arestas em todas as
condições e os graus dos alvos ficam constantes (34, 27, 24, 31 e 42), como confirmado no grafo de contexto
completo. A mudança inserida é estrutural: cresce o número de mercados comuns aos cinco (0 → 4 → 10 → 20), sem
aumentar a atividade medida por grau.

---

## `gephi_global.pdf` — Rede bipartida completa do dia (17,5 × 13 cm)

**Legenda proposta.** Rede bipartida completa do dia com cobertura total: 27.039 carteiras (círculos azuis) e
16.693 mercados (quadrados laranja), 43.732 vértices e 229.072 arestas. Cada par carteira–mercado conta como uma
única aresta, independentemente do número de trades ou outcomes, e as direções computacionais do PyG não são
duplicadas. Há 16 componentes; o maior reúne 43.666 nós (99,85%). Nenhum componente foi removido e nada foi
amostrado. A área de cada nó é proporcional a (log10(1 + grau))². As arestas são cinza, com 0,08 pt e 3,5% de
opacidade. Layout ForceAtlas2 do Gephi 0.10.1 (LinLog, scaling 1, gravidade 0,5, Barnes-Hut θ = 1,2, 2.500
iterações mais 150 com ajuste de tamanho, posições iniciais uniformes com seed 42). Os anéis vermelhos marcam as
cinco líderes; a ampliação mostra o aglomerado da 17762 com as mesmas coordenadas (a 20748 fica fora da janela).
Esta é uma vista de contexto: com 229.072 arestas o núcleo satura visualmente, e a proximidade no desenho não
indica relação entre carteiras nem sustenta conclusões. A exportação completa acompanha a figura
(`gephi/gephi_global.gephi`, `recortes/gephi_global/{gephi_global.gexf, nos.csv, arestas.csv}`).

**Limitação de renderização (declarar no texto).** O PDF da figura é totalmente vetorial (≈ 7,5 MB). O
exportador de PDF do próprio Gephi 0.10.1 não conclui com 229 mil arestas (custo quadrático por aresta). Por
isso o projeto nativo traz apenas a prévia SVG do Gephi, e a figura final foi desenhada a partir das
coordenadas calculadas pelo Gephi.
