# Auditoria e reformulação dos gráficos

Versão de publicação de G1–G6 e S1–S4. Os originais em `entrega/graficos/` ficaram intactos. Todos os valores vêm dos mesmos arquivos de `resultados/artigo/` e `benchmark/`; nenhuma métrica oficial foi recalculada para substituir outra. Comparação lado a lado, com alternância para tons de cinza: `comparacao.html`.

**Regerar:** `make graficos-publicacao` (ou `python -m experimentos.conteudo.graficos_publicacao`). Saída determinística; a geração aborta se alguma validação falhar.

## Problemas transversais da versão anterior

1. **Fonte pequena demais na impressão.** As figuras largas tinham 980 px com texto de 12–13 px. Reduzidas a 7,0 pol. (largura de página), o texto cai para cerca de 6,5 pt. As de coluna tinham 720 px e, a 3,5 pol., ficavam com cerca de 4,6 pt, abaixo do mínimo usual de 6–7 pt. **Correção:** cada figura agora é desenhada no tamanho real (672 px = 7,0 pol.; 336 px = 3,5 pol.; 442 px = 4,6 pol. para G3 e S2), com texto de 9–10,5 px (6,8–7,9 pt).
2. **PNG com resolução insuficiente.** Eram 3× de 96 dpi (≈ 290 dpi no tamanho de impressão). **Correção:** 600 dpi, mais PDF e SVG vetoriais.
3. **Painéis sem identificação.** **Correção:** letras (a, b, c) para citar no texto.
4. **Paleta.** A paleta anterior passava no teste de daltonismo, mas o verde-água (slot 3) tinha contraste de 2,74:1. A paleta proposta no roteiro foi testada com `validate_palette.js` (todos os pares, modo claro):

   | Conjunto | Resultado | Decisão |
   |---|---|---|
   | Verde `#16A34A` × vermelho `#DC2626` | **FALHA**: ΔE 5,0 para deuteranopia | Vermelho trocado por `#B91C1C`: ΔE 11,0, aprovado |
   | Âmbar `#F59E0B` | contraste 2,09:1 com o branco | Trocado por `#D97706` (≥ 3:1) |
   | Azul `#2563EB` × âmbar `#D97706` × ardósia `#64748B` | ΔE ≥ 16,8 em daltonismo; ardósia sem croma de propósito (neutro de ablação) | Mantido |

   Além da cor, cada série tem traço e marcador próprios: o teste em tons de cinza confirma que todas continuam distinguíveis.

## Diagnóstico e notas por figura

Notas de 0 a 10 em cinco critérios: **C**lareza, **E**stética, **R**igor estatístico, **L**egibilidade e **A**dequação acadêmica. A nota geral é a média.

| Figura | Nota original (C/E/R/L/A → média) | Problemas identificados | Modificações | Nota final |
|---|---|---|---|---|
| G1 recuperação | 7/7/6/5/6 → **6,2** | Dispersão por ranking quase invisível (pontos de 6 px a 45%); sem mín.–máx.; eixo sem denominador; fonte de 6,5 pt impressa; legenda dupla de cor e posição | Mín.–máx. empíricos como barras (nomeados como dispersão, não IC); pontos individuais mantidos; eixo "% de 50"; traço e marcador por método; painéis a–d; similaridade como neutro vazado | 9/8/9/9/9 → **8,8** |
| G2 alvos × controles | 7/7/6/4/6 → **6,0** | Barras de erro mín.–máx. com n = 3 escondem os valores; texto a 4,6 pt na coluna; legenda e rótulos diretos repetidos | Os 3 sorteios como marcadores vazados (todos os dados visíveis); média em linha; só rótulos diretos; desenhada a 3,5 pol. | 9/8/9/9/9 → **8,8** |
| G3 ROC | 7/7/6/6/6 → **6,4** | Só curvas por seed sobrepostas; sem curva média, faixa, melhor e pior; zoom em painel separado duplicava os eixos; não dizia que as seeds compartilham os pares de teste | Layout da referência: título com o número de modelos, moldura fechada, média interpolada em azul grosso, faixa ±1 DP (n = 3), modelos em cinza translúcido, melhor (verde, contínuo) e pior (vermelho, tracejado), diagonal preta tracejada, legenda no canto inferior direito na ordem da referência; curvas em degraus. A ampliação até 10% de FPR virou a figura suplementar `G3_curva_roc_detalhe` | 9/9/10/9/10 → **9,4** |
| G4 GNNs | 6/6/**4**/6/6 → **5,6** | **Boxplot com bigodes de 1,5 × IQR e `boxpoints=False`: observações fora dos bigodes omitidas**; barras gastando tinta para 100 e 0 constantes | Todos os 450 pontos de cada caixa visíveis (jitter); regra dos bigodes na legenda; painel b como gráfico de pontos com valores | 9/8/9/9/9 → **8,8** |
| G5 trajetórias | 6/6/7/5/6 → **6,0** | Espaguete de 150 linhas por painel; só a mediana resumia a distribuição; deslocamento pareado não aparecia de forma direta | Faixa interquartil por intensidade; linhas a 10%; novo painel c com o deslocamento pareado de cada carteira, log10 (original / 20), com todos os pontos | 9/8/9/8/9 → **8,6** |
| G6 top 100 | 5/6/7/5/5 → **5,6** | Gráfico de inclinação com 94 linhas cinzas cruzadas escondendo os 12 movimentos de interesse; rótulos dos casos sobre as linhas | Painel a: composição empilhada (94 + 6 / 94 + 6); painel b: halteres só dos que entraram ou saíram, dos casos da T6 e dos dois líderes. Sem Sankey: não há fluxo entre carteiras | 9/8/9/9/9 → **8,8** |
| S1 acumulada | 8/7/7/6/7 → **7,0** | Só 10 e 20 mercados (4 omitido); curvas amostradas em 160 pontos e ligadas por retas, não degraus | Três painéis (4, 10, 20); degraus exatos em todos os k, sem amostragem; rótulos de corte sem colisão | 9/8/10/9/9 → **9,0** |
| S2 precisão–revocação | 7/7/6/5/6 → **6,2** | Eixo y iniciado em 0,45 sem aviso; definição do AP não explicitada; legenda com 4 entradas em 2 linhas | Eixo 0–1 com quadro ampliado; AP = Σ(Rₙ − Rₙ₋₁)·Pₙ na legenda (não trapézio); maior e menor AP no padrão do G3; sem curva média, com a justificativa na legenda | 9/9/10/9/9 → **9,2** |
| S3 treino | 7/6/7/5/6 → **6,2** | Legenda de 6 entradas; não dizia que a parada antecipada nunca disparou | 2 entradas (treino × validação); anotação "melhor época = 100 (teto), paciência 15 não atingida"; perda e AUROC em painéis separados (sem eixo duplo) | 9/8/9/9/9 → **8,8** |
| S4 custos | 7/7/6/6/7 → **6,6** | Mín.–máx. em barras de erro com n = 3, e a 1ª repetição (cache frio) inflava a faixa sem explicação | Barra = mediana; cada repetição como traço; cores por grupo de etapa (dados, treino, scores); cache frio citado na legenda | 9/8/9/9/9 → **8,8** |

## Verificação de integridade (`validacao.json`)

| Figura | Conferência automática | Resultado |
|---|---|---|
| G1 | Médias por ranking contra a T3 (`resumo.csv` e `pareado/resumo.csv`) | diferença máxima 0 p.p.; 168 pontos; 1, 3 ou 9 rankings por célula |
| G2 | Médias dos sorteios contra `pareado/resumo.csv` | diferença 0; 3 sorteios por ponto |
| G3 | AUROC recalculado das curvas contra o oficial | diferença máxima 3,4 × 10⁻⁷; faixa em [0, 1] |
| G4 | 450 posições por caixa; nenhuma omitida | ok |
| G5 | 150 alvos e 150 controles com as 4 intensidades | pares completos |
| G6 | 94 permaneceram, líderes 18753 → 17762 contra `avaliacao_cobertura.json` | ok |
| S1 | Curvas monótonas e terminando em 100% | ok |
| S2 | AP recalculado contra o oficial | diferença < 10⁻⁴ |
| S3 | 300 linhas (3 seeds × 100 épocas), sem suavização | ok |
| S4 | Medianas recalculadas de `etapas.csv` contra `resumo.csv` | ok |

### Transformações aplicadas só à visualização

- **G3, média e faixa:** para cada seed, TPR máximo em cada FPR distinto e interpolação linear numa grade de 1.001 FPR (0 a 1). O DP é amostral (n − 1); a faixa é limitada a [0, 1]. AUROC médio entre seeds = 0,9620 ± 0,0008 (oficial). Área da curva média interpolada = 0,96198, um valor diferente, citado na legenda.
- **G3 e S2, desenho:** vértices a menos de 0,0004 (em unidades de taxa) do último mantido são omitidos só no traço, para o PDF e o SVG não ficarem pesados. AUROC e AP usam todos os 45.814 pares. Os valores completos estão no CSV.
- **G5c:** log10(posição original / posição com 20 mercados), por carteira, derivado das trajetórias.
- **G1, G2 e G4:** deslocamento horizontal entre métodos e jitter dos pontos, para evitar sobreposição. O jitter usa semente fixa.

## Limitações que a figura não resolve

- **G3:** as 3 seeds compartilham divisão, pares de teste e negativos. A faixa mede só a sensibilidade à seed (n = 3), não a incerteza amostral. Por isso a diferença melhor − pior (0,0015) não deve ser lida como diferença entre modelos. O maior DP da TPR é 0,011, perto da origem; no resto da curva a faixa é imperceptível, e isso está dito na legenda em vez de ampliado artificialmente.
- **G3 e S2:** não existe referência de 25 execuções: há 3. Melhor e pior são duas das três curvas.
- **S2:** não há curva média. A média vertical de precisões não corresponde a nenhum AP.
- **Barras com n = 3 (G1 e S4):** mínimo e máximo de 3 valores não são intervalo de confiança, e assim aparecem na legenda.
- **S3:** a melhor época é a 100ª nas três seeds. O treino não convergiu dentro do teto, e isso é um resultado, não um problema do gráfico.
- **Fonte:** Arial/Liberation Sans, presentes no Windows e no Linux. Para combinar com uma fonte serifada do LaTeX, troque `FONTE` em `graficos_publicacao.py`.
