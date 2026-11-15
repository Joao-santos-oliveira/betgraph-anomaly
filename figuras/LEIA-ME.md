# Figuras de grafos (Gephi) do artigo

| Arquivo | Conteúdo |
|---|---|
| `gephi_casos.{pdf,svg,png}` | Casos e controles, 4 painéis |
| `gephi_nucleo.{pdf,svg,png}` | Núcleo das cinco líderes, 2 painéis |
| `gephi_intervencao.{pdf,svg,png}` | Intervenção artificial, 4 condições |
| `gephi_global.{pdf,png}` | Rede bipartida completa do dia |
| `LEGENDAS.md` | Legendas propostas e números de cada figura |
| `gephi/*.gephi` | Projetos nativos do Gephi 0.10.1 (um por painel; mesmas coordenadas da figura) |
| `gephi/*_posicoes.csv` | Coordenadas finais lidas do Gephi |
| `recortes/<figura>/` | Listas de nós e arestas de cada recorte, GEXF, `figura.json` (universo, layout, tamanho e arquivos); a rede global também inclui `manifesto.json` |

PDF vetorial com fontes TrueType embutidas, SVG com texto editável e PNG de 600 dpi só para pré-visualização.

**Fora do Git (> 50 MB, compartilhar à parte ou regenerar):** `gephi_global.svg`,
`gephi/gephi_global_preview.svg`, `recortes/gephi_global/gephi_global.gexf` e `gephi_global_fa2.gexf`. O projeto
`gephi/gephi_global.gephi` (2,9 MB) e as listas `nos.csv`/`arestas.csv` da rede completa estão versionados.

## Como abrir no Gephi

`File > Open` → `gephi/<figura>__<painel>.gephi`. As posições já vêm gravadas. Não rode layouts livres nas
Figs. 2–4, porque os anéis são determinísticos e servem para comparar os painéis. Para a rede global, o
projeto já contém o resultado do ForceAtlas2.

## Reprodução (Windows, a partir da raiz do repositório)

Requisitos: Gephi 0.10.1 em `C:\Program Files\Gephi-0.10.1`, um JDK (`javac`) em
`C:\Program Files\Java\openJdk-24` (ajustável em `visualizacao_gephi/gephi.py`), Python com pandas, numpy,
matplotlib e networkx.

```bash
python -m visualizacao_gephi.figuras casos nucleo intervencao global
python -m unittest tests.test_visualizacao_gephi -v
```

`--sem-gephi` refaz só a composição, sem chamar o Gephi. A Fig. 4 depende de
`recortes/gephi_intervencao/fonte/`, que é gerada no WSL (ambiente com torch/PyG) a partir da rodada do
`make artigo`:

```bash
python -B -m visualizacao_gephi.intervencao_wsl /caminho/para/data/execucoes/<rodada>
```

## Observações de método

- O Gephi 0.10.1 é usado sem interface gráfica (`visualizacao_gephi/java/RenderGephi.java`) para o
  ForceAtlas2, a prévia nativa e o projeto `.gephi`. A composição final (painéis, quadros, legendas e formas
  círculo/quadrado, que o Preview do Gephi não oferece) é desenhada em Matplotlib com as coordenadas gravadas
  pelo Gephi.
- Dois contornos para o modo sem interface do Gephi 0.10.1: o projeto é salvo sobre um arquivo pré-criado
  (a limpeza do `SaveTask` apaga o destino quando ele não existia) e o PDF nativo é desligado para a rede
  global (limitação observada no exportador utilizado).
- Rede global: `gephi/gephi_global_posicoes_fa2.csv` traz as coordenadas brutas do ForceAtlas2. Ao reabrir o
  GEXF, o importador do Gephi reescala as posições: o `.gephi` e `gephi_global_posicoes.csv` são uma reescala
  uniforme exata delas (fator 0,005656 nos dois eixos, mais translação; resíduo < 0,001). A figura normaliza
  as coordenadas, então o desenho é o mesmo.
- Modalidade: regra explícita por tags (`visualizacao_gephi/comum.py::modalidade`). Evento: regra textual
  registrada (`comum.py::eventos`), com categoria “indeterminado” preservada.

As prévias nativas do Gephi são intermediárias regeneráveis e não são versionadas. Os projetos `.gephi`, coordenadas e figuras finais permanecem disponíveis. O roteiro inicial foi consolidado neste guia e em `LEGENDAS.md`.

Em Linux, com pandas, numpy, matplotlib e networkx instalados, as coordenadas já versionadas permitem recompor as figuras sem Gephi: `python -m visualizacao_gephi.figuras casos nucleo intervencao global --sem-gephi`.
