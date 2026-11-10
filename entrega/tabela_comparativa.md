# Tabela comparativa

### Métodos do artigo

| Método | Branch | Rec@100 S1 combinado | Rec@100 S3 estrutura 60% | Rec@100 S3 volume ×10 | AP S1 combinado | AP S3 estrutura 60% | AP S3 volume ×10 | Decoys S2 ↓ | Tempo por execução | Pico de RAM |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Relacional direto (A) | main | 0,50 (0–1) | 32,00 (30–34) | 0,50 (0–1) | 0,0343 | 0,2319 | 0,0228 | 29,50 (25–34) | 0,32 s | 470 MiB |
| Atributos + IsolationForest (E_features) | main | 0,00 | 0,00 | 9,00 (6–12) | 0,0130 | 0,0133 | 0,0745 | 2,00 (1–3) | 0,26 s | 467 MiB |
| Híbrido GNN v2 (E_v2) | v2 | 0,50 (0–1) | 31,83 (29–34) | 0,50 (0–1) | 0,0348 | 0,2325 | 0,0226 | 29,00 (24–34) | 24,85 s (24,4–25,4) | 655 MiB |
| GNN v3, score de pares (C) | v3 | 0,00 | 0,00 | 0,00 | 0,0300 | 0,0465 | 0,0246 | 0,00 | 30,14 s (29,6–30,9) | 730 MiB |

**Como ler.** Rec@100 é o número de alvos artificiais, de 50, nas 100 primeiras posições de 5.000 carteiras (prevalência de 1%). Decoys S2 é o número de decoys, de 50, no top 100; quanto menor, melhor. Os valores são médias sobre as realizações 30101 e 30201 e, nas redes, sobre as seeds 42, 73 e 2026; entre parênteses, o mínimo e o máximo quando diferem. AP é a precisão média, na mesma agregação.

**Custo.** Tempo e pico de RAM de uma execução do método na coorte de 5.000 carteiras, na mesma máquina, sem a preparação compartilhada dos dados. Nas redes, média entre as três seeds.

**Fontes.** Métricas: `resultados/v3/metricas.csv` no commit `a67bcb5` (SHA-256 `70c8b10f49c2…`). Custos: ver `fontes.json`.

### Apêndice: variantes, controle e canais isolados da v3

| Método | Branch | Rec@100 S1 combinado | Rec@100 S3 estrutura 60% | Rec@100 S3 volume ×10 | AP S1 combinado | AP S3 estrutura 60% | AP S3 volume ×10 | Decoys S2 ↓ | Tempo por execução | Pico de RAM |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GNN v3 + IsolationForest (C_IF) | v3 | 0,17 (0–1) | 0,00 | 3,83 (1–6) | 0,0115 | 0,0125 | 0,0409 | 0,17 (0–1) | — | — |
| MLP sem mensagens (D) | v3 | 0,00 | 0,00 | 0,00 | 0,0338 | 0,0524 | 0,0263 | 0,00 | 6,25 s (6,2–6,4) | 578 MiB |
| MLP + IsolationForest (D_IF) | v3 | 0,00 | 0,33 (0–1) | 3,83 (0–9) | 0,0272 | 0,0458 | 0,0551 | 0,50 (0–1) | — | — |
| Manual (B) | v3 | 0,00 | 0,00 | 0,00 | 0,0454 | 0,0829 | 0,0211 | 0,00 | — | — |
| Canal isolado: lado (outcome) (O) | v3 | 0,00 | 0,00 | 0,00 | 0,0345 | 0,0595 | 0,0210 | 0,00 | — | — |
| Canal isolado: horário (T) | v3 | 0,00 | 0,00 | 0,00 | 0,0405 | 0,0539 | 0,0213 | 0,00 | — | — |
| Canal isolado: valor (V) | v3 | 0,00 | 0,00 | 0,00 | 0,0793 | 0,0705 | 0,0204 | 0,00 | — | — |
