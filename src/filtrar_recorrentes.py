"""
Remove mercados marcados como 'eh_mercado_recorrente' (tag 'Recurring' --
mercados tipo "Bitcoin above X", criados de hora em hora, dominados por
bots) tanto do CSV de nos de evento quanto do CSV de arestas, usando
'condition_id' para manter os dois arquivos consistentes entre si.

Sem rodar este script, as arestas continuariam apontando para mercados
recorrentes mesmo depois de removidos do arquivo de nos (arestas orfas).

IMPORTANTE: rode DEPOIS de 'processar_arestas.py' e 'processar_nos.py'
(usa os arquivos '..._processado.csv' gerados por eles).

Uso:
    python filtrar_recorrentes.py \
        data/raw/betgraph_nos_evento_16set_processado.csv \
        data/raw/betgraph_arestas_16set_processado.csv
"""

import sys
import pandas as pd


def filtrar(caminho_nos, caminho_arestas):
    df_nos = pd.read_csv(caminho_nos)
    df_arestas = pd.read_csv(caminho_arestas)

    if "eh_mercado_recorrente" not in df_nos.columns:
        print(
            "Erro: 'eh_mercado_recorrente' não existe no arquivo de nós. "
            "Rode a versão atualizada de 'processar_nos.py' primeiro."
        )
        sys.exit(1)

    n_nos_antes = len(df_nos)
    n_arestas_antes = len(df_arestas)

    mercados_recorrentes = set(df_nos.loc[df_nos["eh_mercado_recorrente"], "condition_id"])
    print(f"Mercados recorrentes identificados: {len(mercados_recorrentes)} de {n_nos_antes}")

    df_nos_filtrado = df_nos[~df_nos["eh_mercado_recorrente"]].copy()
    df_arestas_filtrado = df_arestas[~df_arestas["condition_id"].isin(mercados_recorrentes)].copy()

    n_nos_depois = len(df_nos_filtrado)
    n_arestas_depois = len(df_arestas_filtrado)

    print(f"Nós: {n_nos_antes} -> {n_nos_depois}  (-{n_nos_antes - n_nos_depois})")
    print(f"Arestas: {n_arestas_antes} -> {n_arestas_depois}  (-{n_arestas_antes - n_arestas_depois})")

    # Confere se sobrou alguma aresta apontando para mercado removido
    orfas = (~df_arestas_filtrado["condition_id"].isin(set(df_nos_filtrado["condition_id"]))).sum()
    if orfas > 0:
        print(f"AVISO: {orfas} arestas ainda ficaram órfãs após o filtro!")
    else:
        print("OK: nenhuma aresta órfã -- todo condition_id em arestas existe em nós.")

    caminho_nos_final = caminho_nos.replace("_processado.csv", "_final.csv")
    caminho_arestas_final = caminho_arestas.replace("_processado.csv", "_final.csv")

    df_nos_filtrado.to_csv(caminho_nos_final, index=False)
    df_arestas_filtrado.to_csv(caminho_arestas_final, index=False)

    print(f"\nSalvos:\n - {caminho_nos_final}\n - {caminho_arestas_final}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso correto: python filtrar_recorrentes.py <nos_processado.csv> <arestas_processado.csv>")
        sys.exit(1)

    filtrar(sys.argv[1], sys.argv[2])
