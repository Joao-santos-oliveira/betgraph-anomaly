"""
Gera os nos de usuario (carteiras) do grafo, agregando o CSV final de
arestas por 'wallet'.

IMPORTANTE: rode DEPOIS de 'processar_graficos_arestas.py',
'processar_graficos_nos.py' e 'filtrar_recorrentes.py' -- usa o arquivo
'..._arestas_..._final.csv' como entrada.

Para cada carteira, calcula:
  - volume_total_usd        : soma de 'volume_usd' em todas as arestas
  - n_trades_total           : soma de 'n_trades' em todas as arestas
  - n_mercados_distintos     : quantos 'condition_node_id' diferentes ela tocou
  - n_outcomes_distintos     : quantos 'token_outcome_id' diferentes ela tocou
  - ticket_medio_geral       : volume_total_usd / n_trades_total
  - preco_medio_ponderado    : media de 'preco_medio' ponderada por volume_usd
  - primeira_atividade       : menor timestamp de primeiro_trade entre as arestas
                                (antes de este virar duracao_trade_segundos --
                                 ver nota abaixo)
  - duracao_total_segundos   : soma de 'duracao_trade_segundos' (informativo,
                                 nao e um "tempo de atividade" real, so a soma
                                 das duracoes de cada aresta)

  - primeira_atividade       : menor valor de 'primeiro_trade' entre todas
                                as arestas da carteira (precisa que o CSV de
                                arestas final tenha a coluna 'primeiro_trade'
                                preservada por 'processar_graficos_arestas.py')

Uso:
    python processar_graficos_wallets.py data/raw/betgraph_arestas_16set_final.csv
"""

import sys
import pandas as pd
import numpy as np


def preco_medio_ponderado(grupo):
    pesos = grupo["volume_usd"]
    if pesos.sum() == 0:
        return grupo["preco_medio"].mean()
    return np.average(grupo["preco_medio"], weights=pesos)


def gerar_nos_wallet(caminho_arestas):
    print(f"Lendo: {caminho_arestas}")
    df = pd.read_csv(caminho_arestas)

    colunas_necessarias = ["wallet", "volume_usd", "n_trades", "condition_node_id"]
    faltando = [c for c in colunas_necessarias if c not in df.columns]
    if faltando:
        print(f"Erro: colunas esperadas não encontradas no arquivo: {faltando}")
        sys.exit(1)

    print(f"{len(df)} arestas, {df['wallet'].nunique()} carteiras distintas.\n")

    if "primeiro_trade" in df.columns:
        df["primeiro_trade"] = pd.to_datetime(df["primeiro_trade"])
    else:
        print(
            "AVISO: coluna 'primeiro_trade' não encontrada -- "
            "'primeira_atividade' não será calculada. Confirme se o CSV de "
            "entrada veio da versão atualizada de 'processar_graficos_arestas.py'."
        )

    agregados = []
    for wallet, grupo in df.groupby("wallet"):
        linha = {
            "wallet": wallet,
            "wallet_id": grupo["wallet_id"].iloc[0] if "wallet_id" in grupo.columns else None,
            "volume_total_usd": grupo["volume_usd"].sum(),
            "n_trades_total": grupo["n_trades"].sum(),
            "n_mercados_distintos": grupo["condition_node_id"].nunique(),
            "n_outcomes_distintos": grupo["token_outcome_id"].nunique() if "token_outcome_id" in grupo.columns else None,
        }
        linha["ticket_medio_geral"] = (
            linha["volume_total_usd"] / linha["n_trades_total"] if linha["n_trades_total"] else 0
        )
        if "preco_medio" in grupo.columns:
            linha["preco_medio_ponderado"] = preco_medio_ponderado(grupo)
        if "duracao_trade_segundos" in grupo.columns:
            linha["duracao_total_segundos"] = grupo["duracao_trade_segundos"].sum()
        if "primeiro_trade" in grupo.columns:
            linha["primeira_atividade"] = grupo["primeiro_trade"].min()

        agregados.append(linha)

    df_wallets = pd.DataFrame(agregados)

    print("Resumo dos nós de usuário gerados:")
    print(f"  - {len(df_wallets)} carteiras")
    print(f"  - volume_total_usd:  min={df_wallets['volume_total_usd'].min():.2f}  "
          f"mediana={df_wallets['volume_total_usd'].median():.2f}  "
          f"max={df_wallets['volume_total_usd'].max():.2f}")
    print(f"  - n_mercados_distintos:  min={df_wallets['n_mercados_distintos'].min()}  "
          f"mediana={df_wallets['n_mercados_distintos'].median():.0f}  "
          f"max={df_wallets['n_mercados_distintos'].max()}")
    print()

    caminho_saida = caminho_arestas.replace("_arestas_", "_wallets_").replace("_final.csv", "_nos.csv")
    df_wallets.to_csv(caminho_saida, index=False)
    print(f"Sucesso! Arquivo salvo em: {caminho_saida}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso correto: python processar_graficos_wallets.py <arestas_final.csv>")
        sys.exit(1)

    gerar_nos_wallet(sys.argv[1])