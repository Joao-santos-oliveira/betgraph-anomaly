"""
Processa os CSVs de arestas (apostas) exportados do Dune.

Para cada CSV de entrada:
  - Codifica 'wallet' -> 'wallet_id' (inteiro, mapeamento persistente em JSON)
  - Codifica 'condition_id' -> 'condition_node_id' (inteiro, mapeamento persistente em JSON)
  - Codifica 'token_outcome' -> 'token_outcome_id' (inteiro, mapeamento persistente em JSON)
  - Calcula 'duracao_trade_segundos' a partir de primeiro_trade/ultimo_trade
  - Valida o resultado e avisa sobre qualquer problema antes de salvar

Uso:
    python processar_arestas.py data/raw/betgraph_arestas_16set.csv [outros.csv ...]

Os mapeamentos (mapeamento_wallets.json, mapeamento_conditions.json,
mapeamento_outcomes.json) são lidos e atualizados a cada execução, para
que o mesmo wallet/condition/outcome sempre receba o mesmo id inteiro,
mesmo entre arquivos ou execuções diferentes.

IMPORTANTE: rode este script ANTES de processar os CSVs de nós de evento,
já que o script de nós depende de 'mapeamento_conditions.json' já existir.
"""

import sys
import os
import json
import pandas as pd


MAPEAMENTOS = {
    "wallets": "mapeamento_wallets.json",
    "conditions": "mapeamento_conditions.json",
    "outcomes": "mapeamento_outcomes.json",
}


def carregar_json(caminho):
    """Carrega um arquivo JSON se existir, caso contrário retorna um dicionário vazio."""
    if os.path.exists(caminho):
        with open(caminho, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def salvar_json(dados, caminho):
    """Salva os dados em um arquivo JSON formatado."""
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(dados, f, indent=4, ensure_ascii=False)


def atualizar_mapeamento(mapa, valores):
    """Adiciona ao mapa, na ordem de chegada, qualquer valor ainda não mapeado."""
    for v in valores:
        if v not in mapa:
            mapa[v] = len(mapa)
    return mapa


def validar_dataframe(df, caminho_arquivo):
    """Confere pontos que, se vierem errados, indicam problema no pipeline.
    Não interrompe a execução -- só avisa, porque quem decide o que fazer
    com um aviso é a pessoa rodando o script, não o script.
    """
    avisos = []

    n = len(df)

    for coluna in ["wallet_id", "condition_node_id", "token_outcome_id"]:
        if coluna in df.columns:
            n_nulos = df[coluna].isna().sum()
            if n_nulos > 0:
                avisos.append(
                    f"  - {coluna}: {n_nulos}/{n} linhas ficaram sem valor (NaN) "
                    f"({n_nulos / n:.1%}). Investigue antes de usar este arquivo."
                )

    if "duracao_trade_segundos" in df.columns:
        n_negativas = (df["duracao_trade_segundos"] < 0).sum()
        if n_negativas > 0:
            avisos.append(
                f"  - duracao_trade_segundos: {n_negativas} linhas negativas "
                f"(ultimo_trade antes de primeiro_trade -- não deveria acontecer)."
            )

    if "token_outcome_id" in df.columns:
        n_outcomes = df["token_outcome_id"].nunique()
        avisos.append(f"  - {n_outcomes} outcomes distintos codificados neste arquivo (informativo).")

    if avisos:
        print(f"Validação de '{caminho_arquivo}':")
        for a in avisos:
            print(a)
        print()
    else:
        print(f"Validação de '{caminho_arquivo}': nenhum problema encontrado.\n")


def processar_arquivos_arestas(arquivos_csv):
    wallets_map = carregar_json(MAPEAMENTOS["wallets"])
    conditions_map = carregar_json(MAPEAMENTOS["conditions"])
    outcomes_map = carregar_json(MAPEAMENTOS["outcomes"])

    for caminho_arquivo in arquivos_csv:
        print(f"Processando: {caminho_arquivo}")
        try:
            df = pd.read_csv(caminho_arquivo)
            n_linhas_original = len(df)

            # 1. wallet -> wallet_id
            if "wallet" in df.columns:
                wallets_map = atualizar_mapeamento(wallets_map, df["wallet"].dropna().unique())
                df["wallet_id"] = df["wallet"].map(wallets_map)

            # 2. condition_id -> condition_node_id
            if "condition_id" in df.columns:
                conditions_map = atualizar_mapeamento(conditions_map, df["condition_id"].dropna().unique())
                df["condition_node_id"] = df["condition_id"].map(conditions_map)

            # 3. token_outcome -> token_outcome_id
            #    (NÃO é binário Yes/No -- nos dados reais aparecem "Over", "Under",
            #     nomes de times, etc. Por isso o encoding é automático, a partir
            #     dos valores que realmente existem, igual wallet/condition.)
            if "token_outcome" in df.columns:
                outcomes_map = atualizar_mapeamento(outcomes_map, df["token_outcome"].dropna().unique())
                df["token_outcome_id"] = df["token_outcome"].map(outcomes_map)

            # 4. datas -> duração em segundos, preservando a data bruta do
            #    primeiro trade (necessária depois para calcular a "primeira
            #    atividade" de cada carteira, agregando por wallet). Só
            #    'ultimo_trade' é descartada, já que a duração já resume
            #    a diferença entre as duas.
            if "primeiro_trade" in df.columns and "ultimo_trade" in df.columns:
                df["primeiro_trade"] = pd.to_datetime(df["primeiro_trade"])
                df["ultimo_trade"] = pd.to_datetime(df["ultimo_trade"])
                df["duracao_trade_segundos"] = (
                    df["ultimo_trade"] - df["primeiro_trade"]
                ).dt.total_seconds()
                df = df.drop(columns=["ultimo_trade"])

            # Confere que nenhuma linha sumiu no processo
            if len(df) != n_linhas_original:
                print(
                    f"  AVISO: número de linhas mudou de {n_linhas_original} "
                    f"para {len(df)} durante o processamento!"
                )

            validar_dataframe(df, caminho_arquivo)

            caminho_saida = caminho_arquivo.replace(".csv", "_processado.csv")
            df.to_csv(caminho_saida, index=False)
            print(f"Sucesso! Arquivo processado salvo em: {caminho_saida}\n")

        except FileNotFoundError:
            print(f"Erro: o arquivo '{caminho_arquivo}' não foi encontrado.\n")
        except Exception as e:
            print(f"Erro inesperado ao processar '{caminho_arquivo}': {e}\n")

    salvar_json(wallets_map, MAPEAMENTOS["wallets"])
    salvar_json(conditions_map, MAPEAMENTOS["conditions"])
    salvar_json(outcomes_map, MAPEAMENTOS["outcomes"])
    print("Mapeamentos salvos com sucesso em:")
    print(f" - {MAPEAMENTOS['wallets']}  ({len(wallets_map)} wallets)")
    print(f" - {MAPEAMENTOS['conditions']}  ({len(conditions_map)} mercados)")
    print(f" - {MAPEAMENTOS['outcomes']}  ({len(outcomes_map)} outcomes)")


if __name__ == "__main__":
    arquivos_csv = sys.argv[1:]

    if not arquivos_csv:
        print("Nenhum arquivo fornecido.")
        print("Uso correto: python processar_arestas.py arquivo1.csv arquivo2.csv ...")
        sys.exit(1)

    processar_arquivos_arestas(arquivos_csv)