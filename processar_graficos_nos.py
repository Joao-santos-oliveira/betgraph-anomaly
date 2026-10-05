"""
Processa os CSVs de nos de evento (mercados) exportados do Dune.

IMPORTANTE: rode este script DEPOIS de 'processar_arestas.py', ja que ele
depende de 'mapeamento_conditions.json' (gerado por aquele script) para
atribuir o mesmo id inteiro de mercado usado nas arestas.

Para cada CSV de entrada:
  - Mantem 'question' sem transformar (serve para consulta manual depois)
  - Extrai 'categoria_principal' de 'tags' (primeira palavra do array --
    o unico pedaco confiavel, ja que o delimitador entre tags nao e
    preservado de forma consistente no CSV exportado pelo Dune)
  - Marca 'eh_mercado_recorrente' = True quando 'tags' contem "Recurring"
    (mercados tipo "Bitcoin above X" / "Ethereum above X" criados de hora
    em hora, dominados por bots -- mesma familia dos mercados "Up or Down"
    ja excluidos no SQL, so que com outro texto na pergunta)
  - Checa se 'active' e redundante com 'closed', e se 'archived' tem
    variancia -- so remove o que de fato nao carregar informacao
  - Calcula 'duracao_mercado_segundos' a partir de market_start/end_time
  - Aplica o mapeamento de condition_id -> condition_node_id
  - Valida o resultado antes de salvar

Uso:
    python processar_nos.py data/raw/betgraph_nos_evento_16set.csv [outros.csv ...]
"""

import sys
import os
import json
import pandas as pd


ARQUIVO_MAPEAMENTO_CONDITIONS = "mapeamento_conditions.json"


def carregar_json(caminho):
    if os.path.exists(caminho):
        with open(caminho, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def checar_redundancia_status(df):
    """Verifica se active/closed/archived carregam informacao de fato,
    e imprime o que encontrar. So reporta -- a decisao continua sendo sua.
    """
    print("Checagem de redundância das colunas de status:")

    if "active" in df.columns and "closed" in df.columns:
        sao_opostas = (df["active"] == ~df["closed"]).all()
        print(f"  - 'active' é sempre o oposto de 'closed'? {sao_opostas}")
        if sao_opostas:
            print("    -> 'active' é redundante; pode ser removida mantendo só 'closed'.")

    if "archived" in df.columns:
        contagem = df["archived"].value_counts(dropna=False)
        print(f"  - distribuição de 'archived': {contagem.to_dict()}")
        if df["archived"].nunique() <= 1:
            print("    -> 'archived' tem um único valor neste recorte; não carrega informação, pode remover.")

    if "outcome_status" in df.columns:
        contagem = df["outcome_status"].value_counts(dropna=False)
        print(f"  - distribuição de 'outcome_status': {contagem.to_dict()}")

    print()


def extrair_categoria_tags(df):
    """'tags' chega como texto de array (ex.: '[Sports baseball MLB Games]'),
    mas o delimitador original entre os elementos nao e preservado de forma
    confiavel no CSV exportado (a maioria das linhas nao tem virgula nenhuma,
    so ~1% tem). Por isso extraimos so a categoria principal (primeira
    palavra, que e sempre um unico token, tipo 'Sports'/'Politics'/'Crypto'),
    e marcamos mercados recorrentes via busca por substring no texto bruto.
    """
    if "tags" not in df.columns:
        return df

    def primeira_palavra(s):
        if pd.isna(s):
            return "Desconhecida"
        texto = s.strip("[]")
        partes = texto.split()
        return partes[0] if partes else "Desconhecida"

    df["categoria_principal"] = df["tags"].apply(primeira_palavra)
    df["eh_mercado_recorrente"] = df["tags"].fillna("").str.contains("Recurring")
    return df


def validar_dataframe(df, caminho_arquivo):
    avisos = []
    n = len(df)

    if "condition_node_id" in df.columns:
        n_nulos = df["condition_node_id"].isna().sum()
        if n_nulos > 0:
            avisos.append(
                f"  - condition_node_id: {n_nulos}/{n} linhas sem valor (NaN). "
                f"Isso indica que 'processar_arestas.py' não rodou antes, "
                f"ou foi rodado sobre um recorte diferente deste arquivo."
            )

    if "duracao_mercado_segundos" in df.columns:
        # Depois do patch, durações negativas não aparecem mais como valor
        # negativo -- viram NaN. Esta checagem é um alarme de segurança:
        # se ainda aparecer negativa aqui, algo no patch quebrou.
        n_negativas = (df["duracao_mercado_segundos"] < 0).sum()
        if n_negativas > 0:
            avisos.append(
                f"  - ALERTA: {n_negativas} linhas ainda com duracao_mercado_segundos "
                f"negativa -- esperado era 0, o patch de 'duracao_mercado_valida' "
                f"deveria ter zerado isso. Investigue."
            )

    if "duracao_mercado_valida" in df.columns:
        n_invalidas = (~df["duracao_mercado_valida"]).sum()
        avisos.append(
            f"  - {n_invalidas}/{n} mercados com duração não confiável "
            f"(duracao_mercado_segundos = NaN, duracao_mercado_valida = False). "
            f"Dados do mercado preservados, só a duração é ausente."
        )

    if "categoria_principal" in df.columns:
        n_categorias = df["categoria_principal"].nunique()
        avisos.append(f"  - {n_categorias} categorias principais distintas (informativo).")

    if "eh_mercado_recorrente" in df.columns:
        n_recorrentes = df["eh_mercado_recorrente"].sum()
        avisos.append(
            f"  - {n_recorrentes}/{n} mercados marcados como recorrentes "
            f"({n_recorrentes / n:.1%}). Ainda NÃO foram removidos -- "
            f"use 'filtrar_recorrentes.py' para isso depois."
        )

    if avisos:
        print(f"Validação de '{caminho_arquivo}':")
        for a in avisos:
            print(a)
        print()
    else:
        print(f"Validação de '{caminho_arquivo}': nenhum problema encontrado.\n")


def processar_arquivo_nos(caminho_arquivo, conditions_map):
    print(f"Processando: {caminho_arquivo}")
    try:
        df = pd.read_csv(caminho_arquivo)
        n_linhas_original = len(df)

        # 'question' é mantida sem transformar -- serve para inspeção manual
        # dos casos que a GNN apontar depois.

        checar_redundancia_status(df)

        colunas_para_remover = []
        if "active" in df.columns and "closed" in df.columns:
            if (df["active"] == ~df["closed"]).all():
                colunas_para_remover.append("active")
        if "archived" in df.columns and df["archived"].nunique() <= 1:
            colunas_para_remover.append("archived")
        if colunas_para_remover:
            df = df.drop(columns=colunas_para_remover)
            print(f"  Colunas removidas por redundância/falta de variância: {colunas_para_remover}\n")

        df = extrair_categoria_tags(df)

        if "market_start_time" in df.columns and "market_end_time" in df.columns:
            df["market_start_time"] = pd.to_datetime(df["market_start_time"])
            df["market_end_time"] = pd.to_datetime(df["market_end_time"])
            df["duracao_mercado_segundos"] = (
                df["market_end_time"] - df["market_start_time"]
            ).dt.total_seconds()
            # Duração negativa = market_end_time não confiável para este
            # mercado (comum em prop bets de sub-evento, ex.: stats de
            # jogador, vencedor de mapa em esports -- o end_time parece ser
            # um template padrão do tipo de aposta, não o fim real do jogo).
            # Em vez de descartar o mercado inteiro, marcamos a duração como
            # ausente e sinalizamos isso numa coluna própria, para não
            # esconder o problema nem perder o resto dos dados do mercado.
            df["duracao_mercado_valida"] = df["duracao_mercado_segundos"] >= 0
            df.loc[~df["duracao_mercado_valida"], "duracao_mercado_segundos"] = pd.NA
            df = df.drop(columns=["market_start_time", "market_end_time"])

        if conditions_map and "condition_id" in df.columns:
            df["condition_node_id"] = df["condition_id"].map(conditions_map)
            cols = ["condition_node_id", "condition_id"] + [
                c for c in df.columns if c not in ["condition_node_id", "condition_id"]
            ]
            df = df[cols]

        if len(df) != n_linhas_original:
            print(
                f"  AVISO: número de linhas mudou de {n_linhas_original} "
                f"para {len(df)} durante o processamento!"
            )

        validar_dataframe(df, caminho_arquivo)

        caminho_saida = caminho_arquivo.replace(".csv", "_processado.csv")
        df.to_csv(caminho_saida, index=False)
        print(f"Sucesso! Arquivo salvo em: {caminho_saida}\n")

    except FileNotFoundError:
        print(f"Erro: o arquivo '{caminho_arquivo}' não foi encontrado.\n")
    except Exception as e:
        print(f"Erro inesperado ao processar '{caminho_arquivo}': {e}\n")


if __name__ == "__main__":
    arquivos_csv = sys.argv[1:]

    if not arquivos_csv:
        print("Nenhum arquivo fornecido.")
        print("Uso correto: python processar_nos.py nos_1.csv nos_2.csv ...")
        sys.exit(1)

    conditions_map = carregar_json(ARQUIVO_MAPEAMENTO_CONDITIONS)
    if conditions_map is not None:
        print("Mapeamento de conditions carregado com sucesso!\n")
    else:
        print(
            "AVISO: 'mapeamento_conditions.json' não encontrado. "
            "Rode 'processar_arestas.py' primeiro -- sem ele, "
            "'condition_node_id' não será gerado.\n"
        )

    for arquivo in arquivos_csv:
        processar_arquivo_nos(arquivo, conditions_map)