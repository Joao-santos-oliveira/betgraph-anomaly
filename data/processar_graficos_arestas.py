import sys
import os
import json
import pandas as pd

def carregar_json(caminho):
    """Carrega um arquivo JSON se existir, caso contrário retorna um dicionário vazio."""
    if os.path.exists(caminho):
        with open(caminho, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def salvar_json(dados, caminho):
    """Salva os dados em um arquivo JSON formatado."""
    with open(caminho, 'w', encoding='utf-8') as f:
        json.dump(dados, f, indent=4)

def processar_arquivos_arestas(arquivos_csv):
    # Carregar mapeamentos existentes para manter consistência entre múltiplos arquivos
    wallets_map = carregar_json('mapeamento_wallets.json')
    conditions_map = carregar_json('mapeamento_conditions.json')

    for caminho_arquivo in arquivos_csv:
        print(f"Processando: {caminho_arquivo}")
        try:
            # Carregar o CSV
            df = pd.read_csv(caminho_arquivo)
            
            # 1. Transformar 'token_outcome' em numérico (Yes = 1, No = 0)
            if 'token_outcome' in df.columns:
                df['token_outcome'] = df['token_outcome'].map({'Yes': 1, 'No': 0})
            
            # 2. Modificar as colunas de data para a duração em segundos
            if 'primeiro_trade' in df.columns and 'ultimo_trade' in df.columns:
                df['primeiro_trade'] = pd.to_datetime(df['primeiro_trade'])
                df['ultimo_trade'] = pd.to_datetime(df['ultimo_trade'])
                
                # Diferença de tempo em segundos
                df['duracao_trade_segundos'] = (df['ultimo_trade'] - df['primeiro_trade']).dt.total_seconds()
                df = df.drop(columns=['primeiro_trade', 'ultimo_trade'])
            
            # 3. Atualizar e aplicar mapeamento de 'wallet' -> 'wallet_id'
            if 'wallet' in df.columns:
                for wallet in df['wallet'].dropna().unique():
                    if wallet not in wallets_map:
                        wallets_map[wallet] = len(wallets_map)
                df['wallet_id'] = df['wallet'].map(wallets_map)
            
            # 4. Atualizar e aplicar mapeamento de 'condition_id' -> 'condition_node_id'
            if 'condition_id' in df.columns:
                for cond in df['condition_id'].dropna().unique():
                    if cond not in conditions_map:
                        conditions_map[cond] = len(conditions_map)
                df['condition_node_id'] = df['condition_id'].map(conditions_map)
            
            # Salvar o CSV processado
            caminho_saida = caminho_arquivo.replace('.csv', '_processado.csv')
            df.to_csv(caminho_saida, index=False)
            print(f"Sucesso! Arquivo processado salvo em: {caminho_saida}\n")
            
        except FileNotFoundError:
            print(f"Erro: O arquivo '{caminho_arquivo}' não foi encontrado.\n")
        except Exception as e:
            print(f"Erro inesperado ao processar '{caminho_arquivo}': {e}\n")

    # Salvar os mapeamentos completos atualizados
    salvar_json(wallets_map, 'mapeamento_wallets.json')
    salvar_json(conditions_map, 'mapeamento_conditions.json')
    print("Mapeamentos salvos com sucesso em:")
    print(" - mapeamento_wallets.json")
    print(" - mapeamento_conditions.json")

if __name__ == "__main__":
    arquivos_csv = sys.argv[1:]
    
    if not arquivos_csv:
        print("Nenhum arquivo fornecido.")
        print("Uso correto: python script.py arquivo1.csv arquivo2.csv ... arquivoN.csv")
        sys.exit(1)
        
    processar_arquivos_arestas(arquivos_csv)