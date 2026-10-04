import sys
import os
import pandas as pd
import json

def processar_arquivo_nos(caminho_arquivo, mapeamento_conditions):
    print(f"Processando: {caminho_arquivo}")
    try:
        # Carregar o CSV
        df = pd.read_csv(caminho_arquivo)
        
        # 1. Remover colunas desnecessárias
        colunas_para_remover = ['question', 'closed', 'active', 'archived']
        # Interseção com as colunas existentes para evitar erros caso alguma já tenha sido removida
        colunas_presentes = [col for col in colunas_para_remover if col in df.columns]
        df = df.drop(columns=colunas_presentes)
        
        # 2. Modificar as colunas de data para uma única coluna de duração em segundos
        if 'market_start_time' in df.columns and 'market_end_time' in df.columns:
            # Converter strings para objetos datetime
            df['market_start_time'] = pd.to_datetime(df['market_start_time'])
            df['market_end_time'] = pd.to_datetime(df['market_end_time'])
            
            # Calcular a duração total do mercado em segundos
            df['duracao_mercado_segundos'] = (df['market_end_time'] - df['market_start_time']).dt.total_seconds()
            
            # Remover as colunas de data originais
            df = df.drop(columns=['market_start_time', 'market_end_time'])
            
        # 3. Aplicar mapeamento de ID para inteiros (Integração com a GNN)
        if mapeamento_conditions and 'condition_id' in df.columns:
            df['condition_node_id'] = df['condition_id'].map(mapeamento_conditions)
            
            # Reorganizar as colunas para o ID inteiro ficar no começo
            cols = ['condition_node_id', 'condition_id'] + [c for c in df.columns if c not in ['condition_node_id', 'condition_id']]
            df = df[cols]

        # Salvar o resultado
        caminho_saida = caminho_arquivo.replace('.csv', '_processado.csv')
        df.to_csv(caminho_saida, index=False)
        print(f"Sucesso! Arquivo salvo em: {caminho_saida}\n")
        
    except FileNotFoundError:
        print(f"Erro: O arquivo '{caminho_arquivo}' não foi encontrado.\n")
    except Exception as e:
        print(f"Erro inesperado ao processar '{caminho_arquivo}': {e}\n")

if __name__ == "__main__":
    arquivos_csv = sys.argv[1:]
    
    if not arquivos_csv:
        print("Nenhum arquivo fornecido.")
        print("Uso correto: python processar_nos.py nos_1.csv nos_2.csv ...")
        sys.exit(1)
        
    # Tenta carregar o mapeamento gerado anteriormente para sincronizar os IDs
    mapeamento_conditions = None
    if os.path.exists('mapeamento_conditions.json'):
        with open('mapeamento_conditions.json', 'r') as f:
            mapeamento_conditions = json.load(f)
        print("Mapeamento de conditions carregado com sucesso!\n")
    else:
        print("Aviso: 'mapeamento_conditions.json' não encontrado. Os IDs inteiros não serão gerados.\n")

    # Processar cada arquivo passado como argumento
    for arquivo in arquivos_csv:
        processar_arquivo_nos(arquivo, mapeamento_conditions)