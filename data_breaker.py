import os
import numpy as np
import pandas as pd

class DatasetDisruptor:
    def __init__(self, seed=None):
        """
        Inicializa o disruptor de datasets com uma seed opcional para controle de aleatoriedade.

        Args:
            seed (int, optional): Seed para o gerador de números aleatórios. Default é None.
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def introduce_missing_values(self, df, missing_fraction=0.1):
        """
        Introduz valores faltantes (NaN) no DataFrame de forma aleatória.

        Args:
            df (pd.DataFrame): O DataFrame original e limpo.
            missing_fraction (float): Fração dos dados que serão convertidos para valores faltantes (máx 20%).

        Returns:
            pd.DataFrame: DataFrame com valores faltantes introduzidos.
        """
        df_broken = df.copy()
        n_total = df.size
        n_missing = int(n_total * min(missing_fraction, 0.2))  # Limitar para 20% de valores faltantes

        # Obter índices aleatórios para colocar valores faltantes
        indices = [(row, col) for row in range(df.shape[0]) for col in range(df.shape[1])]
        missing_indices = np.random.choice(len(indices), n_missing, replace=False)
        for idx in missing_indices:
            row, col = indices[idx]
            df_broken.iat[row, col] = np.nan
        return df_broken

    def introduce_column_missing(self, df, col_missing_fraction=0.1):
        """
        Remove colunas inteiras de valores em uma fração específica.

        Args:
            df (pd.DataFrame): O DataFrame original e limpo.
            col_missing_fraction (float): Fração de colunas a serem completamente preenchidas com NaN (máx 10%).

        Returns:
            pd.DataFrame: DataFrame com algumas colunas contendo 100% de valores faltantes.
        """
        df_broken = df.copy()
        n_cols = df.shape[1]
        n_cols_missing = int(n_cols * min(col_missing_fraction, 0.1))  # Limitar para 10% de colunas faltantes

        # Selecionar colunas aleatórias para quebrar
        if n_cols_missing > 0:
            cols_to_break = np.random.choice(df.columns, size=n_cols_missing, replace=False)
            df_broken[cols_to_break] = np.nan
        return df_broken

    def introduce_row_missing(self, df, row_missing_fraction=0.05):
        """
        Remove linhas inteiras de valores em uma fração específica.

        Args:
            df (pd.DataFrame): O DataFrame original e limpo.
            row_missing_fraction (float): Fração de linhas a serem completamente preenchidas com NaN (máx 5%).

        Returns:
            pd.DataFrame: DataFrame com algumas linhas contendo 100% de valores faltantes.
        """
        df_broken = df.copy()
        n_rows = df.shape[0]
        n_rows_missing = int(n_rows * min(row_missing_fraction, 0.05))  # Limitar para 5% de linhas faltantes

        # Selecionar linhas aleatórias para quebrar
        if n_rows_missing > 0:
            rows_to_break = np.random.choice(df.index, size=n_rows_missing, replace=False)
            df_broken.loc[rows_to_break] = np.nan
        return df_broken

    def add_noise(self, df, noise_fraction=0.02, noise_level=0.05):
        """
        Adiciona ruído gaussiano a uma pequena fração dos valores numéricos do DataFrame.

        Args:
            df (pd.DataFrame): O DataFrame original e limpo.
            noise_fraction (float): Fração dos dados numéricos que receberão ruído (máx 2%).
            noise_level (float): Nível do ruído (desvio padrão da distribuição gaussiana).

        Returns:
            pd.DataFrame: DataFrame com ruído adicionado a alguns valores numéricos.
        """
        df_broken = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        n_total = df[numeric_cols].size
        n_noisy = int(n_total * min(noise_fraction, 0.02))  # Limitar para 2% de valores numéricos com ruído

        # Obter índices aleatórios para adicionar ruído
        indices = [(row, col) for row in df.index for col in numeric_cols]
        noisy_indices = np.random.choice(len(indices), n_noisy, replace=False)
        for idx in noisy_indices:
            row, col = indices[idx]
            original_value = df_broken.at[row, col]
            if pd.notnull(original_value):
                df_broken.at[row, col] += np.random.normal(0, noise_level)
        return df_broken

    def break_dataset(self, df, missing_fraction=0.1, col_missing_fraction=0.1, row_missing_fraction=0.05, noise_fraction=0.02, noise_level=0.05):
        """
        Aplica uma combinação de quebras ao dataset original de forma controlada.

        Args:
            df (pd.DataFrame): O DataFrame original e limpo.
            missing_fraction (float): Fração geral de valores a serem convertidos para NaN.
            col_missing_fraction (float): Fração de colunas a serem completamente preenchidas com NaN.
            row_missing_fraction (float): Fração de linhas a serem completamente preenchidas com NaN.
            noise_fraction (float): Fração de valores numéricos a receber ruído.
            noise_level (float): Nível de ruído gaussiano a ser adicionado.

        Returns:
            pd.DataFrame: DataFrame quebrado com valores faltantes e ruído introduzidos.
        """
        df_broken = self.introduce_missing_values(df, missing_fraction)
        df_broken = self.introduce_column_missing(df_broken, col_missing_fraction)
        df_broken = self.introduce_row_missing(df_broken, row_missing_fraction)
        df_broken = self.add_noise(df_broken, noise_fraction, noise_level)
        return df_broken

if __name__ == "__main__":
    # Diretório com os dataframes originais
    input_folder = "data"
    # Diretório para salvar os dataframes quebrados
    output_folder = "broken_data"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Criada a pasta de saída: {output_folder}")

    # Verifica se a pasta de entrada existe
    if not os.path.exists(input_folder):
        print(f"A pasta de entrada '{input_folder}' não existe. Verifique o caminho.")
        exit(1)

    # Lista os arquivos na pasta de entrada
    input_files = os.listdir(input_folder)
    print(f"Arquivos encontrados na pasta '{input_folder}': {input_files}")

    # Inicializa o disruptor
    disruptor = DatasetDisruptor()

    # Percorre todos os arquivos .csv na pasta de entrada
    for filename in input_files:
        if filename.endswith(".csv"):
            filepath = os.path.join(input_folder, filename)
            print(f"Processando arquivo: {filepath}")
            try:
                df = pd.read_csv(filepath)
            except Exception as e:
                print(f"Erro ao ler o arquivo '{filename}': {e}")
                continue

            # Aplica as quebras no dataframe
            df_broken = disruptor.break_dataset(df)

            # Salva o dataframe quebrado na pasta de saída
            output_filepath = os.path.join(output_folder, filename)
            df_broken.to_csv(output_filepath, index=False)
            print(f"Arquivo quebrado salvo em: {output_filepath}")
        else:
            print(f"Arquivo ignorado (não é .csv): {filename}")
