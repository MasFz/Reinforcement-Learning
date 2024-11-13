# data_cleaning_env.py
import os
import numpy as np
import pandas as pd
from llm_client import LLMClient

class DataCleaningEnv:
    def __init__(self, llm_client, data_folder='broken_data'):
        self.llm_client = llm_client
        self.data_folder = data_folder
        self.data_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.csv')]
        self.current_file_index = -1
        self.reset()

    def reset(self):
        # Carrega o próximo dataframe quebrado
        self.current_file_index = (self.current_file_index + 1) % len(self.data_files)
        data_file = self.data_files[self.current_file_index]
        self.df = pd.read_csv(data_file)
        self.original_df = self.df.copy()  # Para referência
        self.state = self._get_state()
        # Inicializa o atributo missing_percent_before_feedback
        self.missing_percent_before_feedback = self.df.isnull().mean().mean()
        return self.state

    def _get_state(self):
        # Retorna uma representação do estado atual (por exemplo, percentagem de valores faltantes)
        missing_percent = self.df.isnull().mean().mean()
        if np.isnan(missing_percent):
            missing_percent = 1.0  # Assume 100% de valores faltantes se o valor for NaN
        return np.array([missing_percent])

    def step_coder(self, code, action_index):
        fallback_actions = {
            0: (
                "if not df.dropna().empty:\n"
                "    df.dropna(inplace=True)\n"
                "else:\n"
                "    pass  # Evita esvaziar o DataFrame"
            ),
            1: (
                "numeric_cols = df.select_dtypes(include=[np.number]).columns\n"
                "df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())"
            ),
            2: "df.fillna(0, inplace=True)",
            3: (
                "threshold = int(0.8 * df.shape[0])\n"
                "df.dropna(axis=1, thresh=threshold, inplace=True)"
            ),
        }

        print("Código gerado:\n", code)
        try:
            # Verifica se o código gerado contém chamadas não permitidas
            if 'input(' in code or 'import os' in code:
                raise ValueError("O código gerado contém chamadas não permitidas.")

            # Compila e executa o código gerado
            compiled_code = compile(code, '<string>', 'exec')
            safe_globals = {'df': self.df.copy(), 'pd': pd, 'np': np, '__builtins__': {}}
            safe_locals = {}
            exec(compiled_code, safe_globals, safe_locals)

            # Atualiza o DataFrame se foi modificado
            if 'df' in safe_globals:
                self.df = safe_globals['df']
            elif 'df' in safe_locals:
                self.df = safe_locals['df']
            else:
                print("O código não modificou o DataFrame.")

            reward = self._calculate_reward_coder()
            done = self._check_done()
            return self._get_state(), reward, done, {}

        except Exception as e:
            print(f"Erro no código gerado: {e}. Executando fallback para ação {action_index}")
            fallback_code = fallback_actions.get(action_index, "")
            # Compila e executa o código de fallback
            exec(fallback_code, {'df': self.df, 'pd': pd, 'np': np})
            # Após o fallback, calcula a recompensa
            reward = self._calculate_reward_coder()
            done = self._check_done()
            return self._get_state(), reward, done, {}

    def step_reviewer(self, feedback):
        print("Feedback recebido:", feedback)

        # Medir a porcentagem de valores faltantes após a ação do codificador
        missing_percent_after = self.df.isnull().mean().mean()
        if np.isnan(missing_percent_after):
            missing_percent_after = 1.0  # Assume 100% de valores faltantes

        # Calcular a recompensa com base na melhoria
        if missing_percent_after < self.missing_percent_before_feedback:
            reward = (self.missing_percent_before_feedback - missing_percent_after) * 10  # Recompensa positiva pela melhoria
        elif self.df.empty or self.df.isnull().all().all():
            reward = -1.0  # Penalidade severa
        else:
            reward = -0.5  # Pequena penalidade se não houver melhoria

        # Atualiza `missing_percent_before_feedback` para a próxima rodada
        self.missing_percent_before_feedback = missing_percent_after

        done = self._check_done()
        return self._get_state(), reward, done, {}

    def _calculate_reward_coder(self):
        # Calcula a recompensa para o codificador
        if self.df.empty or self.df.isnull().all().all():
            reward = -1.0  # Penalidade severa se o DataFrame estiver vazio ou com todos valores faltantes
        else:
            missing_percent = self.df.isnull().mean().mean()
            reward = -missing_percent  # Recompensa negativa proporcional aos valores faltantes
        return reward

    def _check_done(self):
        # Verifica se o episódio terminou
        if self.df.empty or self.df.isnull().all().all():
            return True
        else:
            return False