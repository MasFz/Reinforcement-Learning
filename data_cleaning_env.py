import gym
from gym import spaces
import numpy as np
import pandas as pd

class DataCleaningEnv(gym.Env):
    def __init__(self, llm_client):
        super(DataCleaningEnv, self).__init__()

        self.llm_client = llm_client

        # Estados to usando aqui números discretos para cada estado
        self.state_space = spaces.Discrete(1)
        self.state = 0

        # Ações do codificador
        self.action_space_coder = spaces.Discrete(4)
        # Ações do revisor
        self.action_space_reviewer = spaces.Discrete(5)

        # Dataset bruto (simulado)
        self.raw_data = self.generate_raw_data()
        self.df = self.raw_data.copy()

        self.done = False

    def generate_raw_data(self):
        # Gerar um DataFrame com valores faltantes
        data = {
            "A": [1, np.nan, 3, np.nan, 5],
            "B": [np.nan, 2, np.nan, 4, np.nan],
            "C": [1, 2, 3, 4, 5],
            "D": [np.nan, np.nan, np.nan, np.nan, np.nan]  # Coluna com 100% de valores faltantes
        }
        df = pd.DataFrame(data)
        return df

    def reset(self):
        self.state = 0
        self.df = self.raw_data.copy()
        self.done = False
        return self.state

    def step_coder(self, action):
        # Mapear a ação para uma operação de limpeza
        actions = {
            0: "Remover linhas com valores faltantes.",
            1: "Remover colunas com mais de 67% de valores faltantes.",
            2: "Interpolar valores faltantes.",
            3: "Preencher valores faltantes com a média."
        }
        prompt = actions[action]

        # Obter código de limpeza da LLM
        code = self.llm_client.generate_code(prompt)

        # Aplicar o código ao DataFrame
        try:
            exec(code, {"df": self.df, "np": np, "pd": pd})
            # Atualizar o DataFrame após a execução
            self.df = eval("df")
            reward = self.evaluate_cleaning()
            done = False
        except Exception as e:
            # Penalizar se o código gerar erro
            reward = -1
            done = True

        # Atualizar o estado (pode ser incrementado ou mantido simples)
        self.state = 0

        return self.state, reward, done, {}

    def step_reviewer(self, action):
        # Mapear a ação para um feedback
        actions = {
            0: "Aprovar limpeza.",
            1: "Sugerir remoção de colunas com muitos valores faltantes.",
            2: "Sugerir interpolação de valores faltantes.",
            3: "Sugerir preenchimento com média.",
            4: "Sugerir nenhuma ação."
        }
        prompt = actions[action]

        # Obter feedback da LLM
        feedback = self.llm_client.generate_feedback(prompt)

        # Avaliar o feedback (simples para este exemplo)
        reward = self.evaluate_feedback(feedback)
        done = False

        return self.state, reward, done, {}

    def evaluate_cleaning(self):
        # Avaliar o DataFrame limpo
        total_missing = self.df.isnull().sum().sum()
        if total_missing == 0:
            reward = 1  # Recompensa máxima se não houver valores faltantes
        else:
            reward = -total_missing / (self.df.size)  # Penaliza proporcionalmente aos valores faltantes
        return reward

    def evaluate_feedback(self, feedback):
        # Avaliar se o feedback é útil (simplificado)
        if "considere" in feedback.lower() or "verifique" in feedback.lower():
            reward = 1
        else:
            reward = 0
        return reward

    def render(self, mode="human"):
        print("Estado atual:", self.state)
        print("DataFrame atual:")
        print(self.df)
