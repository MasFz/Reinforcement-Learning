import numpy as np

class CoderAgent:
    def __init__(self, num_states, num_actions, epsilon=1.1, alpha=0.1, gamma=0.95):
        self.q_table = np.zeros((num_states, num_actions))
        self.epsilon = epsilon  # Taxa de exploração
        self.alpha = alpha      # Taxa de aprendizado
        self.gamma = gamma      # Fator de desconto
        # Defina as ações possíveis (por exemplo, diferentes tipos de prompts)
        self.actions = [
            "Remover linhas com valores nulos em 'df'",
            "Preencher valores nulos em 'df' com a média",
            "Preencher valores nulos em 'df' com zero",
            "Remover colunas com muitos valores nulos em 'df'",
            # Adicione mais ações conforme necessário
        ]

    def select_action(self, state_index):
        if np.random.rand() < self.epsilon:
            action_index = np.random.randint(len(self.actions))  # Exploração
        else:
            action_index = np.argmax(self.q_table[state_index])   # Exploração
        return action_index

    def update(self, state_index, action_index, reward, next_state_index):
        best_next_action = np.argmax(self.q_table[next_state_index])
        td_target = reward + self.gamma * self.q_table[next_state_index, best_next_action]
        td_error = td_target - self.q_table[state_index, action_index]
        self.q_table[state_index, action_index] += self.alpha * td_error

    def decay_epsilon(self, decay_rate, min_epsilon):
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)

class ReviewerAgent:
    def __init__(self, num_states, num_actions, epsilon=1.1, alpha=0.1, gamma=0.95):
        self.q_table = np.zeros((num_states, num_actions))
        self.epsilon = epsilon  # Taxa de exploração
        self.alpha = alpha      # Taxa de aprendizado
        self.gamma = gamma      # Fator de desconto
        # Defina as ações possíveis (por exemplo, diferentes tipos de feedback)
        self.actions = [
            "O código está correto",
            "O código tem erros de sintaxe",
            "O código não resolve o problema",
            "O código é ineficiente",
            # Adicione mais ações conforme necessário
        ]

    def select_action(self, state_index):
        if np.random.rand() < self.epsilon:
            action_index = np.random.randint(len(self.actions))  # Exploração
        else:
            action_index = np.argmax(self.q_table[state_index])   # Exploração
        return action_index

    def update(self, state_index, action_index, reward, next_state_index, next_action_index):
        td_target = reward + self.gamma * self.q_table[next_state_index, next_action_index]
        td_error = td_target - self.q_table[state_index, action_index]
        self.q_table[state_index, action_index] += self.alpha * td_error

    def decay_epsilon(self, decay_rate, min_epsilon):
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)
