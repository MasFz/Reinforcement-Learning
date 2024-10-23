from data_cleaning_env import DataCleaningEnv
from agents import CoderAgent, ReviewerAgent
from llm_client import LLMClient
import matplotlib.pyplot as plt
import numpy as np

# Inicializa o cliente LLM
llm_client = LLMClient()

# Inicializa o ambiente
env = DataCleaningEnv(llm_client)

# Definir número de estados e ações
num_states = 10  # Você precisa definir como quantizar os estados
num_coder_actions = len(CoderAgent(0,0).actions)
num_reviewer_actions = len(ReviewerAgent(0,0).actions)

# Inicializa os agentes
coder_agent = CoderAgent(num_states, num_coder_actions)
reviewer_agent = ReviewerAgent(num_states, num_reviewer_actions)

# Parâmetros de treinamento
num_episodes = 50
coder_rewards = []
reviewer_rewards = []

def discretize_state(state, num_states):
    # Implementar a discretização do estado contínuo em um índice inteiro
    # Por simplicidade, assumiremos que o estado é uma única variável entre 0 e 1
    state_value = state[0]
    if state_value >= 1.0:
        state_value = 0.999
    return int(state_value * num_states)

for episode in range(num_episodes):
    state = env.reset()
    state_index = discretize_state(state, num_states)
    done = False
    total_coder_reward = 0
    total_reviewer_reward = 0

    # Inicializa as ações para SARSA
    coder_action_index = coder_agent.select_action(state_index)
    reviewer_action_index = reviewer_agent.select_action(state_index)

    while not done:
        # Agente Codificador executa uma ação
        action_coder = coder_agent.actions[coder_action_index]

        # Gera o prompt para o LLM
        prompt = f"Escreva apenas o código Python para {action_coder}. Não use a função input()."

        # O LLM gera o código
        code = llm_client.generate_code(prompt)

        # O ambiente executa o código
        next_state, reward_coder, done_coder, _ = env.step_coder(code)
        total_coder_reward += reward_coder

        # Atualiza o agente Codificador
        next_state_index = discretize_state(next_state, num_states)
        next_coder_action_index = coder_agent.select_action(next_state_index)
        coder_agent.update(state_index, coder_action_index, reward_coder, next_state_index)
        coder_action_index = next_coder_action_index

        # Agente Revisor executa uma ação
        action_reviewer = reviewer_agent.actions[reviewer_action_index]

        # Gera o prompt para o LLM
        prompt_feedback = f"O código fornecido foi:\n{code}\n\nDê feedback: {action_reviewer}."

        # O LLM gera o feedback
        feedback = llm_client.generate_feedback(prompt_feedback)

        # O ambiente processa o feedback
        next_state_reviewer, reward_reviewer, done_reviewer, _ = env.step_reviewer(feedback)
        total_reviewer_reward += reward_reviewer

        # Atualiza o agente Revisor usando SARSA
        next_reviewer_state_index = discretize_state(next_state_reviewer, num_states)
        next_reviewer_action_index = reviewer_agent.select_action(next_reviewer_state_index)
        reviewer_agent.update(state_index, reviewer_action_index, reward_reviewer, next_reviewer_state_index, next_reviewer_action_index)
        reviewer_action_index = next_reviewer_action_index

        # Atualiza o estado
        state_index = next_state_index

        done = done_coder or done_reviewer

    # Decai o epsilon (exploração)
    coder_agent.decay_epsilon(decay_rate=0.99, min_epsilon=0.1)
    reviewer_agent.decay_epsilon(decay_rate=0.99, min_epsilon=0.1)

    coder_rewards.append(total_coder_reward)
    reviewer_rewards.append(total_reviewer_reward)

    print(f"Episódio {episode + 1}/{num_episodes} concluído.")
    print(f"Recompensa Codificador: {total_coder_reward}")
    print(f"Recompensa Revisor: {total_reviewer_reward}")
    print(f"Estado atual: {state}")
    print(f"DataFrame atual:\n{env.df.head()}")
    print(f"Valores faltantes totais: {env.df.isnull().sum().sum()}")
    print("--------------------------------------------------")

# Plotar as recompensas
plt.plot(np.arange(num_episodes), coder_rewards, label='Codificador')
plt.plot(np.arange(num_episodes), reviewer_rewards, label='Revisor')
plt.xlabel('Episódios')
plt.ylabel('Recompensa')
plt.legend()
plt.show()
