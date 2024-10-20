from data_cleaning_env import DataCleaningEnv
from agents import CoderAgent, ReviewerAgent
from llm_client import LLMClient
import matplotlib.pyplot as plt

api_key = "sua_chave_de_api_aqui"
llm_client = LLMClient(api_key)

env = DataCleaningEnv(llm_client)

num_states = env.state_space.n
num_actions_coder = env.action_space_coder.n
num_actions_reviewer = env.action_space_reviewer.n

coder_agent = CoderAgent(num_states, num_actions_coder)
reviewer_agent = ReviewerAgent(num_states, num_actions_reviewer)

num_episodes = 100
rewards_coder = []
rewards_reviewer = []

missing_values = [] # Lista para armazenar a quantidade de valores faltantes em cada episódio

for episode in range(num_episodes):
    state = env.reset()
    total_reward_coder = 0
    total_reward_reviewer = 0
    done = False

    # Agente Codificador age
    action_coder = coder_agent.select_action(state)
    next_state, reward_coder, done_coder, _ = env.step_coder(action_coder)
    total_reward_coder += reward_coder
    coder_agent.update(state, action_coder, reward_coder, next_state)
    coder_agent.decay_epsilon(0.995, 0.01)
    state = next_state

    # Agente Revisor age
    action_reviewer = reviewer_agent.select_action(state)
    next_state, reward_reviewer, done_reviewer, _ = env.step_reviewer(action_reviewer)
    total_reward_reviewer += reward_reviewer
    next_action_reviewer = reviewer_agent.select_action(next_state)
    reviewer_agent.update(state, action_reviewer, reward_reviewer, next_state, next_action_reviewer)
    reviewer_agent.decay_epsilon(0.995, 0.01)
    state = next_state

    rewards_coder.append(total_reward_coder)
    rewards_reviewer.append(total_reward_reviewer)

    # Armazenar a quantidade de valores faltantes restantes após o episódio
    missing_values.append(env.df.isnull().sum().sum())

    if (episode + 1) % 10 == 0:
        print(f"Episódio {episode+1}/{num_episodes} concluído.")
        print(f"Recompensa Codificador: {total_reward_coder}, Recompensa Revisor: {total_reward_reviewer}")
        env.render()
        print("-" * 50)

# Plotar as recompensas
plt.plot(rewards_coder, label="Agente Codificador")
plt.plot(rewards_reviewer, label="Agente Revisor")
plt.xlabel("Episódio")
plt.ylabel("Recompensa")
plt.title("Recompensas por Episódio")
plt.legend()
plt.show()

# Plotar os valores faltantes
plt.plot(missing_values, label="Valores faltantes")
plt.xlabel("Episódio")
plt.ylabel("Valores Faltantes Restantes")
plt.title("Evolução dos Valores Faltantes por Episódio")
plt.legend()
plt.show()
