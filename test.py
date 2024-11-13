# test.py
from data_cleaning_env import DataCleaningEnv
from agents import CoderAgent, ReviewerAgent
from llm_client import LLMClient
import numpy as np
import os

# Initialize the LLM client
llm_client = LLMClient()

# Initialize the environment
env = DataCleaningEnv(llm_client)

# Define number of states and actions
num_states = 10  # Must match the value used during training
num_coder_actions = len(CoderAgent(0, 0).actions)
num_reviewer_actions = len(ReviewerAgent(0, 0).actions)

# Initialize the agents
coder_agent = CoderAgent(num_states, num_coder_actions, epsilon=0.0)  # Set epsilon to 0 for evaluation
reviewer_agent = ReviewerAgent(num_states, num_reviewer_actions, epsilon=0.0)  # Set epsilon to 0 for evaluation

# Load the trained Q-tables
models_folder = "models"
coder_agent.q_table = np.load(os.path.join(models_folder, 'best_coder_q_table.npy'))
reviewer_agent.q_table = np.load(os.path.join(models_folder, 'best_reviewer_q_table.npy'))

# Testing parameters
num_test_episodes = 5

def discretize_state(state, num_states):
    state_value = state[0]
    if np.isnan(state_value):
        state_value = 1.0  # Assume worst case if NaN
    state_value = min(max(state_value, 0), 0.999)
    return int(state_value * num_states)

for episode in range(num_test_episodes):
    state = env.reset()
    state_index = discretize_state(state, num_states)
    done = False
    total_coder_reward = 0
    total_reviewer_reward = 0

    step_count = 0

    max_steps_per_episode = 50

    while not done and step_count < max_steps_per_episode:
        step_count += 1

        # Coder Agent selects action (exploitation only)
        coder_action_index = np.argmax(coder_agent.q_table[state_index])
        action_coder = coder_agent.actions[coder_action_index]

        # Generate code from LLM
        prompt = f"""
        Você é um assistente que ajuda a limpar dataframes do pandas.
        O dataframe 'df' contém dados com valores faltantes e possivelmente ruídos.
        Sua tarefa é: {action_coder}.
        Forneça apenas o código Python necessário para realizar essa tarefa no dataframe 'df'.
        Não inclua explicações ou uso de 'input()' ou 'import os'.
        """

        code = llm_client.generate_code(prompt)

        # Environment executes the code
        next_state, reward_coder, done_coder, _ = env.step_coder(code, coder_action_index)
        total_coder_reward += reward_coder

        # Discretize next state
        next_state_index = discretize_state(next_state, num_states)

        # Reviewer Agent selects action (exploitation only)
        reviewer_action_index = np.argmax(reviewer_agent.q_table[state_index])
        action_reviewer = reviewer_agent.actions[reviewer_action_index]

        # Generate feedback from LLM
        prompt_feedback = f"""
        O código fornecido foi:
        {code}

        Dê feedback: {action_reviewer}.
        """

        feedback = llm_client.generate_feedback(prompt_feedback)

        # Environment processes the feedback
        next_state_reviewer, reward_reviewer, done_reviewer, _ = env.step_reviewer(feedback)
        total_reviewer_reward += reward_reviewer

        # Update state index
        state_index = next_state_index

        done = done_coder or done_reviewer

    print(f"Teste Episódio {episode + 1}/{num_test_episodes} concluído.")
    print(f"Recompensa Total do Codificador: {total_coder_reward}")
    print(f"Recompensa Total do Revisor: {total_reviewer_reward}")
    print(f"Estado Final: {next_state}")
    print(f"DataFrame Final:\n{env.df.head()}")
    print(f"Valores Faltantes Totais: {env.df.isnull().sum().sum()}")
    print("--------------------------------------------------")