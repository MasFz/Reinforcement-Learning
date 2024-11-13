# train.py
from data_cleaning_env import DataCleaningEnv
from agents import CoderAgent, ReviewerAgent
from llm_client import LLMClient
import matplotlib.pyplot as plt
import numpy as np
import os

# Create folders for logs and models if they don't exist
logs_folder = "logs"
models_folder = "models"

if not os.path.exists(logs_folder):
    os.makedirs(logs_folder)

if not os.path.exists(models_folder):
    os.makedirs(models_folder)

# Initialize the LLM client
llm_client = LLMClient()

# Initialize the environment
env = DataCleaningEnv(llm_client)

# Define number of states and actions
num_states = 10  # You can adjust this value
num_coder_actions = len(CoderAgent(0, 0).actions)
num_reviewer_actions = len(ReviewerAgent(0, 0).actions)

# Initialize the agents
coder_agent = CoderAgent(num_states, num_coder_actions)
reviewer_agent = ReviewerAgent(num_states, num_reviewer_actions)

# Variables to track the best performance
best_coder_reward = float('-inf')
best_reviewer_reward = float('-inf')

# Training parameters
num_episodes = 30
coder_rewards = []
reviewer_rewards = []

def discretize_state(state, num_states):
    # Discretize the continuous state into an integer index
    state_value = state[0]
    if np.isnan(state_value):
        state_value = 1.0  # Assume the worst case if the value is NaN
    state_value = min(max(state_value, 0), 0.999)
    return int(state_value * num_states)

for episode in range(num_episodes):
    state = env.reset()
    state_index = discretize_state(state, num_states)
    done = False
    total_coder_reward = 0
    total_reviewer_reward = 0

    # Initialize actions for SARSA
    coder_action_index = coder_agent.select_action(state_index)
    reviewer_action_index = reviewer_agent.select_action(state_index)

    step_count = 0  # Step counter within the episode

    max_steps_per_episode = 50

    while not done and step_count < max_steps_per_episode:
        step_count += 1

        # Coder Agent executes an action
        action_coder = coder_agent.actions[coder_action_index]

        # Generate the prompt for the LLM
        prompt = f"""
        Você é um assistente que ajuda a limpar dataframes do pandas.
        O dataframe 'df' contém dados com valores faltantes e possivelmente ruídos.
        Sua tarefa é: {action_coder}.
        Forneça apenas o código Python necessário para realizar essa tarefa no dataframe 'df'.
        Não inclua explicações ou uso de 'input()' ou 'import os'.
        """

        # The LLM generates the code
        code = llm_client.generate_code(prompt)

        # Save the generated code to a file (optional)
        code_filename = f"episode_{episode + 1}_step_{step_count}_code.py"
        code_filepath = os.path.join(logs_folder, code_filename)
        with open(code_filepath, 'w', encoding='utf-8') as code_file:
            code_file.write(code)

        # The environment executes the code
        next_state, reward_coder, done_coder, _ = env.step_coder(code, coder_action_index)
        total_coder_reward += reward_coder

        # Update the Coder Agent
        next_state_index = discretize_state(next_state, num_states)
        next_coder_action_index = coder_agent.select_action(next_state_index)
        coder_agent.update(state_index, coder_action_index, reward_coder, next_state_index)
        coder_action_index = next_coder_action_index

        # Reviewer Agent executes an action
        action_reviewer = reviewer_agent.actions[reviewer_action_index]

        # Generate the prompt for the LLM
        prompt_feedback = f"""
        O código fornecido foi:
        {code}

        Dê feedback: {action_reviewer}.
        """

        # The LLM generates the feedback
        feedback = llm_client.generate_feedback(prompt_feedback)

        # Save the feedback to a file (optional)
        feedback_filename = f"episode_{episode + 1}_step_{step_count}_feedback.txt"
        feedback_filepath = os.path.join(logs_folder, feedback_filename)
        with open(feedback_filepath, 'w', encoding='utf-8') as feedback_file:
            feedback_file.write(feedback)

        # The environment processes the feedback
        next_state_reviewer, reward_reviewer, done_reviewer, _ = env.step_reviewer(feedback)
        total_reviewer_reward += reward_reviewer

        # Update the Reviewer Agent using SARSA
        next_reviewer_state_index = discretize_state(next_state_reviewer, num_states)
        next_reviewer_action_index = reviewer_agent.select_action(next_reviewer_state_index)
        reviewer_agent.update(state_index, reviewer_action_index, reward_reviewer, next_reviewer_state_index, next_reviewer_action_index)
        reviewer_action_index = next_reviewer_action_index

        # Update the state
        state_index = next_state_index

        done = done_coder or done_reviewer

    # Decay epsilon (exploration)
    coder_agent.decay_epsilon(decay_rate=0.995, min_epsilon=0.1)
    reviewer_agent.decay_epsilon(decay_rate=0.995, min_epsilon=0.1)

    coder_rewards.append(total_coder_reward)
    reviewer_rewards.append(total_reviewer_reward)

    print(f"Episódio {episode + 1}/{num_episodes} concluído.")
    print(f"Recompensa Codificador: {total_coder_reward}")
    print(f"Recompensa Revisor: {total_reviewer_reward}")
    print(f"Estado atual: {state}")
    print(f"DataFrame atual:\n{env.df.head()}")
    print(f"Valores faltantes totais: {env.df.isnull().sum().sum()}")
    print("--------------------------------------------------")

    # Check if performance improved and save the model
    if total_coder_reward > best_coder_reward:
        best_coder_reward = total_coder_reward
        np.save(os.path.join(models_folder, 'best_coder_q_table.npy'), coder_agent.q_table)
        print(f"Novo melhor modelo do Codificador salvo no episódio {episode + 1} com recompensa {best_coder_reward}")

    if total_reviewer_reward > best_reviewer_reward:
        best_reviewer_reward = total_reviewer_reward
        np.save(os.path.join(models_folder, 'best_reviewer_q_table.npy'), reviewer_agent.q_table)
        print(f"Novo melhor modelo do Revisor salvo no episódio {episode + 1} com recompensa {best_reviewer_reward}")

# After training, you can also save the final Q-tables
np.save(os.path.join(models_folder, 'final_coder_q_table.npy'), coder_agent.q_table)
np.save(os.path.join(models_folder, 'final_reviewer_q_table.npy'), reviewer_agent.q_table)

# Plot the rewards
plt.plot(np.arange(num_episodes), coder_rewards, label='Codificador')
plt.plot(np.arange(num_episodes), reviewer_rewards, label='Revisor')
plt.xlabel('Episódios')
plt.ylabel('Recompensa')
plt.legend()
plt.title('Recompensas ao longo dos Episódios')
plt.show()