import numpy as np

class CoderAgent:
    def __init__(self, num_states, num_actions, epsilon=1.0, alpha=0.1, gamma=0.95):
        self.q_table = np.zeros((num_states, num_actions))
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.q_table.shape[1])
        else:
            action = np.argmax(self.q_table[state])
        return action

    def update(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state, best_next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

    def decay_epsilon(self, decay_rate, min_epsilon):
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)

class ReviewerAgent:
    def __init__(self, num_states, num_actions, epsilon=1.0, alpha=0.1, gamma=0.95):
        self.q_table = np.zeros((num_states, num_actions))
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.q_table.shape[1])
        else:
            action = np.argmax(self.q_table[state])
        return action

    def update(self, state, action, reward, next_state, next_action):
        td_target = reward + self.gamma * self.q_table[next_state, next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

    def decay_epsilon(self, decay_rate, min_epsilon):
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)
