import numpy as np

class QTable:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = np.zeros((state_size, action_size))

    def update(self, state, action, reward, next_state, done, learning_rate, gamma):
        state_index = np.argmax(state)  # Convert state to an index
        action_index = np.argmax(action)  # Convert action to an index
        current_q = self.q_table[state_index, action_index]
        max_future_q = np.max(self.q_table[next_state])
        new_q = (1 - learning_rate) * current_q + learning_rate * (reward + gamma * max_future_q)
        self.q_table[state_index, action_index] = new_q

class QTrainer:
    def __init__(self, q_table, learning_rate, gamma):
        self.q_table = q_table
        self.learning_rate = learning_rate
        self.gamma = gamma

    def train_step(self, states, actions, rewards, next_states, dones):
        for i in range(len(states)):
            state = states[i]
            action = actions[i]
            reward = rewards[i]
            next_state = next_states[i]
            done = dones[i]
            self.q_table.update(state, action, reward, next_state, done, self.learning_rate, self.gamma)

