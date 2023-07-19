import heapq
import random
import numpy as np
from game import Direction, GameNode
from snake_agent import BATCH_SIZE

class CatchAgent:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.target_x = w // 2

    def get_action(self, game):
        head_x = self.head.x

        if head_x < self.target_x:
            return [1, 0, 0]  # Move right
        elif head_x > self.target_x:
            return [0, 1, 0]  # Move left
        else:
            return [0, 0, 1]  # Move down

    def update_target(self):
        self.target_x = np.random.randint(0, self.w)

    def remember(self, state, action, reward, next_state, done):
        self.experience_replay_buffer.add_experience((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.experience_replay_buffer.buffer) >= BATCH_SIZE:
            states, actions, rewards, next_states, dones = self.experience_replay_buffer.sample_batch(BATCH_SIZE)
            self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.q_table.update(state, action, reward, next_state, done, self.trainer.learning_rate, self.trainer.gamma)
        learning_rate = self.trainer.learning_rate
        gamma = self.trainer.gamma
        state_index = np.argmax(state)  # Convert state to an index
        action_index = np.argmax(action)  # Convert action to an index
        self.q_table.update(state_index, action_index, reward, next_state, done, learning_rate, gamma)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
        else:
            move = np.argmax(self.q_table.q_table[state])

        if move >= 0 and move < len(final_move):
            final_move[move] = 1

        return final_move

    def search_food(self, game):
        start_node = GameNode(game.head, game.food, game.snake)
        path = self.astar_search(start_node)
        if path:
            return self.convert_path_to_action(path, game.direction)
        return None

    def astar_search(self, start_node):
        open_set = []
        closed_set = set()
        heapq.heappush(open_set, (0, start_node))

        while open_set:
            current_node = heapq.heappop(open_set)[1]
            if current_node.is_goal():
                return current_node.reconstruct_path()

            closed_set.add(current_node)
            neighbors = current_node.get_neighbors()

            for neighbor in neighbors:
                if neighbor in closed_set:
                    continue
                neighbor.g = current_node.g + 1
                neighbor.h = self.heuristic(neighbor)
                neighbor.f = neighbor.g + neighbor.h

                if neighbor not in open_set:
                    heapq.heappush(open_set, (neighbor.f, neighbor))

        return None

    def heuristic(self, node):
        return abs(node.position.x - node.goal.x) + abs(node.position.y - node.goal.y)

    def convert_path_to_action(self, path, current_direction):
        head = path[0].position
        next_point = path[1].position
        if next_point.x > head.x:
            if current_direction != Direction.LEFT:
                return [0, 1, 0]  # Right turn
        elif next_point.x < head.x:
            if current_direction != Direction.RIGHT:
                return [1, 0, 0]  # Left turn
        elif next_point.y > head.y:
            if current_direction != Direction.UP:
                return [0, 0, 1]  # Down turn
        elif next_point.y < head.y:
            if current_direction != Direction.DOWN:
                return [0, 1, 0]  # Up turn
        return None


