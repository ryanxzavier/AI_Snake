import random
import numpy as np
from collections import deque
from game import GameNode,SnakeAI, Direction, Point
from model import QTable, QTrainer
from helper import plot
import heapq
#from catcher_agent import CatchAgent

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class SnakeAgent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.q_table = QTable(11, 3)  # Assuming state size is 11 and action size is 3
        self.trainer = QTrainer(self.q_table, learning_rate=LR, gamma=self.gamma)
        self.experience_replay_buffer = ExperienceReplayBuffer(max_size=MAX_MEMORY)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r))
            or (dir_l and game.is_collision(point_l))
            or (dir_u and game.is_collision(point_u))
            or (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r))
            or (dir_d and game.is_collision(point_l))
            or (dir_l and game.is_collision(point_u))
            or (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r))
            or (dir_u and game.is_collision(point_l))
            or (dir_r and game.is_collision(point_u))
            or (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y,  # food down

            # Distance to food (optional)
            abs(game.food.x - game.head.x) / game.w,
            abs(game.food.y - game.head.y) / game.h
        ]

        return np.array(state, dtype=int)

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

class ExperienceReplayBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []

    def add_experience(self, experience):
        self.buffer.append(experience)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def sample_batch(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        return states, actions, rewards, next_states, dones

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    snake_agent = SnakeAgent()
    #catcher_agent = CatchAgent()

    game = SnakeAI()

    while True:
        state_old = snake_agent.get_state(game)

        final_move = snake_agent.get_action(state_old)

        reward, done, score = game.play_step(final_move)

        state_new = snake_agent.get_state(game)

        snake_agent.train_short_memory(state_old, final_move, reward, state_new, done)
        snake_agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            snake_agent.n_games += 1
            snake_agent.train_long_memory()

            if score > record:
                record = score

            print('Game', snake_agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / snake_agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()
