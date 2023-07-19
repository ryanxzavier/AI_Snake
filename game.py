import pygame
import random
from enum import Enum
from collections import namedtuple, deque
import numpy as np

pygame.init()
font = pygame.font.Font('OpenSans-Bold.ttf', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# Colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
YELLOW = (234, 221, 202)
GREEN = (0, 255, 0)  # New color for obstacles

BLOCK_SIZE = 20
SPEED = 400000000000000000000000000

class GameNode:
    def __init__(self, position, goal, snake, catcher, obstacles):  # Added obstacles parameter
        self.position = position
        self.goal = goal
        self.snake = snake
        self.catcher = catcher
        self.obstacles = obstacles  # New attribute for obstacles
        self.g = 0
        self.h = 0
        self.f = 0

    def is_goal(self):
        return self.position == self.goal

    def get_neighbors(self):
        neighbors = []
        directions = [Direction.RIGHT, Direction.LEFT, Direction.UP, Direction.DOWN]

        for direction in directions:
            new_position = self.calculate_new_position(direction)
            if self.is_valid_move(new_position):
                neighbor = GameNode(new_position, self.goal, self.snake, self.catcher, self.obstacles)  # Pass obstacles to the new node
                neighbors.append(neighbor)

        return neighbors

    def calculate_new_position(self, direction):
        x = self.position.x
        y = self.position.y

        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE

        return Point(x, y)

    def is_valid_move(self, position):
        # Check if position is within the game boundaries
        if position.x < 0 or position.x >= self.w or position.y < 0 or position.y >= self.h:
            return False

        # Check if position is not colliding with the snake's body
        if position in self.snake:
            return False

        # Check if position is not colliding with obstacles
        if position in self.obstacles:
            return False

        return True

    def reconstruct_path(self):
        path = []
        current_node = self
        while current_node is not None:
            path.append(current_node)
            current_node = current_node.parent
        return path

class SnakeAI:
    def __init__(self, w=640, h=480, catcher_length=3):
        self.n_games = 0  # Initialize the n_games attribute
        self.w = w
        self.h = h
        self.catcher_length = catcher_length
        self.head = Point(self.w / 2, self.h / 2)
        self.head2 = Point(self.w / 8, self.h / 8)  # Initial position of the second snake's head

        self.catcher = deque([
            Point(self.head.x - (i + 1) * BLOCK_SIZE, self.head.y) for i in range(catcher_length)
        ])
        self.obstacles = []  # List to store obstacles
        # Init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('SnakeAI')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # Init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w / 2, self.h / 2)
        self.head2 = Point(self.w / 8, self.h / 8)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)
        ]
        self.catcher = deque([
            Point(self.head.x - (i + 1) * BLOCK_SIZE, self.head.y) for i in range(self.catcher_length)
        ])
        self.score = 0
        self.food = None
        self.obstacles = []  # Reset obstacles
        self._place_food()
        self._place_obstacles()  # Place new obstacles
        self.frame_iteration = 0

    def _place_food(self):
        # Place food in the world
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake or self.food in self.obstacles:
            self._place_food()

    def _place_obstacles(self):
        # Place obstacles in the world
        num_obstacles = random.randint(5, 10)  # Number of obstacles to place
        self.obstacles = []
        for _ in range(num_obstacles):
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            obstacle = Point(x, y)
            if obstacle not in self.snake and obstacle != self.food:
                self.obstacles.append(obstacle)

    def play_step(self, action):
        self.frame_iteration += 1
        # 1. Collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. Move
        # Update the snake head
        self._move(action)
        self.snake.insert(0, self.head)

        # 3. Check if the game is over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 10000 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. Place new food or remove tail
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        # 5. Update UI and clock
        self._update_ui()
        self.clock.tick(SPEED)

        # 6. Return game over and score
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # Hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # Hits itself
        if pt in self.snake[1:]:
            return True
        # Hits obstacles
        if pt in self.obstacles:
            return True
        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        for obstacle in self.obstacles:
            pygame.draw.rect(self.display, GREEN, pygame.Rect(obstacle.x, obstacle.y, BLOCK_SIZE, BLOCK_SIZE))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        pygame.draw.rect(self.display, YELLOW, pygame.Rect(self.head2.x, self.head2.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, YELLOW, pygame.Rect(self.head2.x + 4, self.head2.y + 4, 12, 12))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):  # No change
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):  # Right turn r -> d -> l -> u
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:  # Left turn r -> u -> l -> d
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)

        # Update the position of the catcher
        self.catcher.pop()
        self.catcher.appendleft(Point(self.head.x - BLOCK_SIZE, self.head.y))
