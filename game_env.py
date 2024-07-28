from random import random
import numpy as np
from render_network import modify_eval_functions, get_node_centers

# ---------------- game settings --------------

N_ROWS = 10
N_COLS = 10

screen = None

# ---------------- game environment --------------
class SnekEnv():
    snake = [(3, 3)]
    apple = (5, 5)
    dead = False
    running = True
    v_x, v_y = 1, 0
    n_step = 0
    n_step_hunger = 0
    done = False
    display = False
    rewards = 0
    
    def __init__(self):
        pass

    def render_init(self, net, genome, config):
        # calling render init will enable display
        self.net = net
        self.genome = genome
        self.config = config

        # Modify nets to accomodate display
        modify_eval_functions(self.net, self.genome, self.config)
        has_eval = set(eval[0] for eval in self.net.node_evals)
        has_input = set(con[1] for con in self.genome.connections)
        self.hidden_nodes = [node for node in self.genome.nodes if not 0 <= node <= 3 and node in has_input and node in has_eval]
        self.node_centers = get_node_centers(self.net, self.genome, self.hidden_nodes)
        
    def reset(self):
        self.snake = [((int) (random() * N_COLS), (int) (random() * N_ROWS))]
        self.apple = (int) (random() * N_COLS), (int) (random() * N_ROWS)
        self.v_x, self.v_y = 1, 0 
        self.running = True
        self.score = 0
        self.n_step = 0
        self.last_ate_apple = 0
        self.done = False
        self.rewards = 0

        return self.get_observation()

    def step(self, action):
        if not self.done: 
            self.change_direction(action)
            self.move_snake()
        
        return self.get_observation(), self.done

    def move_snake(self):
        ate_apple = False
        x, y = self.snake[-1]

        # append snake length
        self.snake.append((x + self.v_x, y + self.v_y))
        x, y = self.snake[-1]

        # hit wall
        if x < 0 or x >= N_COLS or y < 0 or y >= N_ROWS:
            self.done = True

        # hit body
        for s in self.snake[:-1]:
            if s == self.snake[-1]:
                self.done = True
                break

        # moving, delete 1 box
        if not self.snake[-1] == self.apple:
            self.snake.pop(0)
            self.n_step_hunger += 1
        
        # ate apple
        else:
            self.apple = (int) (random() * N_COLS), (int) (random() * N_ROWS)
            ate_apple = True
            self.rewards += 1
            self.n_step_hunger = 0

        if ate_apple:
            self.last_ate_apple = self.n_step

    def change_direction(self, action):
        assert(0 <= action <= 3)

        # wasd
        if action == 0:
            self.v_x = 0
            self.v_y = -1
        elif action == 1:
            self.v_x = -1
            self.v_y = 0
        elif action == 2:
            self.v_x = 0
            self.v_y = 1
        else:
            self.v_x = 1
            self.v_y = 0 

    def get_observation(self):
        x, y = self.snake[-1]

        # inverted distance to wall
        # d_N, d_S, d_E, d_W
        dist_to_wall = [
            (1 / (y + 1)) if y + 1 != 0 else 10, 
            (1 / (N_ROWS - y)) if N_ROWS - y != 0 else 10, 
            (1 / (N_COLS - x)) if N_COLS - x != 0 else 10, 
            (1 / (x + 1)) if x + 1 != 0 else 10
        ]

        # flag for if will hit tail in this cardinal direction
        will_hit_tail = [0, 0, 0, 0]

        for (body_x, body_y) in self.snake[:-1]:
            if body_x == x:
                if body_y > y:
                    will_hit_tail[1] = 1
                else:
                    will_hit_tail[0] = 1
            elif body_y == y:
                if body_x > x:
                    will_hit_tail[2] = 1
                else:
                    will_hit_tail[3] = 1

        # apple
        a_x, a_y = self.apple

        apple_info = [
            x == a_x and a_y < y,
            x == a_x and a_y > y,
            y == a_y and a_x > x,
            y == a_y and a_x < x,
        ]

        # return 1.0 * np.array(dist_to_wall + will_hit_tail)
        return 1.0 * np.array(dist_to_wall + will_hit_tail + apple_info)


        