import pygame

# ---------------- display settings --------------

N_ROWS = 10
N_COLS = 10
NET_W, NET_H = 450, 900
GAME_W, GAME_H = 450, 450
INTERVAL = 1000
WINDOW_BUFFER = 25
SCREEN_WIDTH = 500 
SCREEN_HEIGHT = 900
BLOCK_W, BLOCK_H = GAME_W / N_COLS, GAME_H / N_ROWS
GAME_TOP_LEFT = (WINDOW_BUFFER + 0, 400 + WINDOW_BUFFER)
NODE_SIZE = 10
BUFFER = 8
screen = None

# ---------------- color settings --------------

RED = (255, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
BLACK = (0, 0, 0)
BLUE = (70, 130, 180)
ORANGE = (255, 165, 13)

def spawn_window():
   # Initialize screen
    global screen
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    STEP = pygame.USEREVENT + 1
    pygame.time.set_timer(STEP, INTERVAL)

def render_basic(snake, apple):
    global screen

    for event in pygame.event.get():
        if (event.type == pygame.QUIT):
            pygame.quit()

    screen.fill(BLACK)
    draw_square() 
    draw_snake(snake) 
    draw_apple(apple) 
    pygame.display.flip()

def render(snake, apple, net, genome, node_centers, hidden_nodes):
    global screen

    for event in pygame.event.get():
        if (event.type == pygame.QUIT):
            pygame.quit()

    screen.fill(BLACK)
    draw_square(screen) 
    draw_snake(snake) 
    draw_apple(apple) 
    draw_network( 
        net, 
        genome, 
        node_centers, 
        hidden_nodes
    )
    pygame.display.flip()

def draw_square():
    global screen
    draw = GAME_TOP_LEFT[0] - BUFFER, GAME_TOP_LEFT[1] - BUFFER
    rect = pygame.Rect(draw, (GAME_W + 2 * BUFFER, GAME_H + 2 * BUFFER))
    pygame.draw.rect(screen, WHITE, rect, width=BUFFER // 2)

def draw_apple(apple_coordinate):
    global screen
    x, y = apple_coordinate
    rect = pygame.Rect(getLeftTop(x, y), (BLOCK_W - BUFFER * 2, BLOCK_H - BUFFER * 2))
    pygame.draw.rect(screen, RED, rect)

def getLeftTop(x, y):
    return (x / N_ROWS) * GAME_W + BUFFER + GAME_TOP_LEFT[0], (y / N_ROWS) * GAME_H + BUFFER + GAME_TOP_LEFT[1]

def draw_snake(snake):
    global screen
    print(snake)
      
    for i, (x, y) in enumerate(snake):
        rect = pygame.Rect(getLeftTop(x, y), (BLOCK_W - BUFFER * 2, BLOCK_H - BUFFER * 2))
        pygame.draw.rect(screen, YELLOW if i == len(snake) - 1 else WHITE, rect)

def draw_connections(first_set, second_set, net, genome, node_centers):
  global screen
  for first in first_set:
    for second in second_set:
      if (first, second) in genome.connections:
        start = node_centers[first]
        stop = node_centers[second]
        weight = genome.connections[(first, second)].weight
        color = BLUE if weight >= 0 else ORANGE

        surf = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        alpha = 255 * (0.3 + net.values[first] * 0.7)
        pygame.draw.line(surf, color + (alpha,), start, stop, width=5)
        screen.blit(surf, (0, 0))

def draw_network(net, genome, node_centers, hidden_nodes):
    global screen
    font = pygame.font.Font("cmunbtl.otf", 24)

    node_names = {
        -1 : "d_N_wall",
        -2 : "d_S_wall",
        -3 : "d_E_wall",
        -4 : "d_W_wall",
        -5 : "tail_N",
        -6 : "tail_S",
        -7 : "tail_E",
        -8 : "tail_W",
        -9 : "apple_N",
        -10 : "apple_S",
        -11 : "apple_E",
        -12 : "apple_W",
        0: 'up', 1 : "left", 2 : "down", 3 : "right"
    }

    # draw connections between input and output nodes
    draw_connections(net.input_nodes, net.output_nodes, net, genome, node_centers)
    draw_connections(net.input_nodes, hidden_nodes, net, genome, node_centers)
    draw_connections(hidden_nodes, hidden_nodes, net, genome, node_centers)
    draw_connections(hidden_nodes, net.output_nodes, net, genome, node_centers)

    # draw input nodes
    for i, input_node in enumerate(net.input_nodes):
        center = node_centers[input_node]

        center2 = center[0] - 5.5 * NODE_SIZE - 75, center[1] - 13
        img = font.render(node_names[input_node], True, WHITE)
        screen.blit(img, center2)

        color = (net.values[input_node] * 255, 0, 0)

        pygame.draw.circle(screen, color, center, NODE_SIZE)
        pygame.draw.circle(screen, WHITE, center, NODE_SIZE, width=5)

    # draw output nodes
    for i, output_node in enumerate(net.output_nodes):
        center = node_centers[output_node]
        color = (net.values[output_node] * 255, 0, 0)
        pygame.draw.circle(screen, color, center, NODE_SIZE)
        pygame.draw.circle(screen, WHITE, center, NODE_SIZE, width=5)

        center2 = center[0] + 1.5 * NODE_SIZE, center[1] - 13
        img = font.render(node_names[output_node], True, WHITE)
        screen.blit(img, center2)

    # draw hidden nodes
    for hidden in hidden_nodes:
        center = node_centers[hidden]
        color = (net.values[hidden] * 255, 0, 0)

        # center2 = center[0] - 5.5 * NODE_SIZE, center[1] - 10
        # img = font.render(str(hidden), True, WHITE)
        # screen.blit(img, center2)

        pygame.draw.circle(color, center, NODE_SIZE)
        pygame.draw.circle(WHITE, center, NODE_SIZE, width=5)





