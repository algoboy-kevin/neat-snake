import multiprocessing
import pickle
import pygame
import os
import neat
import sys
import random
import time
from utils import does_checkpoint_exist, get_last_episode
from game_env import *
from render import spawn_window,render_basic, render

WINNER_FILENAME = "winner/latest-winner"
FPS = 5
FRAME_RATE = 1/FPS

def test_env():
    scores = []
    iterations = 10
    env = SnekEnv()
    pygame.init()
    spawn_window()

    # test genome fitness by 10 iterations
    for _ in range(iterations):
        obs = env.reset()
        done = False
        while not done:
            render_basic(env.snake, env.apple)
            action = random.randint(0,3)
            obs, done = env.step(action)
           
            # if snake does not eat after 100 step, round is reset
            if env.n_step_hunger > 100:
                done = True

            # append score if snake dead
            if done:
                scores.append(env.rewards)
            
            time.sleep(FRAME_RATE)

    # return the average scores
    mean = np.mean(scores)
    return mean

def simulate(net):
    scores = []
    iterations = 10
    env = SnekEnv()

    # test genome fitness by 10 iterations
    for _ in range(iterations):
        obs = env.reset()
        done = False
        while not done:
            activation = net.activate(obs)
            action = np.argmax(activation)
            obs, done = env.step(action)
            
            # if snake does not eat after 100 step, round is reset
            if env.n_step_hunger > 1000:
                done = True

            # append score if snake dead
            if done:
                scores.append(env.rewards)

    # return the average scores
    return np.mean(scores)

def replay_genome(genome, config, display=False):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    env = SnekEnv()
    obs = env.reset()
    done = False
    if display:
        # modify neural net to accomodate rendering
        env.render_init(net, genome, config)

        # pygame initialization
        spawn_window()
        pygame.init()

        while not done:
            # net activation must be triggered to see hidden nodes
            activation = env.net.activate(obs)
            render(
                env.snake, 
                env.apple, 
                env.net, 
                env.genome, 
                env.node_centers, 
                env.hidden_nodes,
            )
            action = np.argmax(activation)
            obs, done = env.step(action)
            time.sleep(FRAME_RATE)

        pygame.quit()
    else:
        while not done:
            activation = net.activate(obs)
            action = np.argmax(activation)
            obs, done = env.step(action)

    # env close
            
def eval_genomes(genomes, config):
    best_genome = None
    best_fit = -1
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = simulate(net)  

        if genome.fitness > best_fit:
            best_fit = genome.fitness
            best_genome = genome

    if best_fit >= 40:
        replay_genome(best_genome, config)

def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    fitness = simulate(net)  
    return fitness

def run_trained(config_file, genome):
    with open(genome, "rb") as f:
        winner = pickle.load(f, encoding="latin-1")

    config = neat.Config(
        neat.DefaultGenome, 
        neat.DefaultReproduction,                 
        neat.DefaultSpeciesSet, 
        neat.DefaultStagnation,
        config_file
    )
    
    replay_genome(winner, config, True)

def run(config_file, checkpoint_path, arg):
    # Load configuration.
    config = neat.Config(
        neat.DefaultGenome, 
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet, 
        neat.DefaultStagnation,
        config_file
    )

    # Create the population, which is the top-level object for a NEAT run.
    # Check if a winner file already exists
    if does_checkpoint_exist(checkpoint_path):
        print("Found checkpoint on ", checkpoint_path)
        last_episode = get_last_episode(checkpoint_path)
        p = neat.Checkpointer.restore_checkpoint(last_episode)
          
    else:
        # Start training from scratch
        p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(10, filename_prefix='checkpoints/neat-checkpoint-'))

    # Run for up to 500 generations.
    if arg == 'save':
        winner = p.run(eval_genomes, 2)
    elif arg == 'train':
        pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
        winner = p.run(pe.evaluate, n=500)

    # Save file once winner is found
    with open(WINNER_FILENAME, 'wb') as f:
        pickle.dump(winner, f)

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    checkpoint_path = os.path.join(local_dir, 'checkpoints')
    winner_path = os.path.join(local_dir, 'winner/latest-winner')
    default_winner_path = os.path.join(local_dir, 'winner/default-winner')
    
    if len(sys.argv) == 0:
        run(config_path)
    elif sys.argv[1] == 'test':
        test_env()
    elif sys.argv[1] == 'save':
        run(config_path, checkpoint_path, 'save')
    elif sys.argv[1] == 'train':
        run(config_path, checkpoint_path, 'train')
    elif sys.argv[1] == 'run':
        run_trained(config_path, winner_path)
    elif sys.argv[1] == 'run_master':
        run_trained(config_path, default_winner_path)

    