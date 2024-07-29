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

WINNER_FILENAME = "winner/last-winner"

def test():
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
            
            time.sleep(0.5)

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
            if env.n_step_hunger > 100:
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
        env.render_init(net, genome, config)
        spawn_window()
        pygame.init()
        while not done:
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
            time.sleep(0.03)

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

    if best_fit >= 20:
        replay_genome(best_genome, config)

def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    fitness = simulate(net, genome, config)  
    return fitness

def test_winner(config_file, genome):
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
    p.add_reporter(neat.Checkpointer(5, filename_prefix='checkpoints/neat-checkpoint-'))

    # Run for up to 500 generations.
    if arg == 'serial':
        winner = p.run(eval_genomes, 3)
    elif arg == 'parallel':
        pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
        winner = p.run(pe.evaluate, n=500)

    # Save file once winner is found
    with open(WINNER_FILENAME, 'wb') as f:
        pickle.dump(winner, f)

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    checkpoint_path = os.path.join(local_dir, 'checkpoints')
    winner_path = os.path.join(local_dir, 'winner/last-winner')
    default_winner_path = os.path.join(local_dir, 'winner/default-winner')
    
    if len(sys.argv) == 0:
        run(config_path)
    elif sys.argv[1] == 'test':
        test()
    elif sys.argv[1] == 'train':
        run(config_path, checkpoint_path, 'serial')
    elif sys.argv[1] == 'train_fast':
        run(config_path, checkpoint_path, 'parallel')
    elif sys.argv[1] == 'test_winner':
        test_winner(config_path, winner_path)
    elif sys.argv[1] == 'default_winner':
        test_winner(config_path, default_winner_path)

    