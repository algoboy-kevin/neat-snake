from __future__ import print_function
import multiprocessing
import pickle
from game_env import *
from render import spawn_window,render_basic
import pygame
import os
import neat
import sys
import random
import time

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
            
            time.sleep(0.1)

    # return the average scores
    print(scores)
    mean = np.mean(scores)
    print(mean)
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

def replay_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    env = SnekEnv(net, genome, config)
    obs = env.reset()
    done = False
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

def run(config_file, arg):
    # Load configuration.
    config = neat.Config(
        neat.DefaultGenome, 
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet, 
        neat.DefaultStagnation,
        config_file
    )

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5, filename_prefix='checkpoints/neat-checkpoint-'))

    # Run for up to 500 generations.
    if arg == 'serial':
        winner = p.run(eval_genomes, 500)
    elif arg == 'parallel':
        pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
        winner = p.run(pe.evaluate, n=500)

    with open('winner/neat-winner', 'wb') as f:
        pickle.dump(winner, f)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    
    if len(sys.argv) == 0:
        run(config_path)
    elif sys.argv[1] == 'test':
        test()
    elif sys.argv[1] == 'train':
        run(config_path, 'serial')
    elif sys.argv[1] == 'train_fast':
        run(config_path, 'parallel')

    