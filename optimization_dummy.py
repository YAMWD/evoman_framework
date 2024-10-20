###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# DEMO : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Karine Miras        			                                      #
# karine.smiras@gmail.com     				                                  #
###############################################################################

# imports framework
import sys

from evoman.environment import Environment
from demo_controller import player_controller

# imports other libs
import numpy as np
import os
import argparse
import pickle
import pandas as pd
from time import sleep

# runs simulation
def simulation(env,x):
    f, p, e, t = env.play(pcont = x)
    return f, p, e, t

# evaluation
def evaluate(env, x):
    return np.array(list(map(lambda y: simulation(env,y), x)))

def int2list(enemy_number):
    s = str(enemy_number)
    tmp = []
    for item in s:
        tmp.append(int(item))
    return tmp

def init(n_pop, n_vars):
    return np.random.normal(0, 1, (n_pop, n_vars))

def tournament(pop, fitness):

    p1, p2 = np.random.randint(0, pop.shape[0], size = 2)

    return p1 if fitness[p1] > fitness[p2] else p2

def crossover(env, pop, fitness, p_mutation, selection):

    n_pop = pop.shape[0]

    pop_new = pop

    for i in range(n_pop):
        if(selection == 'random'):
            p1, p2 = np.random.randint(0, pop.shape[0], size = 2)
        elif(selection == 'tournament'):
            p1, p2 = tournament(pop, fitness), tournament(pop, fitness)
        elif(selection == 'DE'):
            while(True):
                p1, p2 = np.random.randint(0, pop.shape[0], size = 2)
                if(p1 != i and p2 != i):
                    break

        alpha = np.random.rand()

        if(selection == 'DE'):
            x = pop[i]
            v = pop[p1] - pop[p2]
            u = x + alpha * v
            l = [x, u, v]
            f = evaluate(env, l)[:, 0]
            offspring = l[np.argmax(f)]
        else: 
            offspring = alpha * pop[p1] + (1 - alpha) * pop[p2] + (np.random.rand(pop[p1].shape[0]) if np.random.rand() < p_mutation else 0)

        pop_new = np.vstack((pop_new, offspring))
    
    return pop_new

def select(n_pop, pop, fitness):

    index = np.argpartition(fitness, n_pop)[-n_pop:]
    
    return pop[index], fitness[index]
    

def train(enemy_number, Continue, selection, index = 0):
    # choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        
    experiment_name = 'solution/' + str(enemy_number)
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    n_hidden_neurons = 10

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name = experiment_name,
                    enemies = int2list(enemy_number),
                    playermode = "ai",
                    player_controller = player_controller(n_hidden_neurons), # you can insert your own controller here
                    enemymode = "static",
                    level = 2,
                    speed = "fastest",
                    visuals = False,
                    multiplemode = 'yes')

    # number of weights for multilayer with 10 hidden neurons
    n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5
    n_pop = 100
    p_mutation = 0.2
    epoch = 100
    best_f = -1

    if(os.path.exists(experiment_name + '/pop_{}.bin'.format(index)) and Continue):
        pop = np.fromfile(experiment_name + '/pop_{}.bin'.format(index), dtype = np.float64).reshape(n_pop, n_vars)
        results = evaluate(env, pop)
        fitness, player_hp, enemy_hp, time = results[:, 0], results[:, 1], results[:, 2], results[:, 3]        
        best_f = np.max(fitness)
        print('best_f {}'.format(best_f))
        sleep(1)
    else:
        pop = init(n_pop, n_vars)
        results = evaluate(env, pop)
        fitness, player_hp, enemy_hp, time = results[:, 0], results[:, 1], results[:, 2], results[:, 3]        
        print("training from scratch")
        sleep(1)

    data_fitness = {'mean': np.array([]), 'std': np.array([]), 'max': np.array([])}
    data_player_hp = {'mean': np.array([]), 'std': np.array([]), 'max': np.array([])}
    data_enemy_hp = {'mean': np.array([]), 'std': np.array([]), 'max': np.array([])}
    data_time = {'mean': np.array([]), 'std': np.array([]), 'max': np.array([])}
    for i in range(epoch):
        # data collection
        # mean
        results = evaluate(env, pop)
        fitness, player_hp, enemy_hp, time = results[:, 0], results[:, 1], results[:, 2], results[:, 3]  
        data_fitness['mean'] = np.append(data_fitness['mean'], np.mean(fitness))
        data_player_hp['mean'] = np.append(data_player_hp['mean'], np.mean(player_hp))
        data_enemy_hp['mean'] = np.append(data_enemy_hp['mean'], np.mean(enemy_hp))
        data_time['mean'] = np.append(data_time['mean'], np.mean(time))
        
        # std
        data_fitness['std'] = np.append(data_fitness['std'], np.std(fitness))
        data_player_hp['std'] = np.append(data_player_hp['std'], np.std(player_hp))
        data_enemy_hp['std'] = np.append(data_enemy_hp['std'], np.std(enemy_hp))
        data_time['std'] = np.append(data_time['std'], np.std(time))

        # max
        data_fitness['max'] = np.append(data_fitness['max'], np.max(fitness))
        data_player_hp['max'] = np.append(data_player_hp['max'], np.max(player_hp))
        data_enemy_hp['max'] = np.append(data_enemy_hp['max'], np.max(enemy_hp))
        data_time['max'] = np.append(data_time['max'], np.max(time))

        pop = crossover(env, pop, fitness, p_mutation, selection)
        results = evaluate(env, pop)
        fitness, player_hp, enemy_hp, time = results[:, 0], results[:, 1], results[:, 2], results[:, 3]        
        pop, fitness = select(n_pop, pop, fitness)

        index_best = np.argmax(fitness)
        pop_best = pop[index_best]
        fitness_best = fitness[index_best]

        # import pdb; pdb.set_trace()

        print('epoch {} best fitness {}'.format(i, fitness_best))

        if(fitness_best > best_f):
            print('best solution saved to {}/best_{}.bin and {}/pop_{}.bin'.format(experiment_name, index, experiment_name, index))
            pop_best.tofile(experiment_name + '/best_{}.bin'.format(index))
            pop.tofile(experiment_name + '/pop_{}.bin'.format(index))
            best_f = fitness_best
    
    return data_fitness, data_player_hp, data_enemy_hp, data_time


def test(enemy_number, index = 0):
    # choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    n_hidden_neurons = 10

    experiment_name = 'solution/' + str(enemy_number)
    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name = experiment_name,
                    enemies = int2list(enemy_number),
                    playermode = "ai",
                    player_controller = player_controller(n_hidden_neurons), # you can insert your own controller here
                    enemymode = "static",
                    level = 2,
                    speed = "normal",
                    visuals = False,
                    multiplemode = 'yes')

    # number of weights for multilayer with 10 hidden neurons
    n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

    pop = np.fromfile(experiment_name + '/best_{}.bin'.format(index), dtype = np.float64).reshape(1, n_vars)
    
    results = evaluate(env, pop)[0]
    fitness, player_hp, enemy_hp, time = results[0], results[1], results[2], results[3]

    print('fitness {}, player_hp {}, enemy_hp {}, time {}'.format(fitness, player_hp, enemy_hp, time))

    return player_hp - enemy_hp

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type = str, default = 'train')
    parser.add_argument('-n', '--enemy_number', type = int, default = 1)
    parser.add_argument('-c', '--Continue', action = 'store_true')
    parser.add_argument('-s', '--seed', type = int, default = 0)
    parser.add_argument('--selection', type = str, default = 'random')
    
    args = parser.parse_args()

    if(args.mode == 'train'):
        train(args.enemy_number, args.Continue, args.selection)
    elif(args.mode == 'test'):
        test(args.enemy_number)
    elif(args.mode == 'data'):
        experiment_name = 'solution/' + str(args.enemy_number)

        f = np.array([])
        p = np.array([])
        e = np.array([])
        t = np.array([])
        for i in range(10):
            data_f, data_p, data_e, data_t = train(args.enemy_number, False, args.selection, i)
            f = np.append(f, data_f)
            p = np.append(p, data_p)
            e = np.append(e, data_e)
            t = np.append(t, data_t)
        
        with open(experiment_name + '/data_f.pkl', 'wb') as file:
            pickle.dump(f, file)

    elif(args.mode == 'data_test'):
        data = {'score': []}
        for i in range(10):
            score = test(args.enemy_number, i)
            data['score'].append(score)
        
        experiment_name = 'solution/' + str(args.enemy_number)
        with open(experiment_name + '/data_score.pkl', 'wb') as file:
            pickle.dump(data, file)


