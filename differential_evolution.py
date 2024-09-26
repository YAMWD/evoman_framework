import sys
import os
import argparse
import pickle
import numpy as np
from evoman.environment import Environment
from demo_controller import player_controller

# 设置环境变量以禁用pygame的音频,这可以提高性能
os.environ['SDL_AUDIODRIVER'] = 'dummy'

def simulation(env, x):
    return env.play(pcont=x)

def evaluate(env, x):
    return np.array(list(map(lambda y: simulation(env, y), x)))

def int2list(enemy_number):
    return [int(i) for i in str(enemy_number)]

# GA specific functions
def init_population_ga(n_pop, n_vars):
    return np.random.normal(0, 1, (n_pop, n_vars))

def tournament(pop, fitness):
    p1, p2 = np.random.randint(0, pop.shape[0], size=2)
    return p1 if fitness[p1] > fitness[p2] else p2

def crossover_ga(env, pop, fitness, p_mutation, selection):
    n_pop = pop.shape[0]
    pop_new = pop.copy()

    for i in range(n_pop):
        if selection == 'random':
            p1, p2 = np.random.randint(0, pop.shape[0], size=2)
        elif selection == 'tournament':
            p1, p2 = tournament(pop, fitness), tournament(pop, fitness)

        alpha = np.random.rand()
        offspring = alpha * pop[p1] + (1 - alpha) * pop[p2]
        
        if np.random.rand() < p_mutation:
            offspring += np.random.normal(0, 0.1, size=offspring.shape)

        pop_new = np.vstack((pop_new, offspring))
    
    return pop_new

def select_ga(n_pop, pop, fitness):
    index = np.argpartition(fitness, -n_pop)[-n_pop:]
    return pop[index], fitness[index]

# DE specific functions
def init_population_de(n_pop, n_vars):
    return np.random.uniform(-1, 1, (n_pop, n_vars))

def differential_evolution(pop, F=0.8, CR=0.7):
    pop_size, n_vars = pop.shape
    for i in range(pop_size):
        a, b, c = np.random.choice(pop_size, 3, replace=False)
        mutant = pop[a] + F * (pop[b] - pop[c])
        cross_points = np.random.rand(n_vars) < CR
        trial = np.where(cross_points, mutant, pop[i])
        if not np.any(cross_points):
            idx = np.random.randint(n_vars)
            trial[idx] = mutant[idx]
        yield trial

def train_ga(env, n_hidden_neurons, n_pop, p_mutation, epoch, selection):
    n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5
    pop = init_population_ga(n_pop, n_vars)
    
    data = {'mean': [], 'std': [], 'max': []}
    best_f = -float('inf')

    for i in range(epoch):
        results = evaluate(env, pop)
        fitness = results[:, 0]
        
        pop = crossover_ga(env, pop, fitness, p_mutation, selection)
        results = evaluate(env, pop)
        fitness = results[:, 0]
        pop, fitness = select_ga(n_pop, pop, fitness)

        data['mean'].append(np.mean(fitness))
        data['std'].append(np.std(fitness))
        data['max'].append(np.max(fitness))

        index_best = np.argmax(fitness)
        if fitness[index_best] > best_f:
            best_f = fitness[index_best]
            best_solution = pop[index_best]

        print(f'Generation {i+1}: Best Fitness = {best_f}')

    return data, best_solution

def train_de(env, n_hidden_neurons, n_pop, epoch, F=0.8, CR=0.7):
    n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5
    pop = init_population_de(n_pop, n_vars)
    fitness = evaluate(env, pop)[:, 0]
    
    data = {'mean': [], 'std': [], 'max': []}
    best_f = -float('inf')

    for i in range(epoch):
        for trial in differential_evolution(pop, F, CR):
            trial_fitness = env.play(pcont=trial)[0]
            idx = np.random.randint(n_pop)
            if trial_fitness > fitness[idx]:
                pop[idx] = trial
                fitness[idx] = trial_fitness

        data['mean'].append(np.mean(fitness))
        data['std'].append(np.std(fitness))
        data['max'].append(np.max(fitness))

        index_best = np.argmax(fitness)
        if fitness[index_best] > best_f:
            best_f = fitness[index_best]
            best_solution = pop[index_best]

        print(f'Generation {i+1}: Best Fitness = {best_f}')

    return data, best_solution

def train(enemy_number, algorithm='ga', n_hidden_neurons=10, n_pop=100, p_mutation=0.2, 
          epoch=100, selection='tournament', F=0.8, CR=0.7):
    experiment_name = f'solution_{algorithm}/{enemy_number}'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    env = Environment(experiment_name=experiment_name,
                      enemies=int2list(enemy_number),
                      playermode="ai",
                      player_controller=player_controller(n_hidden_neurons),
                      enemymode="static",
                      level=2,
                      speed="fastest",
                      visuals=False)

    if algorithm == 'ga':
        data, best_solution = train_ga(env, n_hidden_neurons, n_pop, p_mutation, epoch, selection)
    else:  # DE
        data, best_solution = train_de(env, n_hidden_neurons, n_pop, epoch, F, CR)

    best_solution.tofile(f'{experiment_name}/best.bin')
    return data, best_solution

def test(enemy_number, algorithm='ga', n_hidden_neurons=10):
    experiment_name = f'solution_{algorithm}/{enemy_number}'
    env = Environment(experiment_name=experiment_name,
                      enemies=int2list(enemy_number),
                      playermode="ai",
                      player_controller=player_controller(n_hidden_neurons),
                      enemymode="static",
                      level=2,
                      speed="fastest",
                      visuals=False)

    n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5
    best_solution = np.fromfile(f'{experiment_name}/best.bin', dtype=np.float64)

    f, p, e, t = env.play(pcont=best_solution)
    return p - e  # individual gain

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default='train', choices=['train', 'test', 'data'])
    parser.add_argument('-n', '--enemy_number', type=int, default=1)
    parser.add_argument('-a', '--algorithm', type=str, default='de', choices=['ga', 'de'])
    args = parser.parse_args()

    if args.mode == 'train':
        data, _ = train(args.enemy_number, args.algorithm)
        with open(f'solution_{args.algorithm}/{args.enemy_number}/data.pkl', 'wb') as f:
            pickle.dump(data, f)
    elif args.mode == 'test':
        score = test(args.enemy_number, args.algorithm)
        print(f'Individual gain: {score}')
    elif args.mode == 'data':
        scores = []
        for _ in range(10):
            data, _ = train(args.enemy_number, args.algorithm)
            with open(f'solution_{args.algorithm}/{args.enemy_number}/data_{_}.pkl', 'wb') as f:
                pickle.dump(data, f)
            score = test(args.enemy_number, args.algorithm)
            scores.append(score)
        with open(f'solution_{args.algorithm}/{args.enemy_number}/scores.pkl', 'wb') as f:
            pickle.dump(scores, f)