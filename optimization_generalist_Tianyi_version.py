import sys
from evoman.environment import Environment
from demo_controller import player_controller

import numpy as np
import os
import argparse
import pickle
from itertools import combinations
from deap import base, creator, tools, algorithms

def simulation(env, x):
    f, p, e, t = env.play(pcont=x)  
    return f, p, e, t

def evaluate(env, x):
    return np.array([simulation(env, individual) for individual in x])

def multi_evaluate(multi_env, x):
    res = []
    for individual in x:
        res.append(np.array([simulation(env, individual) for env in multi_env]))
    return np.array(res)

def init(n_pop, n_vars):
    return np.random.uniform(-1, 1, (n_pop, n_vars))

def tournament_selection(pop, fitness, k=2):
    selected_indices = np.random.choice(pop.shape[0], k, replace=False)
    selected = selected_indices[np.argmax(fitness[selected_indices])]
    return selected

def genetic_algorithm(env, n_pop, n_vars, generations, mutation_rate, elitism_size=2):
    pop = init(n_pop, n_vars)
    fitness = evaluate(env, pop)[:, 0]
    
    best_fitness = []
    mean_fitness = []
    std_fitness = []
    
    for gen in range(generations):
        elite_indices = fitness.argsort()[-elitism_size:]
        elites = pop[elite_indices]
        
        new_pop = list(elites)  
        
        while len(new_pop) < n_pop:
            parent1_idx = tournament_selection(pop, fitness)
            parent2_idx = tournament_selection(pop, fitness)
            parent1 = pop[parent1_idx]
            parent2 = pop[parent2_idx]
            
            crossover_mask = np.random.rand(n_vars) < 0.5
            child = np.where(crossover_mask, parent1, parent2)
            
            mutation_mask = np.random.rand(n_vars) < mutation_rate
            mutation_strength = 0.3
            child[mutation_mask] += np.random.normal(0, mutation_strength, np.sum(mutation_mask))
            child = np.clip(child, -1, 1)
            
            new_pop.append(child)
        
        pop = np.array(new_pop)
        fitness = evaluate(env, pop)[:, 0]
        
        best_fitness.append(np.max(fitness))
        mean_fitness.append(np.mean(fitness))
        std_fitness.append(np.std(fitness))
        
        print(f'GA Generation {gen+1}: Best Fitness = {best_fitness[-1]:.4f}, '
              f'Mean Fitness = {mean_fitness[-1]:.4f}, Std Fitness = {std_fitness[-1]:.4f}')
    
    return pop, fitness, best_fitness, mean_fitness, std_fitness

def differential_evolution(env, n_pop, n_vars, generations, mutation_factor=0.8, crossover_rate=0.7):
    pop = init(n_pop, n_vars)
    fitness = evaluate(env, pop)[:, 0]
    
    best_fitness = []
    mean_fitness = []
    std_fitness = []
    
    for gen in range(generations):
        for i in range(n_pop):
            current_strategy = np.random.choice(['rand/1', 'best/1']) 
            
            if current_strategy == 'rand/1':
                indices = list(range(n_pop))
                indices.remove(i)
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                mutant = a + mutation_factor * (b - c)
            elif current_strategy == 'best/1':
                best_idx = np.argmax(fitness)
                a = pop[best_idx]
                indices = list(range(n_pop))
                indices.remove(best_idx)
                b, c = pop[np.random.choice(indices, 2, replace=False)]
                mutant = a + mutation_factor * (b - c)
            else:
                raise ValueError("Unsupported DE strategy.")
 
            mutant = np.clip(mutant, -1, 1)
            
            crossover_mask = np.random.rand(n_vars) < crossover_rate
            if not np.any(crossover_mask):
                crossover_mask[np.random.randint(0, n_vars)] = True
            trial = np.where(crossover_mask, mutant, pop[i])
            
            trial_fitness = evaluate(env, [trial])[0][0]
            if trial_fitness > fitness[i]:
                pop[i] = trial
                fitness[i] = trial_fitness
    
        best_fitness.append(np.max(fitness))
        mean_fitness.append(np.mean(fitness))
        std_fitness.append(np.std(fitness))
        
        print(f'DE Generation {gen+1}: Best Fitness = {best_fitness[-1]:.4f}, '
              f'Mean Fitness = {mean_fitness[-1]:.4f}, Std Fitness = {std_fitness[-1]:.4f}')
    
    return pop, fitness, best_fitness, mean_fitness, std_fitness

def DEMO(env, multi_test_env, n_pop, n_vars, generations, mutation_factor=0.8, crossover_rate=0.7):
    pop = init(n_pop, n_vars)
    fitness = evaluate(env, pop)[:, 0]
        
    best_fitness = []
    mean_fitness = []
    std_fitness = []
    
    for gen in range(generations):
        for i in range(n_pop):
            current_strategy = np.random.choice(['rand/1', 'best/1']) 
            
            if current_strategy == 'rand/1':
                indices = list(range(n_pop))
                indices.remove(i)
                index_a, index_b, index_c = np.random.choice(indices, 3, replace = False)
                a, b, c = pop[[index_a, index_b, index_c]]
                mutant = a + mutation_factor * (b - c)
            elif current_strategy == 'best/1':
                best_idx = np.argmax(fitness)
                a = pop[best_idx]
                index_a = best_idx
                indices = list(range(n_pop))
                indices.remove(best_idx)
                
                index_b, index_c = np.random.choice(indices, 2, replace = False)
                b, c = pop[[index_b, index_c]]
                mutant = a + mutation_factor * (b - c)
            else:
                raise ValueError("Unsupported DE strategy.")
             
            mutant = np.clip(mutant, -1, 1)

            crossover_mask = np.random.rand(n_vars) < crossover_rate
            if not np.any(crossover_mask):
                crossover_mask[np.random.randint(0, n_vars)] = True
            
            trial = np.where(crossover_mask, mutant, a)
            
            trial_fitness, a_fitness = multi_evaluate(multi_test_env, [trial, a])[:, :, 0]
            if np.all(trial_fitness >= a_fitness):
                pop[index_a] = trial
                fitness[index_a] = evaluate(env, [trial])[0][0]
            elif np.all(trial_fitness <= a_fitness):
                continue
            else:
                pop = np.append(pop, trial[np.newaxis, :], axis = 0)
                fitness = np.append(fitness, evaluate(env, [trial])[0][0])
            
            # truncate
            if len(pop) > n_pop:
                sorted_index = np.argsort(fitness)
                pop = pop[sorted_index][:n_pop]
                fitness = fitness[sorted_index][:n_pop]

        best_fitness.append(np.max(fitness))
        mean_fitness.append(np.mean(fitness))
        std_fitness.append(np.std(fitness))
        
        print(f'DEMO Generation {gen+1}: Best Fitness = {best_fitness[-1]:.4f}, '
              f'Mean Fitness = {mean_fitness[-1]:.4f}, Std Fitness = {std_fitness[-1]:.4f}')
    
    return pop, fitness, best_fitness, mean_fitness, std_fitness

def calculate_gain(env, best_individual):
    _, player_hp, enemy_hp, _ = env.play(pcont=best_individual)
    gain = player_hp - enemy_hp
    return gain

def evaluate_individual(env, individual):
    f, p, e, t = env.play(pcont=individual)
    return f, p, -e  # 返回适应度、玩家生命值和敌人生命值的负值

def nsga2(env, n_pop, n_vars, generations, mutation_rate, crossover_rate):
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, 1.0))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.uniform, -1, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n_vars)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate_individual, env)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=mutation_rate)
    toolbox.register("select", tools.selNSGA2)

    pop = toolbox.population(n=n_pop)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    # 评估初始种群
    fitnesses = toolbox.map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    for gen in range(generations):
        offspring = algorithms.varAnd(pop, toolbox, cxpb=crossover_rate, mutpb=mutation_rate)
        
        # 评估后代
        fitnesses = toolbox.map(toolbox.evaluate, offspring)
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit
        
        # 选择下一代
        pop = toolbox.select(pop + offspring, k=len(pop))
        
        # 记录统计信息
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(offspring), **record)
        print(f"Generation {gen}: {record}")

    return pop, logbook

def train(enemy_group, algorithm, run, n_pop, generations, mutation_rate, mutation_factor, crossover_rate):
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    
    enemy_group_str = '-'.join(map(str, enemy_group))
    experiment_name = f'solution_general/{enemy_group_str}/{algorithm}/run{run}'
    os.makedirs(experiment_name, exist_ok=True)
    
    n_hidden_neurons = 10
    
    env = Environment(
        experiment_name=experiment_name,
        enemies=enemy_group,
        multiplemode="yes",
        playermode="ai",
        player_controller=player_controller(n_hidden_neurons),
        enemymode="static",
        level=2,
        speed="fastest",
        visuals=False
    )

    os.makedirs('test/', exist_ok = True)
    multi_test_env = [Environment(
        enemies = [i],
        multiplemode="no",
        playermode="ai",
        player_controller=player_controller(n_hidden_neurons),
        enemymode="static",
        level=2,
        speed="fastest",
        visuals=False) for i in enemy_group
    ]

    num_sensors = env.get_num_sensors()
    n_vars = (num_sensors + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5
    print(f'Enemy Group {enemy_group_str}: num_sensors={num_sensors}, n_vars={n_vars}')
    print(f'Population shape: {init(n_pop, n_vars).shape}')

    if algorithm == 'NSGA2':
        pop, logbook = nsga2(
            env, n_pop, n_vars, generations, mutation_rate, crossover_rate
        )
        best_individual = tools.selBest(pop, k=1)[0]
        final_fitness = best_individual.fitness.values[0]  # 获取最后一代的最佳适应度
    elif algorithm == 'DE':
        pop, fitness, best_f, mean_f, std_f = differential_evolution(
            env, n_pop, n_vars, generations, mutation_factor, crossover_rate
        )
        best_individual = pop[np.argmax(fitness)]
        final_fitness = np.max(fitness)  # 获取最后一代的最佳适应度
    elif algorithm == 'DEMO':
        pop, fitness, best_f, mean_f, std_f = DEMO(
            env, multi_test_env, n_pop, n_vars, generations, mutation_factor, crossover_rate
        )
        best_individual = pop[np.argmax(fitness)]
        final_fitness = np.max(fitness)  # 获取最后一代的最佳适应度
    else:
        raise ValueError("Unsupported algorithm type.")
    
    np.save(os.path.join(experiment_name, 'best_individual.npy'), best_individual)
    
    final_gain = calculate_gain(env, best_individual)
    
    return final_fitness, final_gain

def test(enemy_group, algorithm, run):
    enemy_group_str = '-'.join(map(str, enemy_group))
    print(f"Testing with algorithm: {algorithm} (Run {run}) for Enemy Group: {enemy_group_str}")
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    
    experiment_name = f'solution_general/{enemy_group_str}/{algorithm}/run{run}'
    
    n_hidden_neurons = 10
    env = Environment(
        experiment_name=experiment_name,
        enemies=enemy_group,
        multiplemode="yes",
        playermode="ai",
        player_controller=player_controller(n_hidden_neurons),
        enemymode="static",
        level=2,
        speed="fastest",
        visuals=False
    )
    
    num_sensors = env.get_num_sensors()
    n_vars = (num_sensors + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5
    print(f'Enemy Group {enemy_group_str}: num_sensors={num_sensors}, n_vars={n_vars}')
   
    best_individual_path = os.path.join(experiment_name, 'best_individual.npy')
    if not os.path.exists(best_individual_path):
        print(f'Best individual not found for {algorithm} on Enemy Group {enemy_group_str}, Run {run}')
        return None
    
    best_individual = np.load(best_individual_path)
    print(f'Best individual shape: {best_individual.shape}')
    
    gain = calculate_gain(env, best_individual)
    
    print(f'Test Run {run} for {algorithm} on Enemy Group {enemy_group_str}: Gain = {gain}')
    
    return gain

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default='train', help='Mode: train or test')
    parser.add_argument('-a', '--algorithm', type=str, default='NSGA2', choices=['DE', 'NSGA2'], help='Algorithm: DE or NSGA2')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    args = parser.parse_args()
    np.random.seed(args.seed)
    
    enemy_groups = [(1,3,4), (1,5,6), (1,7,8), (2,3,4), (2,5,6), (2,7,8)]
    # algorithms = ["NSGA2", "DE"]

    algorithms = ["DEMO"]

    generations = 30
    n_pop = 100
    mutation_rate = 0.2
    mutation_factor = 0.5
    crossover_rate = 0.6

    if args.mode == 'train':
        print("Algorithm\tEnemy Group\tFinal Fitness\tFinal Gain")
        for algorithm in algorithms:
            for enemy_group in enemy_groups:
                final_fitness, final_gain = train(
                    enemy_group=enemy_group,
                    algorithm=algorithm,
                    run=1,
                    n_pop=n_pop,
                    generations=generations,
                    mutation_rate=mutation_rate,
                    mutation_factor=mutation_factor,
                    crossover_rate=crossover_rate
                )

                enemy_group_str = '-'.join(map(str, enemy_group))
                print(f"{algorithm}\t{enemy_group_str}\t{final_fitness:.4f}\t{final_gain:.4f}")

        print('\n=== Training Completed ===\n')

    elif args.mode == 'test':
        print("Algorithm\tEnemy Group\tTest Gain")
        for enemy_group in enemy_groups:
            for algorithm in algorithms:
                gain = test(
                    enemy_group=enemy_group,
                    algorithm=algorithm,
                    run=1
                )
                enemy_group_str = '-'.join(map(str, enemy_group))
                print(f"{algorithm}\t{enemy_group_str}\t{gain:.4f}")

        print('\n=== Testing Completed ===\n')

if __name__ == '__main__':
    main()