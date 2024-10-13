import sys
from evoman.environment import Environment
from demo_controller import player_controller

import numpy as np
import os
import argparse
import pickle

def simulation(env, x):
    f, p, e, t = env.play(pcont=x)  
    return f, p, e, t

def evaluate(env, x):
    return np.array([simulation(env, individual) for individual in x])

def int2list(enemy_number):
    return [int(digit) for digit in str(enemy_number)]

# 随机初始化种群
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
        elite_indices = fitness.argsort()[-elitism_size:]   # ascending order
        elites = pop[elite_indices]
        elite_fitness = fitness[elite_indices]
        
        new_pop = list(elites)  
        
        while len(new_pop) < n_pop:
            parent1_idx = tournament_selection(pop, fitness)
            parent2_idx = tournament_selection(pop, fitness)
            parent1 = pop[parent1_idx]
            parent2 = pop[parent2_idx]
            
            # Uniform Crossover
            crossover_mask = np.random.rand(n_vars) < 0.5
            child = np.where(crossover_mask, parent1, parent2)
            
            # Gaussian Mutation
            mutation_mask = np.random.rand(n_vars) < mutation_rate
            mutation_strength = 0.3  # TODO: hyperparameter
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
            # 时而随机选择基向量，时而选择最佳个体
            current_strategy = np.random.choice(['rand/1', 'best/1']) 
            
            if current_strategy == 'rand/1':
                # DE/rand/1
                indices = list(range(n_pop))
                indices.remove(i)
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                mutant = a + mutation_factor * (b - c)
            elif current_strategy == 'best/1':
                # DE/best/1
                best_idx = np.argmax(fitness)
                a = pop[best_idx]
                indices = list(range(n_pop))
                indices.remove(best_idx)
                b, c = pop[np.random.choice(indices, 2, replace=False)]
                mutant = a + mutation_factor * (b - c)
            else:
                raise ValueError("Unsupported DE strategy.")
 
            mutant = np.clip(mutant, -1, 1)
            
            # Binomial Crossover
            crossover_mask = np.random.rand(n_vars) < crossover_rate
            if not np.any(crossover_mask):
                crossover_mask[np.random.randint(0, n_vars)] = True  # 确保至少一个基因交叉
            trial = np.where(crossover_mask, mutant, pop[i])
            
            # 选择
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

def calculate_gain(env, best_individual):
    _, player_hp, enemy_hp, _ = env.play(pcont=best_individual)
    gain = player_hp - enemy_hp
    return gain

def train(enemy_number, algorithm, run, n_pop, generations, mutation_rate, mutation_factor, crossover_rate):
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    
    experiment_name = f'solution_general/{enemy_number}/{algorithm}/run{run}'
    os.makedirs(experiment_name, exist_ok=True)
    
    n_hidden_neurons = 10
    
    env = Environment(
        experiment_name=experiment_name,
        enemies=enemy_number,
        multiplemode="yes",  # generalist
        playermode="ai",
        player_controller=player_controller(n_hidden_neurons),
        enemymode="static",
        level=2,
        speed="fastest",
        visuals=False
    )
    
    num_sensors = env.get_num_sensors()
    n_vars = (num_sensors + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5
    print(f'Enemy {enemy_number}: num_sensors={num_sensors}, n_vars={n_vars}')
    print(f'Population shape: {init(n_pop, n_vars).shape}')

    if algorithm == 'GA':
        pop, fitness, best_f, mean_f, std_f = genetic_algorithm(
            env, n_pop, n_vars, generations, mutation_rate
        )
    elif algorithm == 'DE':
        pop, fitness, best_f, mean_f, std_f = differential_evolution(
            env, n_pop, n_vars, generations, mutation_factor, crossover_rate
        )
    else:
        raise ValueError("Unsupported algorithm type.")
    
    best_idx = np.argmax(fitness)
    best_individual = pop[best_idx]
    np.save(os.path.join(experiment_name, 'best_individual.npy'), best_individual)
    
    # 计算并打印最终的gain
    final_gain = calculate_gain(env, best_individual)
    print(f'{algorithm} on Enemy {enemy_number}, Run {run}')
    print(f'{algorithm} Generation {generations}: Best Fitness = {best_f[-1]:.4f}, '
          f'Mean Fitness = {mean_f[-1]:.4f}, Std Fitness = {std_f[-1]:.4f}, '
          f'Gain = {final_gain}')
    
    return best_f, mean_f, std_f, final_gain

def test(enemy_number, algorithm, run):
    print(f"Testing with algorithm: {algorithm} (Run {run})")
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    
    experiment_name = f'solution_general/{enemy_number}/{algorithm}/run{run}'
    
    n_hidden_neurons = 10
    env = Environment(
        experiment_name=experiment_name,
        enemies=[1,2,3,4,5,6,7,8],
        multiplemode="yes",  # generalist
        playermode="ai",
        player_controller=player_controller(n_hidden_neurons),
        enemymode="static",
        level=2,
        speed="fastest",
        visuals=False
    )
    
    num_sensors = env.get_num_sensors()
    n_hidden_neurons = 10
    n_vars = (num_sensors + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5
    print(f'Enemy {enemy_number}: num_sensors={num_sensors}, n_vars={n_vars}')
   
    best_individual_path = os.path.join(experiment_name, 'best_individual.npy')
    if not os.path.exists(best_individual_path):
        print(f'Best individual not found for {algorithm} on Enemy {enemy_number}, Run {run}')
        return None
    
    best_individual = np.load(best_individual_path)
    print(f'Best individual shape: {best_individual.shape}')
    
    gain = calculate_gain(env, best_individual)
    
    print(f'Test Run {run} for {algorithm} on All Enemy: Gain = {gain}')
    
    return gain

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default='train', help='Mode: train, test, or full')
    parser.add_argument('-n', '--enemy_number', type=int, default=1, help='Enemy number (e.g., 1, 2, 3)')
    parser.add_argument('-a', '--algorithm', type=str, default='GA', choices=['GA', 'DE'], help='Algorithm: GA or DE')
    parser.add_argument('-r', '--runs', type=int, default=1, help='Number of runs per algorithm per enemy')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    args = parser.parse_args()
    np.random.seed(args.seed)
    
    #enemies = args.enemy_number
    enemies = [3,4,6]  # todo
    algorithms = ["GA"]

    n_runs = args.runs
    generations = 30
    n_pop = 200
    mutation_rate = 0.2  # BEST FOR GA
    mutation_factor = 0.5  # BEST FOR DE
    crossover_rate = 0.6  # BEST FOR DE

    results = {
#        'GA': {enemy: {'best_fitness': {}, 'mean_fitness': {}, 'std_fitness': {}, 'gain': {}} for enemy in enemies},
#        'DE': {enemy: {'best_fitness': {}, 'mean_fitness': {}, 'std_fitness': {}, 'gain': {}} for enemy in enemies}
#    }
    'GA':  {'best_fitness': {}, 'mean_fitness': {}, 'std_fitness': {}, 'gain': {}},
    'DE':  {'best_fitness': {}, 'mean_fitness': {}, 'std_fitness': {}, 'gain': {}}
    }

    if args.mode in ['train', 'full']:
        for algorithm in algorithms:
            for run in range(1, n_runs + 1):
                print(f'\n=== Training {algorithm} on Enemy {enemies}, Run {run} ===\n')
                best_f, mean_f, std_f, gain = train(
                    enemy_number=enemies,
                    algorithm=algorithm,
                    run=run,
                    n_pop=n_pop,
                    generations=generations,
                    mutation_rate=mutation_rate,
                    mutation_factor=mutation_factor,
                    crossover_rate=crossover_rate
                )

                if run not in results[algorithm]['best_fitness'].keys():
                    results[algorithm]['best_fitness'][run] = []
                if run not in results[algorithm]['mean_fitness'].keys():
                    results[algorithm]['mean_fitness'][run] = []
                if run not in results[algorithm]['std_fitness'].keys():
                    results[algorithm]['std_fitness'][run] = []

                results[algorithm]['best_fitness'][run].extend(best_f)
                results[algorithm]['mean_fitness'][run].extend(mean_f)
                results[algorithm]['std_fitness'][run].extend(std_f)

        save_name = f'solution_general/avg/{algorithm}'
        os.makedirs(save_name, exist_ok=True)        
        with open(f'{save_name}/data_f.pkl', 'wb') as file:
            pickle.dump(results, file)

        if args.mode == 'train':
            print('\n=== Training Completed ===\n')

    if args.mode in ['test', 'full']:
        with open('Task2_score.txt', 'a') as file:
            for run in range(10):
                gain_single = []
                for algorithm in ['DE','GA']:
                     for i in range(5):  # 5 repeats for each independent run
                         print(f'\n=== Testing {algorithm} on All Enemy, Run {run+1} ===\n')
                         gain = test(
                             enemy_number=enemies,
                             algorithm=algorithm,
                              run=run+1
                            )
                         gain_single.append(gain)
                     file.write(f"{round(sum(gain_single)/5,3)}\t{enemies}\t{algorithm}\n")
#        if args.mode == 'test':
#            print('\n=== Testing Completed ===\n')
    
    if args.mode == 'full':
        print('\n=== Full Process Completed ===\n')
    
    '''
    if args.mode in ['train', 'full']:
        print("\n=== Summary of Results ===\n")
        summary = {
            'GA': {'best_fitness': [], 'mean_fitness': [], 'gain': []},
            'DE': {'best_fitness': [], 'mean_fitness': [], 'gain': []}
        }
        
        for algorithm in algorithms:
            print(f"--- Algorithm: {algorithm} ---")
            for enemy in enemies:
                best_avg = np.mean(results[algorithm][enemy]['best_fitness'])
                mean_avg = np.mean(results[algorithm][enemy]['mean_fitness'])
                std_avg = np.mean(results[algorithm][enemy]['std_fitness'])
                gain_avg = np.mean(results[algorithm][enemy]['gain'])
                summary[algorithm]['best_fitness'].append(best_avg)
                summary[algorithm]['mean_fitness'].append(mean_avg)
                summary[algorithm]['gain'].append(gain_avg)
                print(f"Enemy {enemy}: Best Fitness Avg = {best_avg:.4f}, Mean Fitness Avg = {mean_avg:.4f}, "
                      f"Std Fitness Avg = {std_avg:.4f}, Gain Avg = {gain_avg:.4f}")
            print("\n")
    '''
    
if __name__ == '__main__':
    main()
