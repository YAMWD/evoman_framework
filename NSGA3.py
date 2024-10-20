import argparse
import pickle
import random
import os
import numpy as np
from deap import base, creator, tools, algorithms
from evoman.environment import Environment
from demo_controller import player_controller

def simulation_deap(env, x):
    x = np.array(x)
    f, p, e, t = env.play(pcont=x)
    return p-e

def multi_enemy(envs,individual):
    single_gain = []
    for i in range(len(envs)):
        single_gain.append(simulation_deap(envs[i], individual))
    return single_gain

def init(n_pop, n_vars):
    return np.random.uniform(-1, 1, (n_pop, n_vars))

def nsga3(env,n_pop, n_vars,parent_ratio, crossover_rate, muation_rate, lamda_miu_ratio, generations):

    creator.create("FitnessMax", base.Fitness, weights=([1.0] * len(env)))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, -1.0, 1.0)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n_vars)
    toolbox.register("init_sigma", random.uniform, 0.01, 0.1)

    def init_individual():
        ind = toolbox.individual()
        setattr(ind, 'sigma', toolbox.init_sigma())
        return ind

    toolbox.register("individual_with_sigma", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual_with_sigma)

    def self_adaptive_mutation(individual, low, up):
        sigma = getattr(individual, 'sigma')  # 获取个体的变异率
        sigma *= np.exp(np.random.normal(0, 1))  # 自适应调整变异率

        for i in range(len(individual)):  # 只对权重进行变异
            if random.random() < sigma:
                individual[i] += random.uniform(low, up)
                individual[i] = min(max(individual[i], low), up)

        setattr(individual, 'sigma', min(max(sigma, 0.001), 0.5))  # 保证变异率在合理范围内
        return individual,

    toolbox.register("evaluate", multi_enemy,env)
    toolbox.register("mate", tools.cxBlend, alpha=parent_ratio)
    toolbox.register("mutate", self_adaptive_mutation, low=-1.0, up=1.0)  # self-adaptive mutation
    toolbox.register("select", tools.selNSGA3, ref_points=tools.uniform_reference_points(nobj=len(env), p=4))

    population = toolbox.population(n=n_pop)
    hof = tools.HallOfFame(1, similar=np.array_equal)

    stats = tools.Statistics(lambda ind: sum(ind.fitness.values))
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("max", np.max)
    pop, logs = algorithms.eaMuCommaLambda(population, toolbox,
                                          mu= n_pop,
                                          lambda_= n_pop * lamda_miu_ratio,
                                          cxpb=crossover_rate,
                                          mutpb=muation_rate,
                                          ngen=generations,stats=stats, halloffame=hof, verbose=True)
    best_solutions = hof[0]

    mean_fitness = [_['avg'] for _ in logs]
    std_fitness = [_['std'] for _ in logs]
    best_fitness = [_['max'] for _ in logs]

    return best_solutions, best_fitness, mean_fitness, std_fitness

def calculate_gain(env, best_individual):
    _, player_hp, enemy_hp, _ = env.play(pcont=best_individual)
    return player_hp,enemy_hp

def train(enemy_number, algorithm, run, n_pop, generations, mutation_rate, parent_ratio, lamda_miu_ratio, crossover_rate, outfix):
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    
    experiment_name = f'solution_figure/{enemy_number}/{algorithm}/{outfix}/run{run}'
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

    if algorithm == 'NSGA':
        envs = []
        for i in range(len(enemy_number)):
            env = Environment(
                experiment_name=experiment_name,
                enemies=[enemy_number[i]],
                playermode="ai",
                player_controller=player_controller(n_hidden_neurons),
                enemymode="static",
                level=2,
                speed="fastest",
                visuals=False)
            envs.append(env)
        best_individual, best_f, mean_f, std_f = nsga3(envs, n_pop, n_vars, parent_ratio, crossover_rate, mutation_rate, lamda_miu_ratio, generations)
    else:
        raise ValueError("Unsupported algorithm type.")

    best_individual = np.array(best_individual)

    np.save(os.path.join(experiment_name, 'best_individual.npy'), best_individual)
    
    print(f'{algorithm} Generation {generations}: Best Fitness = {best_f[-1]:.4f}, '
          f'Mean Fitness = {mean_f[-1]:.4f}, Std Fitness = {std_f[-1]:.4f},')
    
    return best_f, mean_f, std_f

def test(enemy_number, dir_name, algorithm, run):

    print(f"Testing with algorithm: {algorithm} (Run {run})")

    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    experiment_name = (f'solution_figure/{enemy_number}/{algorithm}/{dir_name}/run{run}')

    print(experiment_name)
    n_hidden_neurons = 10
    best_individual_path = os.path.join(experiment_name, 'best_individual.npy')

    if not os.path.exists(best_individual_path):
        print(f'Best individual not found for {algorithm} on Enemy {enemy_number}, Run {run}')
        return None
    best_individual = np.load(best_individual_path)
    print(f'Best individual shape: {best_individual.shape}')
    each_player = []
    each_enemy = []

    for i in range(1,9):
        env = Environment(
            experiment_name=experiment_name,
            enemies=[i],
            multiplemode="no",  # generalist
            playermode="ai",
            player_controller=player_controller(n_hidden_neurons),
            enemymode="static",
            level=2,
            speed="fastest",
            visuals=False
        )
        player_hp, enemy_hp = calculate_gain(env, best_individual)
        each_player.append(player_hp)
        each_enemy.append(enemy_hp)

    return each_player,each_enemy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default='train', help='Mode: train, test, or full')
    parser.add_argument('-n', '--enemy_number', type=int, default=1, help='Enemy number (e.g., 1, 2, 3)')
    parser.add_argument('-a', '--algorithm', type=str, default='GA', choices=['GA', 'DE'], help='Algorithm: GA or DE')
    parser.add_argument('-r', '--runs', type=int, default=1, help='Number of runs per algorithm per enemy')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    args = parser.parse_args()
    np.random.seed(args.seed)

    algorithms = ["NSGA"]

    n_runs = 10
    generations = 30
    combinations = [(1, 3, 4, 6, 7), (3, 4, 7, 8)]  # Final enemy group

    # 300_0.8_0.2_0.5_7 # Final parameter for NSGA-III
    n_p = 300
    p_ratio = 0.8
    m_rate = 0.2
    c_rate = 0.5
    l_m_ratio = 7

    sub_dir = f"{n_p}_{p_ratio}_{m_rate}_{c_rate}_{l_m_ratio}"

    for i in range(len(combinations)):

        enemies = combinations[i]
        results = {'NSGA':  {'best_fitness': {}, 'mean_fitness': {}, 'std_fitness': {}, 'gain': {}}}

        if args.mode in ['train', 'full']:
            for algorithm in algorithms:
                save_name = f'solution_figure/{enemies}/avg/{sub_dir}/{algorithm}'
                if os.path.exists(f'{save_name}/data_f.pkl'):
                    continue
                else:
                    for run in range(1, n_runs + 1):
                        print(f'\n=== Training {algorithm} on Enemy {enemies}, Run {run} ===\n')
                        best_f, mean_f, std_f = train(
                            enemy_number=enemies,
                            algorithm=algorithm,
                            run=run,
                            n_pop=n_p,
                            generations=generations,
                            mutation_rate=m_rate,
                            crossover_rate=c_rate,
                            parent_ratio=p_ratio,
                            lamda_miu_ratio=l_m_ratio,
                            outfix=sub_dir
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

                os.makedirs(save_name, exist_ok=True)
                with open(f'{save_name}/data_f.pkl', 'wb') as file:
                    pickle.dump(results, file)
                if args.mode == 'train':
                    print('\n=== Training Completed ===\n')

        if args.mode in ['test', 'full']:
            with open('figure3.txt', 'a') as file:
                for run in range(n_runs):  # num of runs
                    for algorithm in algorithms:
                        for i in range(5):  # 5 repeats for each independent run
                             print(f'\n=== Testing {algorithm} on All Enemy, Run {run+1} ===\n')
                             p_hp, e_hp = test(
                                 enemy_number=enemies,
                                 algorithm=algorithm,
                                 dir_name=sub_dir,
                                 run=run+1)

                             print('run{}-rep{}'.format(run+1,i+1), "player_hp:", p_hp)
                             print('run{}-rep{}'.format(run+1,i+1), "enemy_hp:", e_hp)
                             count = sum(1 for p, e in zip(p_hp, e_hp) if p > e)
                             print('run{}-rep{}'.format(run+1,i+1), "winner_times:", count)
                             print('run{}-rep{}'.format(run+1,i+1), "sum_player_hp:", round(sum(p_hp), 2))
                             print('run{}-rep{}'.format(run+1,i+1), "total_gain:", round((sum(p_hp) - sum(e_hp)), 2))

                             # save the final gain
                             file.write(f"{run+1}\t{round((sum(p_hp) - sum(e_hp)),3)}\t{enemies}\t{algorithm}\n")

        if args.mode == 'full':
            print('\n=== Full Process Completed ===\n')

if __name__ == '__main__':
    main()
