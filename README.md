Evoman is a video game playing framework to be used as a testbed for optimization algorithms.

A demo can be found here:  https://www.youtube.com/watch?v=ZqaMjd1E4ZI


# Dummy Optimization for EvoMan

## 1. Purpose

This project focuses on evolving AI controllers for the EvoMan game environment, using evolutionary computation methods. In Assignment 1, we implemented and compared two optimization algorithms: Genetic Algorithm (GA) and Differential Evolution (DE), which were used to train AI controllers against a single enemy.

In Assignment 2, we extend the approach to multi-objective optimization, training AI controllers against multiple enemies. For this, we upgraded the algorithms as follows:

	•	GA was upgraded to NSGA-II, a multi-objective version of the Genetic Algorithm from the DEAP library (https://deap.readthedocs.io/en/master/).
	•	DE was upgraded to DEMO (Differential Evolution for Multi-objective Optimization), by modifying the selection mechanism to handle multiple objectives.


## 2. Features

The main features of this project include:

	- Implementation of NSGA-II for multi-objective optimization in Genetic Algorithm.
	- Upgrade of Differential Evolution to the DEMO algorithm.
	- Training and testing AI controllers against both single and multiple enemies.
	- Hyperparameter tuning using grid search functionality.
	- Results generation with detailed metrics such as best fitness, mean fitness, and multi-objective trade-offs.


## 3. Setup and Execution

### Prerequisites

- Python 3.x
- NumPy
- SciPy
- EvoMan framework (ensure proper installation and configuration)
- 
### Setup

1. Clone or download this project to a local directory.
2. Ensure the EvoMan framework is correctly installed and can be imported in your Python environment.
3. install deap, numpy and scipy

### Execution
   `
   python optimization_generalist_Tianyi_version.py -m train -n 1 -r 5
   `
   This will train on enemy 1 for 5 runs.

   To implement NSGA-III training and test, use the following commands:
   `
   python NSGA3.py -m train
   `
    
    for training enemies on both enemy group sets.
   `
   python NSGA3.py -m test
   `
    
    for testing enemies on all enemies.
    For parameter tuning for NSGA-III, use:
   `
   python Tuning.py -train
   `
    


### Parameter Descriptions

- `-m` or `--mode`: Execution mode (`train`/`test`/`full`)
- `-n` or `--enemy_number`: Enemy number
- `-a` or `--algorithm`: Algorithm to use (`DE`/`NSGA2`/`MODE`)
- `-r` or `--runs`: Number of runs
- `--seed`: Random seed
- `--grid_search`: Start the grid search mode

After execution, the results will be saved in a `grid_search_results.csv` file.

