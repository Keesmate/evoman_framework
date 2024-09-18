###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# DEMO : Neuroevolution - Genetic Algorithm  neural network.                  #       			                                         				                                  #
###############################################################################


# imports framework
import sys

from evoman.environment import Environment
from demo_controller import player_controller

# imports other libs
import numpy as np
import matplotlib.pyplot as plt
import os

# runs simulation for each individual and returns their fitness score
def simulation(env,x):
    ''' takes in the vector of an individuals variables, plays them in the environment]
    and returns their fitness value '''
    f,p,e,t = env.play(pcont=x)
    return f

# evaluation returns an array of fitness values for each individual in a population
def evaluate(env, x):
    ''' takes in the population matrix (npop x nvariables) and returns an
    array (length npop) with the fitness of each individual in the population '''
    return np.array(list(map(lambda y: simulation(env,y), x)))


#######    TO DO    #########

### add normalization function to avoid negative probabilities?? ###

### apply limits so that our evolved weight values stay within a defined range  ###

# function to select parents (tournament selection)
def tournament_selection(population, pop_fitness):
    '''
    Implements the tournament selection algorithm.
    It draws twice randomly from the population and returns the fittest individual.
    '''
    # select two indices from the population
    index_1, index_2 = np.random.choice(population.shape[0], 2, replace=False)

    # compare the fitness values of the two individuals and return winner
    if pop_fitness[index_1] > pop_fitness[index_2]:
        return population[index_1]
    else:
        return population[index_2]


# crossover function to generate children #
def recombination(population, pop_fitness, x_percent):
    ''' Implements a discrete recombination method (with .5 probability) between two parents to create children
    until there are 100 - x percent the number of children as there is in the population '''

    children = []

    # loop through 100 - x% the number of the population
    for i in range(int(population.shape[0] * ((100 - x_percent) / 100))):

        # select parents and create child
        parent_1 = tournament_selection(population, pop_fitness)
        parent_2 = tournament_selection(population, pop_fitness)
        child = []

        # loop through each gene of a parent and build child
        for j in range(len(parent_1)):

            if np.random.rand() < 0.5:
                child.append(parent_1[j])
            else:
                child.append(parent_2[j])

        # add new child to children
        children.append(child)

    # return newly made children in array format
    return np.array(children)


# define a function to make the next generation of individuals
def next_generation(population, pop_fitness, x_percent):
    ''' this function takes in the current population and outputs the next generation
    It does this by retaining the top x percent solutions of the current population to exploit current good solutions
    and then recombines parents via tournament selection to creat children to make up the
    rest of the next generation (100 - x% of the population size) '''

    # number of best individuals to keep based on chosen percentage and population size
    n_best = int(population.shape[0] * (x_percent / 100))

    # best x percent indices from population
    best_indices = np.argsort(pop_fitness)[::-1][:n_best]

    # add n best individuals to next generation
    best_individuals = population[best_indices]

    # obtain children and add to next generation to make up the rest of the next generation
    new_children = recombination(population, pop_fitness, x_percent)

    # combine best individuals and the newly created children to make the next generation
    next_generation = np.vstack((best_individuals, new_children))

    return next_generation



### OPTIONAL add doomsday function to introduce diversity when population stagnates (idea from optimization_specialist_demo.py) ###



def main():
    # choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"


    experiment_name = 'optimization_test_group_40'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    n_hidden_neurons = 10

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=experiment_name,
                    enemies=[2],                    #### we can just change the enemy number and run separately three times??
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons), # you  can insert your own controller here
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False)


    # number of weights for multilayer with 10 hidden neurons
    num_variables = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

    ##########    TO DO    #########
    ### add GA parameters ###
    domain_upper_bound = 1
    domain_lower_bound = -1
    pop_number = 100
    generations = 30
    mutation = 0.1
    num_experiments = 10

    ### initialize lists for mean and max fitness over 10 generations + best individual of each run
    mean_fitness_experiments = []
    best_fitness_experiments = []
    best_individuals = []

    # loop for n experiments (Change to 10 when doing actual testing)
    for i in range(num_experiments):

        # initialise lists for fitness statistics for following generations
        mean_fitness_generation = []
        best_fitness_generation = []

        # initialize first population and save first generation statistics
        population = np.random.uniform(domain_lower_bound, domain_upper_bound, (pop_number, num_variables))
        pop_fitness = evaluate(env, population)
        best_fitness_generation.append(np.max(pop_fitness))
        mean_fitness_generation.append(np.mean(pop_fitness))

        # set the best individual and what their fitenss is
        current_best_individual = population[np.argmax(pop_fitness)]
        current_best_individual_fitness = simulation(env,current_best_individual)


        # evolution loop
        for j in range(generations - 1):

            # select the new population and determine their fitnesses
            population = next_generation(population, pop_fitness, 10)
            pop_fitness = evaluate(env, population)

            # add their statistics to statistics lists
            best_fitness_generation.append(np.max(pop_fitness))
            mean_fitness_generation.append(np.mean(pop_fitness))

            # compare best individual of this generation to current best and update if better
            if simulation(env, population[np.argmax(pop_fitness)]) > current_best_individual_fitness:
                current_best_individual = population[np.argmax(pop_fitness)]

            ### OPTIONAL apply the doomsday if we do end up implementing ###

        # store the results of this run
        mean_fitness_experiments.append(mean_fitness_generation)
        best_fitness_experiments.append(best_fitness_generation)
        best_individuals.append(list(current_best_individual))

        print("Run 1 complete")

    # calculate final statistics for line plot
    mean_mean_fitness = np.mean(mean_fitness_experiments, axis=0)
    std_mean_fitness = np.std(mean_fitness_experiments, axis=0)
    mean_max_fitness = np.mean(best_fitness_experiments, axis=0)
    std_max_fitness = np.std(best_fitness_experiments, axis=0)


    ### create line-plot: generations on the x axis, with the average mean and average maximum (with std) over n generations per run on the y axis
    generations = np.arange(len(mean_mean_fitness))
    plt.figure(figsize=(10, 6))
    plt.errorbar(generations, mean_mean_fitness, yerr=std_mean_fitness, fmt='-o', label='Mean Fitness', capsize=3)
    plt.errorbar(generations, mean_max_fitness, yerr=std_max_fitness, fmt='-s', label='Max Fitness', capsize=3)
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.title('Average Mean and Max Fitness Over Generations')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
