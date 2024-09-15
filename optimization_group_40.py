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

# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

# evaluation
def evaluate(env, x):
    return np.array(list(map(lambda y: simulation(env,y), x)))


#######    TO DO    #########

### add normalization function to avoid negative probabilities?? ###

### apply limits so that our evolved weight values stay within a defined range  ###

### add function to select parents ###

### add crossover function to generate offspring ###

### OPTIONAL add doomsday function to introduce diversity when population stagnates (idea from optimization_specialist_demo.py) ###



def main():
    # choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"


    experiment_name = 'optimization_test'
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
    n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

    ##########    TO DO    #########
    ### add GA parameters ###
    domain_upper_bound = 
    domain_lower_bound = 
    npop = 
    gens = 
    mutation = 

    ### initialize lists for overall statistics mean and max fitness over 10 generations ###
    mean_fitness_experiments = []
    max_fitness_ex = []

    ### loop for 10 experiments ###

        ### initialize population ###

        ### initialise lists for fitness statistics for this generation and run
        # mean_fitness_generation = []
        # max_fitness_generation = []
    
        ### evolution loop ###

            ### calculate mean and max fitness for this generation and add to list ###
            # add mean to mean_fitness_generation
            # add max to max_fitness_generation
    
            ### generate offspring through crossover ###
            
            ### combine populations (If we choose this strategy) ###
    
            ### select new generation from previously made pool or replace whole generation ###
    
            ### OPTIONAL apply the doomsday if we do end up implementing ###
    
        ### store the results of this run (mean + max fitness) ###
        # add the generational fitness lists into mean_fitness_generation and max_fitness_generation for later calculation including std
        # this would result in a 10 x n array with 10 experiments and n being the amount of generations we choose
    
    ### calculate final statistics for line plot ###
    mean_mean_fitness = 
    std_mean_fitness = 
    mean_max_fitness = 
    std_max_fitness = 

    
    ### create line-plot: generations on the x axis, with the average mean and average maximum (with std) over n generations per run on the y axi



if __name__ == '__main__':
    main()
