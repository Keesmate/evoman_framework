''' in this script we can execute the last part of the assignment, which is to compare our algorithms by enemy
- test our final best solution for each of the 10 independent runs and calculate their individual gains
- repeat this 5 times and get the average individual gains for each of the 10 independent solutions
- do this for each of the 3 enemies and for both of our two algorithms
- create boxplots where the average individual gain of each solution are the points
- resluting in a total of 3 pairs of box-plots (6 boxes), being one pair per enemy

- Additionally, do a statistical test to verify if the differences in the average of these means are significant between the groups of best solutions, when comparing two algorithms of an enemy.
- t-test or Mann-Whitney U test depending on the spread of the data


'''
'''
#####    TO DO  #####

###  function 1 ###
Define a function to take in the env (to be specified in function below) and a list of 10 best solutions. runs each solution through the env 
5 times and calculates mean individual gains of each solution and returns this as a list:


###   function 2   ####
Define a function evaluate_final_best_solutions():

    Define the list of 10 best solutions for Algorithm 1
    Define the list of 10 best solutions for Algorithm 2

    ### loop through 3 enemies ###
      
        initialize game environment with the current enemy and specific parameters
        env = ....

        get mean individual gains for algorithm 1 by calling above function
        get mean individual gains for algorithm 2 by calling above function

        create a box plot to compare the mean individual gains of each algorithm for the current enemy

        perform a t-test 
        



    '''
