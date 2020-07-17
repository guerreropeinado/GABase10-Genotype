import numpy as np
from math import  pi

def f(x):
    return (x*np.sin(6*pi*x)) #Function being optimized

def initialize_population(population_size):
    '''
    Initialize a population randomly
    
    Arguments:
    population_size -- even integer that represents the number of candidate solutions considered in each generation
    
    Returns
    population -- Collection of candidate solutions, each represented by an array where each element represents a digit of the solution
    '''
    population = []
    for j in range(population_size):
        x1 = np.random.uniform(low =-1, high =1) #Value that will determine the sign of the candidate solution
        x2 = np.random.randint(low=0, high=9, size = 5)
        x = np.append(x1,x2) 
        population = np.append(population,x)    
    population = np.split(population, population_size)
    return population


def evaluate_fitness(pop):
    '''
    Evaluates the fitness of each candidate solution in the population
    
    Arguments:
    pop -- the current set of candidate solutions
    
    Returns
    fitness -- an array with the fitness evaluated for each candidate solution
    '''
    population_size = len(pop)
    fitness = []
    for j in range(population_size):
        ind = pop[j]
        if ind[0]>=0:
            sign = 1
        else:
            sign = -1
        x = sign*float(str((int(ind[1])))+"."+str((int(ind[2])))+str((int(ind[3])))+str((int(ind[4])))+str((int(ind[5]))))
        if x<-0.75 or x>1.3: #Solutions that violate the constraints will be penalized with low fitness value
            fit = -1         
        else:
            fit = f(x)
        fitness = np.append(fitness,fit)
    return fitness

def tournament_selection(population, fitness, num_parents):
    '''
    Performs tournament selection to select solutions that will make it to the next generation and
    will be given the chance to produce offspring
    
    Arguments:
    population -- the current set of candidate solutions
    fitness -- evaluated fitness for each candidate solution in the current population
    num_parents -- the number of head-to-head tournaments
    
    Returns
    parents -- a set of candidate solutions that will be used in the next generation and may produce offspring
    '''
    parents =[]
    for i in range(num_parents):
        index1 = np.random.randint(low=0, high=num_parents-1)
        index2 = np.random.randint(low=0, high=num_parents-1)
        #Choose which individual becomes a parent
        if fitness[index1] > fitness[index2]:
            parents = np.append(parents, population[index1])
        else:
            parents = np.append(parents, population[index2])
    parents = np.split(parents, num_parents)
    return parents

def single_point_crossover(parents, num_offspring):
    '''
    Samples, with replacement, from parents and performs single-point crossover to produce offspring for next generation
    
    Arguments:
    parents -- a set of candidate solutions that will be used in the next generation and may produce offspring
    num_offspring -- the number of offspring to produce
    
    Returns
    children -- a set of candidate solutions that will be used in the next generation
    '''
    children = []
    for i in range(num_offspring):
        index1 = np.random.randint(low=0, high=9)
        index2 = np.random.randint(low=0, high=9)
        parent1 = parents[index1]
        parent2 = parents[index2]
        offspring = np.append(parent1[0:2],parent2[2:6])
        children = np.append(children, offspring)
    children = np.split(children, num_offspring)
    return children

def mutation(children):
    '''
    Randomly adjust some genotypes of some of the offspring produced for the next generation
    
    Arguments:
    children -- a set of candidate solutions that will be used in the next generation
    
    Returns
    children -- a set of candidate solutions that will be used in the next generation
    '''
    num_offspring = len(children)
    for i in range(num_offspring):
        for j in range(1,6):
            prob = np.random.uniform(low=0, high=1)
            if prob < 0.5:
                random_value = np.random.randint(low=0, high=9)
                children[i][j] = random_value
    return children

def genetic_algorithm(num_generations, population_size):
    '''
    Perform the genetic algorithm
    
    Arguments:
    num_generations -- the number of iterations to perform the algorithm
    population_size -- even integer that represents the number of candidate solutions considered in each generation 
    
    Returns
    best_OF_value -- the best value of the objective function found by the algorithm
    best_solution -- the candidate solutions giving the best value of the objective function found by the algorithm
    '''
    population = initialize_population(population_size)
    for gen in range(num_generations):
        fitness = evaluate_fitness(population)
        parents = tournament_selection(population,fitness,num_parents =int(population_size/2))
        children = single_point_crossover(parents, num_offspring = int(population_size/2))
        children = mutation(children)
        population  = np.append(parents,children)
        population = np.split(population, population_size)
    fitness = evaluate_fitness(population)
    best_solution_index = np.argmax(fitness)
    best_OF_value = max(fitness)
    ind = population[best_solution_index]
    if ind[0]>=0:
        sign = 1
    else:
        sign = -1
    best_solution = sign*float(str((int(ind[1])))+"."+str((int(ind[2])))+str((int(ind[3])))+str((int(ind[4])))+str((int(ind[5]))))
    return best_OF_value, best_solution


