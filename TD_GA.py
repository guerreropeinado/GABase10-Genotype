# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 14:33:13 2020

@author: peina
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 12:13:34 2020

@author: peina
"""
import numpy as np
from math import sin, pi
from statistics import mean 
import matplotlib.pyplot as plt


def initialize_population(population_size):
    population = [] #Array that stores each candidate solution
    for j in range(population_size):
        x1 = np.random.uniform(low =-1, high =1) #Random value between -1 and 1
        x2 = np.random.randint(low=0, high=9, size = 5) #array of size 4 with each element from 0 and 9
        x = np.append(x1,x2) 
        population = np.append(population,x)    
    population = np.split(population, population_size)
    return population


def evaluate_fitness(pop):
    population_size = len(pop)
    fitness = []
     #measure the firness of each chromosome
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
            fit = x*np.sin(6*pi*x)
        #Track fitness for the generation
        fitness = np.append(fitness,fit)
    return fitness


def tournament_selection(population, fitness, num_parents):
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
    children = []
    #Generate the next generation using single pointcross-over
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
    num_offspring = len(children)
    #Perform mutation on the offspring
    for i in range(num_offspring):
        for j in range(1,6):
            prob = np.random.uniform(low=0, high=1)
            if prob < 0.5:
                random_value = np.random.randint(low=0, high=9)
                children[i][j] = random_value
    return children


def genetic_algorithm(num_generations, population_size):
    population = initialize_population(population_size)
    for gen in range(num_generations):
        fitness = evaluate_fitness(population)
        parents = tournament_selection(population,fitness,num_parents =int(population_size/2))
        children = single_point_crossover(parents, num_offspring = int(population_size/2))
        children = mutation(children)
        population  = np.append(parents,children)
        population = np.split(population, population_size)
    return population
    
def best_solution(fitness):
    best_OF_value = max(fitness)
    best_solution_index = np.argmax(fitness)
    return best_OF_value, best_solution_index


#
pop = genetic_algorithm(num_generations = 100, population_size=100)
fit = evaluate_fitness(pop)
best_OF, best_index = best_solution(fit)


def f(x):
    return (x*np.sin(6*pi*x))
x = np.arange(-0.75, 1.3, 0.01)
y = f(x)
plt.plot(x,y)
plt.plot(1.051,best_OF,'ro') 
plt.xlabel('Value of x')
plt.ylabel('Value of Function')
plt.title('Objective Function')
plt.legend()
plt.show()







GRANDfitness=np.split(GRANDfitness, 50)
GRANDx=np.split(GRANDx, 50)
minvector = []
maxvector =[]
averagevector = []
xvalues =[]
i = 0
while (i<=49):
    minimum = min(GRANDfitness[i])
    maximum = max(GRANDfitness[i])
    average = mean(GRANDfitness[i])
    minvector = np.append(minvector, minimum)
    maxvector  = np.append(maxvector, maximum)
    averagevector = np.append(averagevector, average)
    max_index = np.where(GRANDfitness[i] == maximum)
    max_x = GRANDx[i][max_index]
    real_max = max(max_x)
    xvalues = np.append(xvalues,real_max)
    i = i + 1
    
generations = np.arange(1, 51, 1).tolist()
plt.plot(generations, averagevector, label = "AVERAGE")
plt.plot(generations, minvector, label = "MINIMUM")
plt.plot(generations, maxvector, label = "MAXIMUM")
plt.xlabel('Genertaion No.')
plt.ylabel('Fitness')
plt.title('Average, Minimum, and Maximum Fitness of Each Generation')
plt.legend()
plt.show()


plt.plot(generations, xvalues)
plt.xlabel('Genertaion No.')
plt.ylabel('Value of Most Fit X in Genration')
plt.title('Best Individual in each Successive Generation')
plt.legend()
plt.show()



