from GA import genetic_functions as gaf # genetic.py with the gentic functions
import numpy as np
import pandas as pd
import csv
import logging
logging.basicConfig(level=logging.INFO)

from numpy import sqrt

logger = logging.getLogger(__name__)

# Genetic algorithm
def geneticAlgorithm(errf, str_error_function, lower_bound, upper_bound, p_1, p_2, n_dim, N, N_individuals, M_gens, itermax, save, *args):

    generation = 0
    finished = False
    epsilon = 1.0*np.finfo(float).eps#10**(-14)#M_gens/N_individuals # For termination criteria

    logger.info('\nInitializePopulation\n')

    # Logs
    generations = []
    all_errors_per_generation = [] #se puede mejorar pero por lo pronto asi
    bests_individual_per_generation = []
    worsts_individual_per_generation = []
    mean_individual_error_per_generation = []
    bests_individual_error_per_generation = []
    worsts_individual_error_per_generation = []

    # Initialize population
    population = gaf.create_population(lower_bound, upper_bound, M_gens, N_individuals)
    while not finished:
        print('Generation: ', generation)
        # Evaluate
        _evaluated = gaf.evaluate(population, errf, *args)
        population_errors = [indiv[0] for indiv in _evaluated]
        evaluated_population = [indiv[1] for indiv in _evaluated]
        # Logs
        all_errors_per_generation.append(population_errors) #se puede mejorar pero por lo pronto asi
        bests_individual_per_generation.append(evaluated_population[0])
        worsts_individual_per_generation.append(evaluated_population[-1])
        mean_individual_error_per_generation.append( np.mean(population_errors) )
        bests_individual_error_per_generation.append( np.min(population_errors) )
        worsts_individual_error_per_generation.append( np.max(population_errors) )
        # Termination criteria
        finished = gaf.termination_criteria(all_errors_per_generation,population_errors, epsilon, generation, itermax)
        # Crossover
        childs = gaf.crossover(evaluated_population, M_gens//2, M_gens, N_individuals)
        # Mutation
        childs = gaf.mutation(childs, p_1, p_2, M_gens, lower_bound, upper_bound)
        # Selection/Replacement
        population = gaf.selection_replacement(evaluated_population, childs, errf, N_individuals, *args)
        # Local Technique (optional)
        population = gaf.local_technique(population, M_gens,lower_bound, upper_bound, errf)
        # New Generation
        generations.append(generation)
        generation += 1

    # Print logs
    print(f'\n----\Hidden Layers N = {N}\t----')
    print(f'\nBest Test Error: {bests_individual_error_per_generation[-1]}\nSolution Layer:\n{bests_individual_per_generation[-1]}\n')


    # Plot results
    gaf.plot_errors(generations, mean_individual_error_per_generation, bests_individual_error_per_generation, worsts_individual_error_per_generation,str_error_function, n_dim, N, save)
    if save:
        # Save logs into csv
        gaf.save_results_csv(bests_individual_per_generation,bests_individual_error_per_generation,N,itermax,N_individuals)