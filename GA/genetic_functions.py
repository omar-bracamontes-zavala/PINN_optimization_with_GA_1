'''
Domingo 7 Marzo 2021
Modified: 13 August 2022
Author: Omar A. Bracamontes Z.
'''
# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

def create_individual(lower_bound, upper_bound, M_gens):
  '''
  ABLE TO CREATE A NEW INDIVIDUAL.
    lower_bound float. Domain's lower bound.
    upper_bound float. Domain's upper bound.
  RETURNS.
    individual numpyarray. A individual with continous random M gens (length M)
  '''
  return np.random.randint(lower_bound, upper_bound, M_gens)

def create_population(lower_bound, upper_bound, M_gens, N_individuals):
  '''
  ABLE TO CREATE A POPULATION.
  RETURNS.
    population list. A list with N individuals
  '''
  return [create_individual(lower_bound, upper_bound, M_gens) for _ in range(N_individuals)]

def evaluate(population, errf, *args):
  '''
  ABLE TO EVALUATE EVERY INDIVIDUAL IN THE ERROR FUNCTION.
    population list. A list with N individuals
    errf function. The target function to minimize
  RETURNS.
    _evaluated list. A sorted (ASC) list of ordered pairs like (float, numpy.array)
  '''
  evaluated = []
  for individual in population:
    # Aqui debemos agregar la capa de entrada y se salida (2 y 1)
    test_error_u, training_mse_loss, max_iter, _ = errf(individual, *args)
    print('\n\tMax Iteration:%d, Test Error: %.5e, Training Error: %.5e' % (max_iter, test_error_u, training_mse_loss[-1]))
    # If not overfitted
    if test_error_u > training_mse_loss[-1]:
      evaluated.append( (test_error_u, individual) )
    else:
      print('\n\tOverfitting Penalty Applied\n')
      evaluated.append( (np.inf, individual) )
    ordered = sorted(evaluated, key=lambda tup:tup[0])
    print('ordered', ordered)
  return ordered#sorted(evaluated, key=lambda tup:tup[0])

def termination_criteria(errors_per_iter, errors, epsilon, iteration, itermax):
  '''
  THE ALGORITHM TERMINATES IF THE CRITERIA IS TRUE.
  errors_per_iterlist. List of lists. #se puede mejorar pero por lo pronto asi
  errors list. A sorted (ASC) list.
  epsilon float. Termination criteria
  iteration int. Current iteration
  itermax int. Termination criteria
  RETURNS.
  boolean
  '''

  if (iteration > itermax):
    print('\nTerminationCriteria: iter>itermax\n')
    return True
  elif (abs( np.min(errors)-np.max(errors) ) < epsilon) and np.min(errors)<0:
    print('\nTerminationCriteria: |f_l - f_h| < e\n')
    return True
  elif (iteration > 0) and (np.var(errors_per_iter[iteration])<(np.var(errors_per_iter[iteration-1])/3.)) and np.min(errors)<0:
    print('\nTerminationCriteria: var_i < var_{i-1}/2\n')
    return True
  else:
      return False

def crossover(evaluated_population, cutoff_point, M_gens, N_individuals):
  '''
  ABLE TO EXCHANGE BESTS INDIVIDUALS' GENETIC MATERIAL IN ORDER TO CREATE CHILDS
  evaluated_population list. A sorted list of numpy_arrays
  cutoff_point int. The cutoff point of the genetic material
  RETURNS.
  childs list. A list of numpy arrays with the new individuals.
  '''
  half = N_individuals//2
  parents = evaluated_population[:half]
  childs = [np.zeros(M_gens) for _ in range(half)]
  print('childs: ',childs, len(childs))
  print('parents: ', parents, len(parents))
  for i in range(0, len(parents), 2):
    if i+1 <= len(parents):
      print('i: ',i)
      childs[i][:cutoff_point] = parents[i][:cutoff_point]
      childs[i][cutoff_point:] = parents[i+1][cutoff_point:]
      childs[i+1][:cutoff_point] = parents[i+1][:cutoff_point]
      childs[i+1][cutoff_point:] = parents[i][cutoff_point:]
  print('childs: ',childs, len(childs),'\n')
  return childs

def mutation(childs, p_1, p_2, M_gens, lower_bound, upper_bound):
  '''
  ABLE TO MODIFY 10% OF THE GENS OF THE 5% OF THE POPULATION.
  childs list. A list of numpy arrays
  p_1 float. Chance of affecting p_1 of the population
  p_2 float. Chance of affecting p_2 of the gens of an individual.
  RETURNS.
  childs list. Modified childs
  '''
  for child in childs:
    if np.random.uniform(0,100) <= p_1:
      for gen in range(M_gens):
        if np.random.uniform(0,100) <= p_2:
          child[gen] = np.random.uniform(lower_bound, upper_bound)

  return childs

def selection_replacement(evaluated_population, childs, errf, N_individuals, *args):
  '''
  ABLE TO SORT THE POPULATION AND SELECT THE BEST N_individuals
  evaluated_population list. The current population
  childs list. The childs  of the bests individuals.
  RETRUNS.
  population list. The new sorted population
  '''
  _current_population = evaluated_population + childs
  _current_eval_pop = evaluate(_current_population, errf, *args)
  _selected = _current_eval_pop[:N_individuals]
  return [indiv[1] for indiv in _selected]

def local_technique(population, M_gens, lower_bound, upper_bound, errf, *args):
  '''
  ABLE TO CREATE A TEST INDIVIDUAL AND COMPARE ITS PERFORMANCE VS WORST INDIVIDUAL PERFORMANCE
  population list. A sorted list of N_individuals
  RETURNS
  population list. A new population.
  '''
  foreing_individual = create_individual(lower_bound, upper_bound, M_gens)
  worst_individual = population[-1]
  evaluated_foreing_indiv = evaluate([foreing_individual], errf, *args)
  evaluated_worst_indiv = evaluate([worst_individual], errf, *args)

  if evaluated_foreing_indiv[0][0] <= evaluated_worst_indiv[0][0]:
    population[-1] = foreing_individual

  return population

def plot_errors(generations, means, bests, worsts, str_error_function, n_dim, N, save=False):
  fig, ax = plt.subplots()
  fig.suptitle(f'Optimizing {str_error_function} in R^{n_dim} with GA\n N={N}\nBestError: {bests[-1]}')

  #ax.plot(generations, means,'b', alpha=0.8, label='Mean Error')
  ax.plot(generations, bests,'g', alpha=1.0, label="Elite Error")
  #ax.plot(generations, worsts,'r', alpha=0.6, label="Dreg Error")

  ax.grid(alpha=0.5)
  ax.set_ylabel(r'$Energy/\epsilon$')
  ax.set_xlabel('Generation')
  ax.legend()

  if save==True:
    plt.savefig(f'./optimization_figs/{N}N_E{bests[-1]}.png')

def save_results_csv(bests_individual_per_generation,bests_individual_error_per_generation,N,itermax,N_individuals):
  final_best_dict = {'positions':bests_individual_per_generation[-1]}
  final_best = pd.DataFrame(final_best_dict)
  final_best.to_csv('./optimization_logs/{}N_E{}_g{}_p{}.csv'.format(N,bests_individual_error_per_generation[-1],itermax,N_individuals), sep = ' ', index = False)

