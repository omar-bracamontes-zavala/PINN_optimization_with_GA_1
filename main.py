from PINN import burgers
from GA import genetic_algorithm as ga
from scipy.io import loadmat

# Load training data
data = loadmat('data/burgers_shock.mat')
# Collection data
N_u = 100
N_f = 6000
# Genetic Algorithm parameters
errf = burgers.run
str_error_function = 'Burgers PINN'

lower_bound = 1
upper_bound = 100
p_1 = 5. # Chance of an individual to be mutated (5.)
p_2 = 10. # Chance of a gen to be mutated (10.)


n_dim = 1 #Dimension of every layer: in R^1
hidden_layers_num = [8] # Number of hidden layers in the NN architecture (architecture=individual)

N_individuals =40 * n_dim # Population size (number of individuals)
itermax = 10000

save = False

# For case of a molecule with N particles
for N in hidden_layers_num:
  M_gens = n_dim * N # Individual size
  ga.geneticAlgorithm(errf, str_error_function, lower_bound, upper_bound, p_1, p_2, n_dim, N, N_individuals, M_gens, itermax, save, N_f, N_u, data)




# Deep Neural Network Parameters
#layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
# Train and predict model
# error_u, mse_loss, max_iter, _ = burgers.run(layers, N_f, N_u, data)
# if not error_u > mse_loss[-1]: 
#   print('\nOverfitting Error!')
# print('\n\tMax Iteration:%d, Test Error: %.5e, Training Error: %.5e' % (max_iter, error_u, mse_loss[-1]))
