'''
June 12th, 2022 21:08:00
Omar A. Bracamontes Zavala
omarbracamontes99@gmail.com
'''

# Imports
import torch
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
from pyDOE import lhs
from collections import OrderedDict
from scipy.interpolate import griddata
# from mpl_toolkits.axes_grid1 import make_axes_locatable

# For replication
np.random.seed(1234)

# CUDA support
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Deep Neural Network Class inherited from pytorch
class DNN(torch.nn.Module):
  def __init__(self,layers, activation):
    '''
    Args:
      layers (list): ordered list of number of neurons per layer.
      activation (method): activation function.
    '''
    super(DNN, self).__init__()
    # Parameters
    self.depth = len(layers)-1
    self.activation = activation
    # Set up layers
    layer_list = []
    for i in range(self.depth-1):
      layer_list.append(('layer_%d'%i, torch.nn.Linear(layers[i], layers[i+1])))
      layer_list.append(('activation_%d'%i, self.activation()))
    layer_list.append(('layer_%d'%(self.depth-1), torch.nn.Linear(layers[-2], layers[-1]))) # The last two layers won´t have activation function between them
    # Ordered dictionary of layers
    layerDict = OrderedDict(layer_list)
    # Deploy layers
    self.layers = torch.nn.Sequential(layerDict)

  def forward(self,x):
    return self.layers(x)

# The physics informed neural network: Burger's Equation
class PINN():
  def __init__(self, lb, ub, u, X_u, X_f, nu, layers, activation):
    '''
    Args:
      lb  (tuple): Lower bounds (x,t)
      ub  (tuple): Upper bounds (x,t)
      u   (array): Training data
      X_u (array): Initial and boundary training data (x,t)
      X_f (array): Collocation points for f(x,t)
      nu  (float): diffusion term
      layers      (list): List of neurons per layer
      activation (method): activation function

    '''
    # Boundary conditions
    self.lb = torch.tensor(lb).float().to(device)
    self.ub = torch.tensor(ub).float().to(device)
    # Data
    self.x_u = torch.tensor(X_u[:, 0:1], requires_grad=True).float().to(device)
    self.t_u = torch.tensor(X_u[:, 1:2], requires_grad=True).float().to(device)
    self.x_f = torch.tensor(X_f[:, 0:1], requires_grad=True).float().to(device)
    self.t_f = torch.tensor(X_f[:, 1:2], requires_grad=True).float().to(device)
    self.u = torch.tensor(u).float().to(device)
    # EDP constants
    self.nu = nu
    # Deep Neural Network Parameters
    self.layers = layers
    self.activation = activation

    # Error history
    self.mse_loss = []
    # Initialze Deep Neural Network
    self.dnn = DNN(layers, activation).to(device)
    # Optimizer
    self.optimizer = torch.optim.LBFGS(
      self.dnn.parameters(),
      lr=1.0,
      max_iter=50000,
      max_eval=50000,
      history_size=50,
      tolerance_grad=1e-5,
      tolerance_change=1.0*np.finfo(float).eps,
      line_search_fn='strong_wolfe'
    )
    # Iteration counter
    self.iter = 0
  
  def net_u(self, x, t):
    '''
    DNN for u=u(x,t)
    '''
    return self.dnn(torch.cat([x, t], dim=1))
  
  def net_f(self, x, t):
    '''
    PINN for residual f=f(x,t)
    '''
    # Initialize DNN
    u = self.net_u(x, t)
    # Compute partial derivatives
    u_t = torch.autograd.grad(
      u, t, 
      grad_outputs=torch.ones_like(u),
      retain_graph=True,
      create_graph=True
    )[0]
    u_x = torch.autograd.grad(
      u, x, 
      grad_outputs=torch.ones_like(u),
      retain_graph=True,
      create_graph=True
    )[0]
    u_xx = torch.autograd.grad(
      u_x, x, 
      grad_outputs=torch.ones_like(u_x),
      retain_graph=True,
      create_graph=True
    )[0]
    # Return residual f(x,t)
    return u_t + u*u_x - self.nu*u_xx
  
  def loss_func(self):
    # Initialize optimizer
    self.optimizer.zero_grad()
    # Predict data
    u_pred = self.net_u(self.x_u, self.t_u)
    f_pred = self.net_f(self.x_f, self.t_f)
    # MSE_u
    loss_u = torch.mean((self.u - u_pred)**2)
    # MSE_f
    loss_f =  torch.mean(f_pred**2)
    # MSE
    loss = loss_u + loss_f
    
    loss.backward()
    self.iter += 1
    if self.iter%100==0:
      #print('\n\tIter %d, Loss: %.5e, Loss_u: %.5e, Loss_f: %.5e' % (self.iter, loss.item(), loss_u.item(), loss_f.item()))
      self.mse_loss.append(loss.item())
    # Returns MSE
    return loss

  def train(self):
    # Train DNN
    self.dnn.train()
    # Backward and optimize
    self.optimizer.step(self.loss_func)
  
  def predict(self, X):
    # Data
    x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
    t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)
    # Evaluate
    self.dnn.eval()
    u = self.net_u(x, t)
    f = self.net_f(x, t)
    u = u.detach().cpu().numpy()
    f = f.detach().cpu().numpy()
    # Return predictions
    return u, f

# Clean Dataset
def clean_dataset(data, N_f, N_u):
  t = data['t'].flatten()[:,None]
  x = data['x'].flatten()[:,None]
  Exact = np.real(data['usol']).T # Actual solution
  # Transfom data
  X, T = np.meshgrid(x,t)
  X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
  u_star = Exact.flatten()[:,None]
  # Domain bounds
  lb = X_star.min(0)
  ub = X_star.max(0)    

  xx1 = np.hstack((X[0:1,:].T, T[0:1,:].T))
  uu1 = Exact[0:1,:].T
  xx2 = np.hstack((X[:,0:1], T[:,0:1]))
  uu2 = Exact[:,0:1]
  xx3 = np.hstack((X[:,-1:], T[:,-1:]))
  uu3 = Exact[:,-1:]

  # Final Traing t, x, Exact
  X_u_train = np.vstack([xx1, xx2, xx3])
  X_f_train = lb + (ub-lb)*lhs(2, N_f)
  X_f_train = np.vstack((X_f_train, X_u_train))
  u_train = np.vstack([uu1, uu2, uu3])

  idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
  X_u_train = X_u_train[idx, :]
  u_train = u_train[idx,:]
  return x, t, X, T, Exact, lb, ub, u_train, X_u_train, X_f_train, X_star, u_star

# Main
def run(layers, N_f, N_u, data, plot_results=False):
  layers = np.insert( layers, 0, 2)
  layers = np.append( layers,1)
  print('\n\n',layers)
  # Burgers diffusion term
  nu = 0.01/np.pi
  activation = torch.nn.Tanh
  # Clean 
  x, t, X, T, Exact, lb, ub, u_train, X_u_train, X_f_train, X_star, u_star = clean_dataset(data, N_f, N_u)
  # Model
  model = PINN(lb, ub, u_train, X_u_train, X_f_train, nu, layers, activation) 
  # Train
  print('\tTraining...')
  model.train()
  # Predict
  print('\tTesting...')
  u_pred, f_pred = model.predict(X_star)
  error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
  
  U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
  Error = np.abs(Exact - U_pred)

  #print('\tError U: %e' % (error_u))             

  # Plot
  if plot_results:
    for t in range(0,U_pred.shape[0]):
      fig = plt.figure(figsize=(9,6))
      plt.plot(x,Exact[t,:], 'b-', linewidth = 2, label = 'Exact')       
      plt.plot(x,U_pred[t,:], 'r--', linewidth = 2, label = 'Prediction')
      plt.xlabel('$x$')
      plt.ylabel('$u(t,x)$')
      plt.title('$t=0.{}$'.format(t))
      plt.xlim(-1.1,1.1)
      plt.ylim(-1.1,1.1)  
      plt.grid(alpha=0.4)
      plt.legend()

      plt.savefig(f'./Images/{t}.png', bbox_inches='tight', dpi=300);
      plt.close()
  
  return error_u, model.mse_loss, model.iter, Error