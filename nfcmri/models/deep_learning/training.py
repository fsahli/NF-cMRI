import jax.numpy as np
from jax import random, grad, jit
from tqdm import tqdm # mostrar barras de progreso 
from jax.tree_util import tree_leaves
from jax.flatten_util import ravel_pytree
from typing import Dict, Callable 

def step(loss, i, opt_state, X_batch, Y_batch, get_params, opt_update):
    params = get_params(opt_state)
    g = grad(loss)(params, X_batch, Y_batch)
    return opt_update(i, g, opt_state)

def simple_loss(measures_net_X, measures_Y):
  # return np.mean((np.abs(measures_net_X) - np.abs(measures_Y))**2)
  return np.mean(np.abs(measures_net_X - measures_Y)**2)

def weighted_loss(X,Y,W): 
  return np.mean((np.abs(X - Y) * W)**2)

@jit
def mse(X,Y): 
  return np.mean(np.power(np.abs(X - Y), 2))

@jit
def psnr(X,Y): 
  return -10 * np.log10(mse(X,Y))

import optax 
import jax 

def simple_train(loss, X, Y, params, optimizer, key, batch_size = 5000, nIter = 5000):
    opt_state = optimizer.init(params)

    #@partial(jax.jit, static_argnums=(3,)) # yo diria que 0 
    def step(loss, params, opt_state, batchX, batchY):
        loss_value, grads = jax.value_and_grad(loss)(params, batchX, batchY)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    train_loss = []
    best_loss = np.inf
    iterations = []
    for it in tqdm(range(nIter), desc='train iter', leave=True):
        key, subkey = random.split(key)
        idx_batch = random.choice(subkey, X.shape[0], shape = (batch_size,))#, replace = False)# 
        params, opt_state, train_loss_value = step(loss, params, opt_state, X[idx_batch], Y[idx_batch])
        if train_loss_value < best_loss: 
            best_loss = train_loss_value
            best_param = params           
        if it % 100 == 0:
            train_loss.append(train_loss_value)     
            to_print = " it %i, train loss = %e" % (it, train_loss_value)
            print(to_print)

            iterations.append(it)
    results = {
      'last_param': params, 
      'best_param': best_param,
      'train_loss': train_loss,
      'best_loss': best_loss,
      'iterations': iterations
    }
    return results 

def count_params(params, magnitude_term =1e6): 
    nparams = sum([arr.size for arr in tree_leaves(params)])
    return nparams / magnitude_term

def mean_abs(tree):
    return np.mean(np.abs(ravel_pytree(tree)[0]))

class LossEvaluator:
    """
    
    ## Ejemplo de uso 

    Supongamos que tenemos previamente definidas las funciones de pérdida 
    `loss_registration(params, im1, im2)` y `loss_residual(params, key)`.
    Además hemos entrenado una red y hemos obtenido una evolución de parámetros,
    guardados en `results['param_history']`, bajo los nombres 'param-0', 'param-1', ....
    Podemos usar el evaluador de la siguiente forma: 

    ```
    evaluator = LossEvaluator({
        'registration': loss_registration,
        'residual': loss_residual
        })

    k_params = [results['param_history'][f'param-{it}'] for it in range(3)]
    arg_dic = {
            'registration': (im1, im2),
            'residual': (key,)
        }

    l_vals = evaluator.evaluate_losses(k_params, arg_dic)
    g_vals = evaluator.grad_evo(k_params, arg_dic)
    ```
    """

    def __init__(self, loss_dict:Dict[str, Callable]):
        """
        loss_dict: 
            {'loss_name': loss_func(params, *args)}
        """
        self.loss_dict = loss_dict

    def evaluate_losses(self, param_list, args_losses):
        loss_vals = {
            loss_name: [loss_func(params, *args_losses[loss_name]) for params in param_list] for loss_name, loss_func in self.loss_dict.items()
        }
        return loss_vals

    def grad_evo(self, param_list, args_losses): 
        evo_loss = {
            loss_name: [mean_abs(jax.value_and_grad(loss_func)(params, *args_losses[loss_name])[1]) for params in param_list] for loss_name, loss_func in self.loss_dict.items()
        }
        return evo_loss