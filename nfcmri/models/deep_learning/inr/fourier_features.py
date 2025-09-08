import jax.numpy as np
from jax import random, jit

from functools import partial

def concat_last_axis(arraylist): 
  return np.concatenate(arraylist, axis = -1)

def time_mapping(t, m):
  """
  - `m`: si m = 3, entonces usa los periodos
  2pi, 4pi, 6pi, es decir usa los periodos (2npi), para n = 1,...,m
  """
  periods = np.arange(m)+1
  return concat_last_axis([np.sin(2 * np.pi * periods * t), np.cos(2 * np.pi * periods * t)])

def B_matriz(shape, key): 
  return random.normal(key, shape=shape)

def B_matrix(key, input_dims:int, sigma:float, mapping_size:int):
  """
    Fourier Features matriz
  - key: generada con random.PNRGKey(0)
  - input_dims: int, número de coordenadas de entrada que recibe
  - sigma: float > 0
  - mapping_size: number of directions
  """
  B = sigma * random.normal(key, shape=(mapping_size,input_dims))
  return B

@partial(jit, static_argnums=(2,))
def positional_encoding(x, sigma, m): 
  x_proj = 2 * np.pi * x * sigma ** (np.arange(m)/m)
  return concat_last_axis([np.sin(x_proj), np.cos(x_proj)])

def input_mapping(x, B): 
  """
  Aplica fourier features al vector de entrada x con matriz B
  """
  x_proj = (2.*np.pi*x) @ B.T
  return concat_last_axis([np.sin(x_proj), np.cos(x_proj)])

def spatio_temporal_mapping(xt, B):
  """
  Aplica Fourier Features en espacio, multiplicando ademas por informacion temporal.
  - xt: vector de largo 3, de la forma (x,y,t)
  - B: matriz de fourier faetures espacial, de 2 columnas 
  """
  x = xt[:-1]
  t = xt[-1]
  x_proj = (2.*np.pi*x) @ B.T
  t_proj = 2 * np.pi * t 
  return concat_last_axis([np.sin(x_proj)*np.sin(t_proj),
                         np.sin(x_proj)*np.cos(t_proj),
                         np.cos(x_proj)*np.sin(t_proj), 
                         np.cos(x_proj)*np.cos(t_proj)])

def static_mixed_mapping(xt, B_static, B_mixed): 
  """
  Aplica Fourier Features en espacio, además de hacer un spatio_temporal_mapping,
  concatenando todo en un vector.
  - xt: vector de largo 3, de la forma (x,y,t)
  - `B_static`: matriz de fourier features espacial, de 2 columnas 
  - `B_mixed`: matriz de fourier features espacio-temporal, de 2 columnas 
  """
  ff_static = input_mapping(xt[:2], B_static)
  ff_mixed = spatio_temporal_mapping(xt, B_mixed)
  
  return concat_last_axis([ff_static, ff_mixed])

