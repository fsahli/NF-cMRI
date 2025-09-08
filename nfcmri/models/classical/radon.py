# -------------------------------------------------------------------------------------# 
# Functions to calculate the Radon Transform of an MRI image, including coil
# sensitivities. Es similar a src/radon_training.py, pero ese archivo es específico para
# entrenar Implicit Neural Representations (donde puedo evaluar en coordenadas
# arbitrarias), mientras que este sirve para imágenes (donde tengo que interpolar). 
# -------------------------------------------------------------------------------------# 
import jax.numpy as np 
from jax import random, grad, jit, vmap
from jax.numpy.fft import fftshift, fftfreq

from nfcmri.utils.interpolation import interpolate_points_to_grid 
from nfcmri.models.classical.fourier import fastshiftfourier, get_freqs

from functools import partial 

@jit
def normal(alpha): 
  return np.array([np.cos(alpha), np.sin(alpha)])

@jit
def orthogonal(v): 
  vx, vy = v 
  return np.array([-vy, vx])

@jit
def points_in_vector_direction(distances, direction): 
  return distances[:,None] * direction

@jit
def get_ds(ss): 
  return ss[1] - ss[0]

@jit
def linear_reg_coeffs(x,y):
  """
  Return coefficients of linear regression, 
  p[0] + p[1] * x =(approx) y 
  """
  A = np.c_[np.ones(len(x)), x]
  p = np.matmul(np.linalg.pinv(A), y)
  return p 

def print_angle(name, theta):
  print(f"{name}: {theta:.2f} rad, {theta * 180 / np.pi}°")

def calculate_angle(spoke):
  x0, y0 = spoke[0]
  theta = np.arctan2(y0,x0)
  # alpha = np.mod(theta + np.pi / 2, 2 * np.pi)
  alpha = np.mod(theta, 2 * np.pi)
  # x, y = split(spoke) 
  # _, alpha = distance_normal_parametrization_from_coeff(linear_reg_coeffs(x, y)) 
  return alpha 

@partial(jit, static_argnums=[2])
def _radon_points(alpha, lims, N): 
  n = normal(alpha)
  # ss = np.linspace(*lims, N)
  ss = 2 * fftshift(fftfreq(N))
  radonxy = points_in_vector_direction(ss, -n)
  evalradonxy = points_in_vector_direction(ss, orthogonal(n))
  return ss, radonxy, evalradonxy

def radon_points(alpha, N):
    """
        grilla rotada de valores en [0,1]^2
    - alpha () angulo de rotacion
    - N int 
    """
    ss, radonxy, perpendicularxy = _radon_points(alpha, 1., N)
    radonpoints = (radonxy[:,None,:] + perpendicularxy[None,:,:])
    return radonpoints, ss

def rotate(im, alpha):
    """
    - im: shape (frame, px, py). px=py

    rota en las dimensiones de px, py
    
    """
    N = im.shape[-1]
    radonpoints, _ = radon_points(alpha, N)
    vmapped = interpolate_points_to_grid 
    for _ in range(im.ndim-2):
        vmapped = vmap(vmapped, in_axes=(None,0))
    interp_im = vmapped(radonpoints, im)
    return interp_im 

def radon_integration(im, ds, axis=-1):
    return np.sum(im, axis=axis) * ds 

def get_radon_ds_from_N(N):
    ss = 2 * fftshift(fftfreq(N))
    return ss[1] - ss[0]

class ForwardRadonOperator: 
    
    def __init__(self, csmap, spclim:float=0.5):
        """
        - csmap: array (ncoil, px, px)
        - spclim: scaling del k-space
        """
        assert csmap.ndim == 3 
        assert csmap.shape[1] == csmap.shape[2]
        self.csmap = csmap 
        self.N = csmap.shape[1] 
        self.ds = get_radon_ds_from_N(self.N)
        self.spclim = spclim 

    def rotated_csweighted_ims(self, im, alphas):
        """
            Rotates a batch of images, according to a batch of angles (1st im 
            with 1st angle, 2nd im with 2nd angle, etc)
        
        # Arguments 
                
        - im: (batch, px, py)    
        - alphas: (batch,) 

        # Output 
        - rotated_im: shape (batch, ncoils, px, py)
        """
        im = np.moveaxis(im, 0, -1) # (px, py, batch)        
        im = im[None,:,:,:] * self.csmap[:,:,:,None] #(ncoils, px, py, batch)
        im = np.moveaxis(im, -1, 1)  #(ncoils, batch, px, py)
        rotated_im = vmap(rotate, in_axes=(1, 0))(im, alphas) # (batch, ncoils, px, py)
        return rotated_im 

    def radon_transform(self, rotated_im):
        """
        Calcula la transformada de radon para una lista de imagenes, en una lista de ángulos
        
        - im: (batch, ncoils, px, py)    
        - alphas: (batch,) (aunque creo que tambien funciona con (frames, 1), no sé en el caso en que frames = 1 
        """
        im = radon_integration(rotated_im, self.ds, axis=-1) # (batch, ncoils, px)
        im = fastshiftfourier(im * self.spclim * 2)
        return im
    
# --------------------------------------------------------------------------------- #
# k-space filters 
# --------------------------------------------------------------------------------- #

def cosine_filter(N):
   filter_freqs = np.sin(np.linspace(0, np.pi, N, endpoint=False)) # cosine filter 
   return filter_freqs 

def ramp_filter(N):
   filter_freqs = np.ones(N)
   return filter_freqs 

def shepplogan_filter(N):
    filter_freqs = np.fft.fftshift(np.sinc(np.fft.fftfreq(N))) # shepp logan filter 
    return filter_freqs 

FILTERS = {
    'ramp'          : ramp_filter,
    'cosine'        : cosine_filter,
    'shepp-logan'   : shepplogan_filter
}

def get_weight_freqs(N:int, str_filter:str='ramp'):
    """
    Dar un mayor peso a las altas frecuencias 

    - N: int número de puntos en total 
    - str_filter: str en ['ramp', 'cosine' y 'shepp-logan']
    """
    freqs = np.abs(get_freqs(N))
    filter_freqs = FILTERS[str_filter](N)
    return filter_freqs * freqs