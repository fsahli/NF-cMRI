import jax.numpy as np
from jax import random, grad, jit, vmap, lax
from jax.numpy.fft import fft, fft2, ifft, ifft2, fftshift, ifftshift, fftfreq
from nfcmri.utils.mesh import meshgrid_from_subdiv, split 

def fastshiftfourier(A):
  print(A.shape[-1])
  return fftshift(fft(ifftshift(A), norm = 'forward'))

def fastinvshiftfourier(A):
  return fftshift(ifft(ifftshift(A), norm= 'forward'))

def shiftfourier(A):
  return fftshift(fft(A, norm = 'forward'))

def invshiftfourier(A):
  return ifft(ifftshift(A), norm= 'forward')

def fourierfreqs(A, ds): 
  return np.fft.fftshift(np.fft.fftfreq(A.shape[-1], d = ds))

def pft(k, im, nx, ny, Nx, Ny): # point fourier transform :)
  return np.sum(im*np.exp(-2.0j*np.pi*(k[0]*nx/Nx + k[1]*ny/Ny)))

def get_freqs(npoints): 
  return -fftshift(fftfreq(npoints)*npoints)

def get_points_in_angle(phase, npoints):
  t = get_freqs(npoints)

  xs = t*np.cos(phase)
  ys = t*np.sin(phase)

  return np.c_[xs, ys]

def fourier_freqs_lims(N): 
  return -(N//2), (N-1)//2 + 1

def in_points_fourier_transform(im, points): 
  Nx, Ny = im.shape
  xl = fourier_freqs_lims(Nx)
  yl = fourier_freqs_lims(Ny)
  nx, ny = np.mgrid[xl[0]:xl[1],yl[0]:yl[1]] # +1 si Nx,Ny impar 

  return vmap(pft, in_axes = (0, None, None, None, None, None))(points, im, nx, ny, Nx, Ny)/ (Nx * Ny)

def radial_fourier_transform(im, phase, samples):
  points = get_points_in_angle(phase, samples)
  return in_points_fourier_transform(im, points)


vmap_radial_fourier_transform = vmap(radial_fourier_transform, in_axes = (None, 0, None))
time_radial_fourier = vmap(vmap_radial_fourier_transform, in_axes = (-1, 0, None))

radial_fourier_lax = lambda im, phs, n: lax.map(
    lambda ph: radial_fourier_transform(im, ph, n), 
    phs
)
def time_radial_fourier_seq(im,phs,n): 
  """
  # Argumentos: 
  - im: array (px,py, frames)
  - phs: angulos, array (frames, ...)
  - n: samples per spoke 
  """
  assert im.shape[-1] == phs.shape[0], "el numero de frames de la imagen y de los angulos debe coincidir"
  return np.stack([radial_fourier_lax(im[...,i], phs[i], n) for i in range(phs.shape[0])])

time_points_fourier = vmap(in_points_fourier_transform, in_axes = (-1, 0))

def notzeromean(array, axis = 0):
  resultshape = np.mean(array, axis = axis).shape
  where_zeros = np.count_nonzero(array, axis = axis) > 0
  promedio = np.zeros(resultshape, dtype = array.dtype)
  return promedio.at[where_zeros].set((np.sum(array, axis = axis) / np.count_nonzero(array, axis = axis))[where_zeros])

@jit
def fourier_img(image):
  return np.fft.fftshift(np.fft.fft2(image, norm = 'forward'))

@jit
def pred_to_list_of_imgs(model_prediction):
  return np.moveaxis(np.squeeze(model_prediction, axis = -1), -1, 0)

@jit
def img_from_fourier(freqs):
  return np.abs(np.fft.ifft2(freqs, norm = 'forward'))

@jit
def mask_imgs(ims, mask):
  """
  Enmascara una lista de imagenes usando la misma mascara mask  
  """
  return ims * mask[None,:,:]

@jit
def generate_img(bs):
  return np.expand_dims(np.stack([img_from_fourier(b) for b in bs], axis = -1), axis = -1)

@jit
def apply_masked_fourier(ims, masks):
  return vmap(fourier_img, in_axes = 0)(ims) * masks 

def get_mixed_freqs(bskspace):
  return np.stack([notzeromean(bskspace, axis = 0) for _ in range(len(bskspace))]) 

def get_moving_mixed_freqs(masks, bskspace):
  """
  generate_img(result) para obtener la imagen movil
  """
  mixed = get_mixed_freqs(bskspace)
  # necesito los indices para cada capa, y reemplazar con el valor correcto en esa capa 
  for i, (mask, bs) in enumerate(zip(masks, bskspace)):
    mixed = mixed.at[i, mask].set(bs[mask])
  return mixed 

def make_image(kspace, axes=(0,1)): 

    shiftedkspace = np.fft.ifftshift(kspace, axes=axes)
    image = np.fft.fftshift(np.fft.ifft2(shiftedkspace, axes=axes, norm='forward'), axes=axes)
    return image 

def make_kspace(img, axes=(0,1)): 
    shiftedimg = np.fft.ifftshift(img, axes=axes)
    kspace = np.fft.fft2(shiftedimg, axes=axes, norm='forward')
    shiftedkspace = np.fft.fftshift(kspace, axes=axes)
    return shiftedkspace
