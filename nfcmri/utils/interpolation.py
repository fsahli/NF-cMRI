from jax.scipy.interpolate import RegularGridInterpolator as RGI
import jax.numpy as np

def interpolate_points_to_grid(points, csmap): 
  """
  - points: narray [...,2]
  - csmap: array 2d (px,py)
  """
  Nx, Ny = csmap.shape[:2]
  x, y = np.linspace(-1,1,Nx, endpoint = False), np.linspace(-1,1,Ny, endpoint = False)
  interpolator = RGI((x,y), csmap, bounds_error = False, fill_value = 0.)
  return interpolator(points)
