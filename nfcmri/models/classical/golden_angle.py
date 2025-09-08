import jax.numpy as np
from jax import vmap

def tiny_golden_angle(N):
  goldenratio = (1 + np.sqrt(5))/2
  return np.pi/(goldenratio + N - 1)

def nth_tiny_golden_angle(N,n):
  return np.mod(n * tiny_golden_angle(N), 2 * np.pi)

# np.reshape(np.arange(20), (5,4))
def tiny_golden_angles_in_time(timeframes, samples_per_frame, tiny_number = None):
  """
  genera una secuencia de angulos partiendo en 0 y avanzando
  `tiny_golden_angle(N)`, con `N = tiny_number` si `tiny_number` es dado. Si 
  no, usa `N = samples_per_frame`.
  Agrupa `samples_per_frame` de estos angulos en cada frame, con un total de 
  `timeframes` frames.
  ## Return 
  array de forma (timeframes, samples_per_frame)
  """
  tiny_number = tiny_number if tiny_number is not None else samples_per_frame
  angles = vmap(nth_tiny_golden_angle, in_axes= (None,0))(tiny_number,np.arange(samples_per_frame * timeframes))
  return np.reshape(angles, (timeframes, samples_per_frame))

def add_times_to_phases_array(grid_t, phases):
  assert grid_t.shape[0] == phases.shape[0]
  return np.concatenate([np.c_[t * np.ones(phases_at_t.shape[0]), phases_at_t] for t, phases_at_t in zip(grid_t, phases)], axis = 0)