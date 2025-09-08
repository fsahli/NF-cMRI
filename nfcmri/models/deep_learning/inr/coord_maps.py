import jax.numpy as np 
from jax import vmap 
from typing import Tuple 
from abc import ABC, abstractmethod
from nfcmri.models.deep_learning.inr.fourier_features import B_matrix, input_mapping, static_mixed_mapping

def get_linear_coordinate_mapping(xs_origin, xs_new):
    x0, x1 = xs_origin
    y0, y1 = xs_new 
    def L(x): 
        y = (y1 - y0)/(x1-x0) * (x-x0) + y0
        return y 
    return L 

class CoordLinearMapper:
    def __init__(self, coord_lims:Tuple[float,float,float,float]):
        """
        - coord_lims = xd, xu, yl, yr 
        """
        self.xd, self.xu, self.yl, self.yr = coord_lims
        self.x_map = get_linear_coordinate_mapping((self.xd, self.xu), (-1,1))
        self.y_map = get_linear_coordinate_mapping((self.yl, self.yr), (-1,1))

    def mapping(self, X):
        """
        - X.shape (...,2)
        """
        x, y = X[...,0], X[...,1]
        new_x_coords = self.x_map(x)
        new_y_coords = self.y_map(y)
        new_coords = np.stack((new_x_coords, new_y_coords), axis=-1)
        return new_coords

    def get_coord_lims(self):
        coord_lims = self.xd, self.xu, self.yl, self.yr
        return coord_lims

class CoordinateEncoding(ABC):

    @abstractmethod
    def coordmap(x):
        pass 

class IdentityMap(CoordinateEncoding):
    def coordmap(x):
        return x 

class GaussianFourierMap(CoordinateEncoding): 

    def __init__(self, key, sigma:float=1.0, mapping_size:int=50, dims:int = 2): 
        self.sigma = sigma 
        self.mapping_size = mapping_size 
        self.dims = dims 
        self.B = B_matrix(key, dims, sigma, mapping_size)

    def coordmap(self, x):
        return input_mapping(x, self.B)
    
# ######### Including time 

def helix_map(ts, total_cycles = 1):
    helix = np.stack([np.cos(2 * np.pi * ts), np.sin(2 * np.pi * ts), ts/total_cycles], axis=-1)
    return helix

def circle_map(ts):
    helix = np.stack([np.cos(2 * np.pi * ts), np.sin(2 * np.pi * ts)], axis=-1)
    return helix

def space_helix_concat(X,T): 
    """
    X: (x,): posicion en espacio (en la grilla), usualmente x=2 (2D)
    T: (2,): dos tiempos en [0,1] (origen y destino)
    """
    t0, t1 = T 
    concated = np.concatenate([X, helix_map(t0), helix_map(t1)], axis=-1)
    return concated 

def space_simple_circle_concat(X,t): 
    """
    X: (x,): posicion en espacio (en la grilla), usualmente x=2 (2D)
    t: (1,): un tiempo en [0,1] 
    """
    concated = np.concatenate([X, np.cos(2 * np.pi * t), np.sin(2 * np.pi * t)], axis=-1)
    return concated 

def space_double_circle_concat(X,T): 
    """
    X: (x,): posicion en espacio (en la grilla), usualmente x=2 (2D)
    T: (2,): dos tiempos en [0,1] (origen y destino)
    """
    t0, t1 = T 
    concated = np.concatenate([X, circle_map(t0), circle_map(t1)], axis=-1)
    return concated 

def space_time_simple_concat(X,T): 
    """
    X: (x,): posicion en espacio (en la grilla), usualmente x=2 (2D)
    T: (t,): dos tiempos en [0,1] (origen y destino)
    """
    concated = np.concatenate([X, T], axis=-1)
    return concated 

def gaussian_space_plus_helix_time_encoder(X, T, ff_encoder:GaussianFourierMap):
    """
    X: (ff_encoder.dims,): posicion en espacio (en la grilla)
    T: (2,): dos tiempos en [0,1] (origen y destino)
    """
    ffX = ff_encoder.coordmap(X)
    t0, t1 = T 
    concated = np.concatenate([ffX, helix_map(t0), helix_map(t1)], axis=-1)
    return concated

def gaussian_space_plus_time_encoder(X, T, ff_encoder:GaussianFourierMap):
    """
    X: (ff_encoder.dims,): posicion en espacio (en la grilla)
    T: (t,): tiempos en [0,1], usualmente se usa reference (t=1) o reference y template (t=2)
    """
    ffX = ff_encoder.coordmap(X)
    concated = np.concatenate([ffX, T], axis=-1)
    return concated

def stiff_encoder(X,T, ffencoder_static:GaussianFourierMap, ffencoder_dynamic:GaussianFourierMap): 
    assert X.ndim == 1
    assert T.ndim == 1
    assert X.shape[0] == ffencoder_static.dims 
    assert X.shape[0] == ffencoder_dynamic.dims
    assert T.shape[0] == 1

    Bs = ffencoder_static.B
    Bd = ffencoder_dynamic.B
    xt = np.concat((X, T), axis=0)
    return static_mixed_mapping(xt, Bs, Bd)
    
class SpaceTimeCoordinatesMap: 
    def __init__(self, space_in_dim:int, time_in_dim:int, coordinate_encoder):
        """
        coordinate_encoder: (space_in_dim)(time_in_dim)->(time_in_dim)
        """ 
        self.space_in_dim = space_in_dim
        self.time_in_dim = time_in_dim
        self.encoder = coordinate_encoder
        dummy_x = np.ones(space_in_dim)
        dummy_t = np.ones(time_in_dim)
        dummy_encoded = coordinate_encoder(dummy_x, dummy_t)
        self.out_dim = dummy_encoded.shape[-1]
        self.vectorized_encoded = np.vectorize(self.encoder, signature=f'({self.space_in_dim}),({self.time_in_dim})->({self.out_dim})')

    def mapping(self, X, T):
        """
        - X.shape (...,space_in_dim)
        - T.shape (...,time_in_dim)
        """
        return self.vectorized_encoded(X, T)
    
    def grid_mapping(self, X, T):
        """
        - X.shape (nx, ny ,space_in_dim)
        - T.shape (time_in_dim,)
        """
        return vmap(vmap(self.encoder, in_axes=(0,None)), in_axes=(0,None))(X, T)

    def vec_mapping(self, X, T):
        """
        - X.shape (nx, ny ,space_in_dim)
        - T.shape (time_in_dim,)
        """
        return vmap(self.encoder, in_axes=(0,None))(X, T)