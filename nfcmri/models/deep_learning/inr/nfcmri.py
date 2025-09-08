import jax.numpy as np 
from jax import random, vmap
from jax.lax import map as laxmap 
from functools import partial
from typing import Sequence

from nfcmri.models.deep_learning.basic_flax import MLP 
from nfcmri.models.deep_learning.inr.coord_maps import GaussianFourierMap, static_mixed_mapping, SpaceTimeCoordinatesMap

from nfcmri.models.classical.radon import radon_points, interpolate_points_to_grid
from nfcmri.utils.mesh import is_inside_of_radial_lim 

def calculate_static_dynamic_len(L:int, static_fraction:float):
    """
    L:int usually x**2: eg. 32, 64, ...,
    static_fraction: float en (0.,1.)
    """
    static_fraction = float(static_fraction)
    desired_ffvector_len = int(L)
    M_static = round(static_fraction * desired_ffvector_len)
    M_dynamic = round((1-static_fraction) * desired_ffvector_len)
    return M_static, M_dynamic

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
    
def STiFF(key, sigma, L, static_fraction):
    key_s, key_d = random.split(key, 2)
    M_static, M_dynamic = calculate_static_dynamic_len(L, static_fraction)
    M_static //= 2; M_dynamic //= 4 
    print(f"M_static: {M_static}, M_dynamic: {M_dynamic}")
    ffencoder_static = GaussianFourierMap(key_s, sigma, M_static)
    ffencoder_dynamic = GaussianFourierMap(key_d, sigma, M_dynamic)
    return partial(stiff_encoder, ffencoder_static=ffencoder_static, ffencoder_dynamic=ffencoder_dynamic)

class NFcMRI:
    def __init__(self, key, sigma:float=7.5, L:int=1000, ps:float=0.8, hidden_layers:Sequence[int]=[512]*3): 
        self.encoder = STiFF(key, sigma, L, ps)
        self.mapper = SpaceTimeCoordinatesMap(2, 1, self.encoder)
        self.net = MLP(hidden_layers, out_dims=2)
    
    def init_params(self, key):
        params = self.net.init(key, np.array(self.mapper.mapping(np.zeros(2), np.zeros(1))))
        return params 

    def eval_frame(self, params, X, ts): 
        """
        - X: array con valores en [-1,1], shape (..., 2)
        - ts: array con valores en [0,1], shape (..., 1)
        """
        ims = self.net.apply(params, self.mapper.mapping(X, ts))
        ims = ims[...,0] + 1j * ims[...,1]
        return ims 
    
    def get_grid(self, N, alpha=0.):
        grid, _ = radon_points(alpha, N) # (nx, nx, 2) 
        return grid 
    
    def rotated_csweighted_im(self,params, t, alpha, csmaps): 
        """
        - t: ()
        - alpha ()}
        - csmap (cs, N, N)
        
        # output 

        - csw_im: (ncoils, px, py)    
        """
        N = csmaps.shape[1]
        radonpoints = self.get_grid(N, alpha)
        is_inside = is_inside_of_radial_lim(radonpoints, 1.) # (nx, nx)
        print("is_inside shape", is_inside.shape)  
        NNx = self.eval_frame(params, radonpoints,t[None]) # (nx, nx)  
        print("NNx shape", NNx.shape)
        print("csmaps shape", csmaps.shape) # (cs, nx, nx)
        rotated_coils = vmap(interpolate_points_to_grid, in_axes=(None,0))(radonpoints,csmaps) # (cs, nx, ny)
        csw_im = rotated_coils * is_inside[None,:,:] * NNx[None,:,:] # (cs, nx, ny)
        return csw_im

    def lax_cine(self, params, ts, batch:int, N):
        """
        - ts: array(float)
        - batch: number of batched images created simultaneously 
        """
        Nt = ts.shape[0]
        grid, _ = radon_points(0., N) # (nx, nx, 2) 
        corrected_Nt = batch * (Nt // batch)
        stacked_ts = ts[:corrected_Nt].reshape(Nt//batch,batch)
        print(stacked_ts.shape)
        predim = laxmap(lambda _ts: self.eval_frame(params, grid[:,:,None,:], _ts[:,None]), stacked_ts) # (Nt/batch, n, n, batch)
        predim = np.moveaxis(predim, -1, 1) # (Nt/batch, batch, n, n)
        predim = np.reshape(predim, (corrected_Nt , N, N))
        predim = np.moveaxis(predim, 0, -1)
        #predim = predim * (1-hollow_mask)[:,:, None]
        return predim 