"""
Example of NF-cMRI reconstruction using data from "Replication Data for:
Multi-Domain Convolutional Neural Network (MD-CNN) For Radial Reconstruction of
Dynamic Cardiac MRI" from
https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/CI3WB6
"""
# %%
from jax import random, vmap, jit 
import jax.numpy as np 
import optax
import matplotlib.pyplot as plt

from nfcmri.data.harvard import (
    get_db_config,
    acquisition_from_harvard_data,
    get_bart_dir, safe_mkdir,
    PATIENTS_HEART_LOCATION, # dictionary of heart position (for cropping) of various subjects 
    correctFlip              # dictionary of correct flips of various subjects (for correcting inverted visualizations)
)
from nfcmri.utils.mesh import is_inside_of_radial_lim, meshgrid_from_subdiv
from nfcmri.models.deep_learning.training import simple_train, weighted_loss
from nfcmri.models.deep_learning.inr.nfcmri import NFcMRI 
from nfcmri.models.classical.bart import make_grasp_name 
from nfcmri.models.classical.radon import (
  radon_points,
  interpolate_points_to_grid,
  get_weight_freqs,
  ForwardRadonOperator
)

from nfcmri.evaluation.image_processing import reduce_FOV, BeforeLinRegNormalizer, Flipper
from nfcmri.evaluation.plotting import full_halph_FOV_space_time, NamedReco

key_stiff, key_init, key_train = random.split(random.PRNGKey(0), 3)

# %% ---------------------------------------------------------------------- # 
# (Down)Load de data 
# ------------------------------------------------------------------------- # 

chosen_patient = 'P15'
sub_spokes_per_frame = 8 # retrospective undersampling 

config_data = {
    'chosen_patient'        : chosen_patient, 
    'hermitic_fill'         : True, 
    'relaxed_pad_removal'   : False,
    'sub_spokes_per_frame'  : sub_spokes_per_frame, 
    'tiny_number'           : 1,
}

db_config = get_db_config(config_data)  

# This will download the data if not found in the nfcmri.data_harvard.HARVARD_FOLDER
full_acquisition, undersampled_acquisition = acquisition_from_harvard_data(**db_config)

# %% ---------------------------------------------------------------------- # 
# Fully sampled recon with BART Toolbox for final evaluation 
# ------------------------------------------------------------------------- # 

BART_FOLDER = get_bart_dir(db_config) # or replace with the path where you want to store BART files

safe_mkdir(BART_FOLDER)

bart_acquisitions  = full_acquisition.to_bart(BART_FOLDER, "fullacq")  
reco_fully_sampled = bart_acquisitions.calculate_inufft_reco('inufft')
reco_fully_sampled = np.squeeze(reco_fully_sampled) # (N,N,coils,frames)
reco_fully_sampled = np.sqrt(np.sum(np.abs(reco_fully_sampled)**2, axis=2)) # (N,N, frames) sqrt of sum along coils squared im

del bart_acquisitions, full_acquisition # to save memory 

# %% ---------------------------------------------------------------------- # 
# Get training set for NF-cMRI 
# ------------------------------------------------------------------------- # 

train_X, train_Y = undersampled_acquisition.generate_dataset_angletime()

# %% ---------------------------------------------------------------------- # 
# Calculate coil sensitivity maps 
# ------------------------------------------------------------------------- # 

N       = reco_fully_sampled.shape[0]
NFRAMES = reco_fully_sampled.shape[2]

print(undersampled_acquisition)
undersampled_acquisition.bin_data(sub_spokes_per_frame * NFRAMES) # move all data to a single bin to estimate coils 
print(undersampled_acquisition)
bart_undersampled_acquisition_1bin = undersampled_acquisition.to_bart(BART_FOLDER, f"{sub_spokes_per_frame}spf-1bin")  
bart_coil_sens = bart_undersampled_acquisition_1bin.calculate_coil_sens()
csmaps = bart_coil_sens.to_std_coil() # (coils, N, N)
hollow_mask = bart_coil_sens.hollow_mask() # (N, N) mask zones where csmaps have no sensitivity 
del bart_undersampled_acquisition_1bin
# %% ---------------------------------------------------------------------- # 
# Make recon with classical method GRASP
# ------------------------------------------------------------------------- # 

GRASP_PARAMS = {'lmbda': 0.01, 'lagrangian': 5.0, 'iters': 100}

undersampled_acquisition.bin_data(sub_spokes_per_frame) # move all data to a single bin to estimate coils 
print(undersampled_acquisition)
bart_undersampled_acquisition = undersampled_acquisition.to_bart(BART_FOLDER, f"{sub_spokes_per_frame}spf")  
grasp_reco_name = 'grasp_' + make_grasp_name(GRASP_PARAMS)
reco_grasp = bart_undersampled_acquisition.calculate_bart_reco_with_external_csmappath(bart_coil_sens.obj_path(), grasp_reco_name, **GRASP_PARAMS)
reco_grasp = np.squeeze(reco_grasp) # (N, N, frames)

# %%

net = NFcMRI(key_stiff, L=1000, sigma=7.5, ps=0.2, hidden_layers=[512,512,512])
grid = meshgrid_from_subdiv((N,N), (-1,1), endpoint=True)

params_nfcmri = net.init_params(key_init)
net.eval_frame(params_nfcmri, grid[:,:,None,:], np.array([0.4, 0.6, 0.6])[:,None]).shape # 

# %%

def net_simple_eval(params, t):
  """
    simple evaluation of the network in a grid, at a specific t in [0,1]
  - t: ()
  """
  is_inside = is_inside_of_radial_lim(grid, 1.) # (nx, nx)
  NNx = net.eval_frame(params, grid,t[None]) # (nx, nx)  
  return NNx * is_inside * (1-hollow_mask)

def net_rotated_csweighted_im(params, t, alpha, csmaps): 
  """
    Evaluate network in a rotated grid 
  - t: ()
  - alpha ()
  - csmap (cs, N, N)
  
  # output 

  - csw_im: (ncoils, px, py)    
  """
  N = csmaps.shape[1]
  radonpoints, _ = radon_points(alpha, N) # (nx, nx, 2) 
  is_inside = is_inside_of_radial_lim(radonpoints, 1.) # (nx, nx)
  print("is_inside shape", is_inside.shape)  
  NNx = net.eval_frame(params, radonpoints,t[None]) # (nx, nx)  
  print("NNx shape", NNx.shape)
  print("csmaps shape", csmaps.shape) # (cs, nx, nx)
  rotated_coils = vmap(interpolate_points_to_grid, in_axes=(None,0))(radonpoints,csmaps) # (cs, nx, ny)
  csw_im = rotated_coils * is_inside[None,:,:] * NNx[None,:,:] # (cs, nx, ny)
  return csw_im

# %%

radon_operator = ForwardRadonOperator(csmaps, 0.5)

WEIGHT_FREQS = get_weight_freqs(N, 'ramp') # density compensation 

def loss_nfcmri(params, X, Y): 
    alphas, times = X[:,0], X[:,1]
    rotated_ims = vmap(net_rotated_csweighted_im, in_axes=(None, 0, 0, None))(params, times, alphas, radon_operator.csmap) # (batch, ncoils, n, n)
    spoke_radon_kspace = radon_operator.radon_transform(rotated_ims) # (batch-frame-alpha, ncoils, N)

    Y = Y[...,0] # (frames, cmap, nx)
    print("shapes: spoke ", spoke_radon_kspace.shape, " , Y ", Y.shape)
    loss_value = weighted_loss(spoke_radon_kspace, Y, (1. + WEIGHT_FREQS)[None,None,:])
    return loss_value 

# %%

learning_rate = 2e-3
optimizer = optax.adam(learning_rate)

total_kiter = 3 # 1000 iteraciones 

config_training = {
    'learning_rate': learning_rate,
    'total_kiter': total_kiter,
    'key_train': key_train
}

train_X_stiff = np.stack((train_X[:,0], train_X[:, 1]/NFRAMES), axis=-1)
results = simple_train(jit(loss_nfcmri), train_X_stiff, train_Y, params_nfcmri, optimizer, key_train, batch_size=1, nIter=total_kiter * 1000 + 1)

# %%
final_params = results['last_param']

# %%

from jax.lax import map as laxmap 

get_frame = lambda t: net_simple_eval(final_params, t) # () -> (N,N)
ts = np.linspace(0,1, NFRAMES, endpoint=False)
reco_nfcmri = laxmap(get_frame, ts) # (frames, N, N)
reco_nfcmri = np.abs(np.moveaxis(reco_nfcmri, 0, -1)) # (N, N, frames)
# %%

improc = BeforeLinRegNormalizer(reco_fully_sampled, [0,0,0,0])

crop_ns = PATIENTS_HEART_LOCATION[chosen_patient]
if chosen_patient in correctFlip:
    if not correctFlip[chosen_patient]: 
      improc = Flipper(improc)
      crop_ns = [crop_ns[1], crop_ns[0]] + crop_ns[2:]


named_recos = [
   NamedReco(improc.process(reco_fully_sampled), 'Fully Sampled'), 
   NamedReco(improc.process(reco_grasp),         'GRASP'),
   NamedReco(improc.process(reco_nfcmri),        'NF-cMRI')
]

fig, axs = full_halph_FOV_space_time([named_reco.im for named_reco in named_recos],
  crop_ns, saturation=0.4
)
for ax, named_reco in zip(axs, named_recos):
   ax[0].set_title(named_reco.name)

# %%
