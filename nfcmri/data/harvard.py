"""
    Harvard Patients Data Processing   
Example of NF-cMRI reconstruction using data from "Replication Data for:
Multi-Domain Convolutional Neural Network (MD-CNN) For Radial Reconstruction of
Dynamic Cardiac MRI" from
https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/CI3WB6
"""

from jax import numpy as np 
from jax import vmap, lax 
import scipy.io
from os import system as system_call
from os.path import join 
from pathlib import Path 
from nfcmri.models.radial_acquisitions import radial_acquisition_from_angles
from nfcmri.models.classical.golden_angle import tiny_golden_angles_in_time 
from nfcmri.models.classical.bart import bart_read, bart_acquisition_from_files, calculate_hollow_mask
from nfcmri.evaluation.image_processing import normalize

# ---------------------------------------------------------------------#
# Setup at install
# ---------------------------------------------------------------------#
HARVARD_FOLDER = '/home/tabita/ACIP-MRI/NF-cMRI/harvardDB' # <---------- replace with a absolute path to the directory where you want to store the data
BART_DIRECTORY = '/home/tabita/ACIP-MRI/NF-cMRI/data_bart' # <---------- replace with path to directory to store bart outputs

#----------------------------------------------------------------------#
# Downliad the data
#----------------------------------------------------------------------#

# patient number -> database id 
HARVARD_DB_IDs = {
    "P01":"XS8ZVS",
    "P02":"SAZBU4",
    "P03":"OAU0G8",
    "P04":"2HQMS7",
    "P05":"GM5RRH",
    "P06":"WUEVSL",
    "P07":"5U3MJC",
    "P08":"NFVU9D",
    "P09":"DPBPPW",
    "P10":"ECX7ZC",
    "P11":"A7ID8Q",
    "P12":"GHS2UI",
    "P13":"W1CKBM",
    "P14":"AQX7VV",
    "P15":"OFDWXA",
    "P16":"QGM2LC",
    "P17":"R1Q3FF",
    "P18":"39C8RJ",
    "P19":"UVOHGU",
    "P20":"SNB4GA",
    "P21":"WWC5UV",
    "P22":"XZEIOO",
    "P23":"TST0PE",
    "P24":"PUKAW1",
    "P25":"W0TPJQ",
    "P26":"MPNLPN",
    "P27":"JRRKB9",
    "P28":"OXIDNU",
    "P29":"HMC3VV",
    "P30":"OC1DZJ",
    "P31":"UH83YA",
    "P32":"PNG3SA",
    "P33":"ACHW2E",
    "P34":"MASIMI",
    "P35":"PWBGYH",
    "P36":"PLJHVF",
    "P37":"0ZHZEC",
    "P38":"P2BSYS",
    "P39":"4AAWVJ",
    "P40":"YOBKU3",
    "P41":"LDOLKI",
    "P42":"TCD5Q7",
    "P43":"I8TDNN",
    "P44":"VOY57B",
    "P45":"XDEPKI",
    "P46":"TIMBRL",
    "P47":"GKNB8Y",
    "P48":"JIQLBF",
    "P49":"9WJET3",
    "P50":"9OR9BN",
    "P51":"81LZHW",
    "P52":"XKMNYR",
    "P53":"LEY4SD",
    "P54":"XRIH3C",
    "P55":"Q8LCJH",
    "P56":"EHNQIX",
    "P57":"FLRG4L",
    "P58":"ZCHVEG",
    "P59":"3GI5PS",
    "P60":"FLP20P",
    "P61":"AMCAOI",
    "P62":"5MLZOH",
    "P63":"IAJ755",
    "P64":"5FCQFZ",
    "P65":"JREWJH",
    "P66":"5KFR78",
    "P67":"38H58Q",
    "P68":"HOAUX1",
    "P69":"WBEOKY",
    "P70":"WHVHYN",
    "P71":"BTOYJV",
    "P72":"7TKQCV",
    "P73":"EU99LY",
    "P74":"MM8YRI",
    "P75":"ORBWER",
    "P76":"UQF5PA",
    "P77":"KRGBKE",
    "P78":"XY4CJJ",
    "P79":"CCQ5BN",
    "P80":"JEEUP1",
    "P81":"HQJI62",
    "P82":"XQPKKK",
    "P83":"YASVIW",
    "P84":"CAXEUE",
    "P85":"QCEZE5",
    "P86":"X6SO3X",
    "P87":"CHLSWY",
    "P88":"E7JQG5",
    "P89":"E3QCJ3",
    "P90":"0MQ5OR",
    "P91":"YNI49H",
    "P92":"OFXMG7",
    "P93":"HCM4TY",
    "P94":"MVDXUX",
    "P95":"ZMK7AO",
    "P96":"XNZGVN",
    "P97":"HJHTRD",
    "P98":"DACRC4",
    "P99":"6PH7ZY",
    "P100":"HR66QD",
    "P101":"EUX23S",
    "P102":"O4EBZT",
    "P103":"FZHQGO",
    "P104":"ZNCHMT",
    "P105":"8THAMD",
    "P106":"WIX4ZX",
    "P107":"FNGJ7C",
    "P108":"PBG1VN",
}

def get_patient_datapath(folder, patient): 
    return join(folder, patient, patient + '.mat')

def safe_mkdir(path):
  Path(path).mkdir(parents=True, exist_ok=True)

def download_patient_data_if_not_local(chosen_patient, download_folder = HARVARD_FOLDER): 

    datapath = get_patient_datapath(download_folder, chosen_patient)

    if not Path(datapath).is_file():
        safe_mkdir(join(download_folder, chosen_patient))
        HARVARD_URL = f'https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/CI3WB6/{HARVARD_DB_IDs[chosen_patient]}'
        print(f'Descargando dataset paciente {chosen_patient}...')
        system_call(f'wget {HARVARD_URL} -O {datapath}')

def load_patient_data(chosen_patient, folder = HARVARD_FOLDER): 
    datapath = get_patient_datapath(folder, chosen_patient)
    download_patient_data_if_not_local(chosen_patient, folder)
    patient_data = np.array(scipy.io.loadmat(datapath)['data']) # (1, 25, 832, 196, 16, 2), (slice, frame, ro, spokes/frame, coils, real-imag)
    return patient_data 

#----------------------------------------------------------------------#
# Preprocessing functions 
#----------------------------------------------------------------------#
TDIM = 1 
RODIM = 2 
SPKDIM = 3 
CDIM = 4 

def count_left_padding_per_spoke(data, read_out_axis = RODIM):
  return (data!=0).argmax(axis=read_out_axis) # source: https://stackoverflow.com/a/47269413

def count_right_padding_per_spoke(data, read_out_axis = RODIM):
  return np.flip(data!=0, axis=read_out_axis).argmax(axis=read_out_axis) # source: https://stackoverflow.com/a/47269413
 
def are_all_equal(array, value): 
  return np.all(array == value)

def remove_index_from_ntuple(ntuple, idx): 
  return ntuple[:idx] + ntuple[idx+1:]

def are_shapes_compatible(data_shape, padding_shape, axis): 
  """
    Verifies consistency for relaxed_paddign_removal. padding_shape is the
    remaining tuple from removing axis-th entry from data_shape.
  """
  return remove_index_from_ntuple(data_shape, axis) == padding_shape

# assert are_shapes_compatible(complexdata.shape, count_left_padding_per_spoke(complexdata, 2).shape, 2) 
# assert are_shapes_compatible(complexdata.shape, count_right_padding_per_spoke(complexdata, 2).shape, 2) 

def padding_removal(data, read_out_axis, mode = 'relaxed'): 
  """
  - `mode` = `'relaxed'` or `'aggresive'`
  """
  left_padding_vector = count_left_padding_per_spoke(data, read_out_axis)
  right_padding_vector = count_right_padding_per_spoke(data, read_out_axis)
  index_function = {'relaxed': np.min, 'aggresive': np.max}
  left_pad = index_function[mode](left_padding_vector)
  right_pad = index_function[mode](right_padding_vector)
  pad = index_function[mode](np.array([left_pad, right_pad]))
  left_idx = pad
  right_idx = data.shape[read_out_axis] - pad 
  return data.take(indices=np.array(range(left_idx, right_idx)), axis=read_out_axis)

def is_even(x):
  return x % 2 == 0


def make_even_dim_at_axis(data, axis): 
  """
  # example 
  data.dim = (2,4,5), dim = 2 => make_even_dim(data, dim).shape = (2,4,4)
  """
  if is_even(data.shape[axis]): 
    return data 
  
  return np.take(data, np.array(range(0,data.shape[axis]-1)), axis = axis)

def make_symetric_kspace(data, read_out_axis): 
  """
  In an array of frequencies of shape (N,), associated freqs are:
  - [-N//2, -N//2+1, ..., -1, 0, 1, ..., N//2-1] if N is even
  - [-(N-1)//2, -(N-1)//2+1, ..., -1, 0, 1, ..., (N-1)//2] if N is odd
  Add a padding of 0 of lenght 1 at the end of `read_out_axis` axis, if 
  `read_out_axis`-th element of `data.shape` is even. Make the `read_out_axis`
  axis symetrical and thus invariant to flippling.
  """
  if not is_even(data.shape[read_out_axis]): 
    return data 

  # Implementation: https://stackoverflow.com/a/49766444
  npad = [(0, 0)] * data.ndim
  npad[read_out_axis] = (0, 1)

  return np.pad(data, pad_width=npad, mode='constant', constant_values=0)

def replace_left_side_data_1d_array(basearray, filldata, n): 
  """
  Replace the first `n` elements from `basearray` with the first `n` elements of
  `filldata`.
  - `basearray`: array, ndim = 1
  - `filldata`: array, ndim = 1
  - `n`: int > 0 
  """
  assert basearray.ndim == 1
  assert filldata.ndim == 1
  assert basearray.shape == filldata.shape 
  return basearray.at[:n].set(filldata[:n])

def replace_left_side_data(basearray, filldata, N, axis):
  assert N.shape == () #remove_index_from_ntuple(basearray.shape, axis)
  assert basearray.shape == filldata.shape 
  base_axis_at_last_dim = np.moveaxis(basearray, axis, -1)
  fill_axis_at_last_dim = np.moveaxis(filldata, axis, -1)
  replace_func = np.vectorize(replace_left_side_data_1d_array, signature='(n),(n),()->(n)')
  replaced_axis_at_last = replace_func(base_axis_at_last_dim, fill_axis_at_last_dim, N)
  replaced = np.moveaxis(replaced_axis_at_last, -1, axis)
  return replaced 

def extract_first_element(data): 
  idx = (0,) * data.ndim
  return data[idx]

def hermitic_filling_left_padding(data, read_out_axis): 
  # https://www.mr-tip.com/serv1.php?type=db1&dbs=Hermitian+Symmetry
  left_padding_vector = count_left_padding_per_spoke(data, read_out_axis)
  sym_data = make_symetric_kspace(data, read_out_axis)
  hermitian_fill_data = np.conjugate(np.flip(sym_data, axis=read_out_axis))
  replaced_elements_per_spoke = extract_first_element(left_padding_vector)
  if are_all_equal(left_padding_vector, replaced_elements_per_spoke):    
    return replace_left_side_data(sym_data, hermitian_fill_data, replaced_elements_per_spoke, read_out_axis)
  else: 
    print("Can't make hermitic filling, different left padding")
  # return sym_data.at[idx].set(hermitian_fill_data[idx])

def is_partial_echo(data, read_out_axis = RODIM): 
  left_padding = count_left_padding_per_spoke(data, read_out_axis)
  right_padding = count_right_padding_per_spoke(data, read_out_axis)

  left_idx = extract_first_element(left_padding) 
  right_idx = extract_first_element(right_padding)
  same_left_idx = are_all_equal(left_padding, left_idx)
  same_right_idx = are_all_equal(right_padding, right_idx)
  return same_left_idx and same_right_idx and left_idx != right_idx

#================================================================#
# complexdata preprocessing 
#================================================================#

def generate_uniform_radial_angles(nframes, spokes_per_frame, maxangle = np.pi):
  angles = np.array([np.linspace(0, maxangle, spokes_per_frame, endpoint = False) for _ in range(nframes)])
  return angles # nframes, spoke_per_frame

def harvard_format_to_standard_radial_acquisitions(complexdata): 
  """
  # Arguments 
  - complexdata: array(complex), shape (1, frame, ro, spokes, coil)
  # Output
  - array(complex), shape (coils, frame, spokes, ro)
  """
  complexdata = complexdata[0] # (frame, ro, spokes, coils)
  complexdata = np.moveaxis(complexdata, -1, 0) #  (coils, frame, ro, spokes)
  complexdata = np.moveaxis(complexdata, -1, -2) #  (coils, frame, spokes, ro)
  return complexdata 

# assert harvard_format_to_standard_radial_acquisitions(np.ones((1,3,5,2,4))).shape == (4,2,) 
#---------------------------------------------------------------#
# Subsampling data 
#---------------------------------------------------------------#

def angle_distance(angle1, angle2): 
  """
    Angular distance
  `angle1` y `angle2` measured in radians. Return angle between them (in [0,pi]).
  """
  minval = np.minimum(angle1, angle2)
  phi1, phi2 = angle1 - minval, angle2- minval
  # phimin = 0
  phimax = np.maximum(phi1, phi2)
  phimax = np.mod(phimax, 2 * np.pi)
  return lax.cond(phimax > np.pi, lambda phi: 2 * np.pi - phi, lambda phi: phi, phimax)
 
def find_idx_closer_angle_from_list(angle, comparison_angles_list): 
  """
  Finds index in such a way that i-th angle has the least angular distance to
  `angle` of all angles from `comparison_angles_list`.
  """
  distances = vmap(angle_distance, in_axes=(None, 0))(angle, comparison_angles_list)
  return np.argmin(distances)

def closest_subsampling_indexs(ideal_angles, acquired_angles): 
  """
  For each frame and each desired angle, find the closest of all available angles 
  for that frame.
  - `ideal_angles`: array, shape (frames, desired_sub_angles_per_frame). Desired angles.
  - `acquired_angles`: array, shape (frames, total_spokes_per_frame). Available angles.
  """
  find_closest_idx_in_frame_angles = vmap(find_idx_closer_angle_from_list, in_axes = (0, None)) # out: (desired_sub_angles_per_frame,)
  return vmap(find_closest_idx_in_frame_angles)(ideal_angles, acquired_angles) # out: (frame, desired_sub_angles_per_frame)

def closest_subsampling_idxs_with_inversion(ideal_angles, acquired_angles): 
  """
  Look for the closest angles to `ideal_angles` from all the availables in
  `acquired_angles`. `acquired_angles` has values between [0,pi]. "inverted
  versions" of the angles, i.e. angle + pi, are also valid. So, if `alpha` is in 
  the array of acquired angles, `alpha + pi` also is considered to be in the
  acquired angles array, and is a valid candidate for closest angle from desired.
  Return index from original angle `alpha`. 

  # Example 
  If the acquired angles array contains `alpha=0` at index `i=13`, and
  desired angle is `pi`, then the method returns `i=13` because `alpha + pi=pi` is 
  the closest to the desired angle.

  Shape of arrays is the same than `closest_subsampling_indexs` function. Does 
  not remove repeated indexs.
  """
  assert np.all(0 <= acquired_angles) and np.all(acquired_angles < np.pi)
  full_acquired_angles = np.concatenate((acquired_angles, acquired_angles + np.pi), axis = 1)
  raw_idxs = closest_subsampling_indexs(ideal_angles, full_acquired_angles)
  max_valid_idx = acquired_angles.shape[1]
  corrected_idxs = raw_idxs.at[raw_idxs >= max_valid_idx].set(raw_idxs[raw_idxs >= max_valid_idx] - max_valid_idx)
  return corrected_idxs

def pseudo_golden_subsampling_indexs(acquired_angles, subsampled_spokes_per_frame, tiny_number): 
  """
  Select indexs of angles in `acquired_angles` that are closest of a tiny-golden
  angle acquisition. "Inversions" (see `closest_subsampling_idxs_with_inversion`)
  are valid. `acquired_angles` must have values in [0, pi).
  """
  frames = acquired_angles.shape[0]
  desired_angles = tiny_golden_angles_in_time(frames, subsampled_spokes_per_frame, tiny_number)
  return closest_subsampling_idxs_with_inversion(desired_angles, acquired_angles)

def rotating_subsampling_indexs(nframes, spokes_per_frame, angle_jump): 
  if spokes_per_frame % angle_jump != 0: 
    print(f"WARNING: angle_jump = {angle_jump} is not a divisor of spokes_per_frame = {spokes_per_frame}.")
  idxs = []  
  for i in range(nframes): 
    idxs.append(np.array(range(i%angle_jump, spokes_per_frame, angle_jump)))
  idxs = np.array(idxs)
  return idxs 

def subsampling_framespoke_array_with_per_frame_pattern(data, idxs): 
  return np.array([data[i, idxs[i]] for i in range(data.shape[0])])

def subsampling_standard_array_with_per_frame_pattern(data, idxs): 
  assert TDIM == 1 and SPKDIM == 3 
  assert data.ndim == 5 
  assert idxs.ndim == 2 
  assert data.shape[TDIM] == idxs.shape[0]
  
  subsampled = np.stack([data[:,i][:,:,idxs[i]] for i in range(data.shape[1])], axis = 1)
  return subsampled



#================================================================#
# Full pipeline 
#================================================================#

def acquisition_from_harvard_data(chosen_patient = 'P01', pad_removal_mode = 'aggresive', sub_spokes_per_frame = 14, hermitic_fill = None, tiny_number = 1): 
  """
  # Arguments 
  - 'chosen_patient': 'P01', 'P02', 'P03', 'P04'
  - pad_removal_mode: 'aggresive', 'relaxed'
  - sub_spokes_per_frame: number of chosen spokes, according to `tiny_number`th-golden-angle.
  
  # Output 
  - full_acquisition: RadialAcquisition object with the full data, after removing 
    padding in the read-out dimention.
  - undersampled_acquisition: RadialAcquisition, removing data according to `sub_spokes_per_frame`.
  """

  patient_data = load_patient_data(chosen_patient)
  complexdata = patient_data[...,0] + 1j* patient_data[...,1] 
  print(f"Complex data is{'' if is_partial_echo(complexdata) else ' not'} a partial echo")
  if hermitic_fill is not None:
    print("Hermitic filling of partial echo...")
    complexdata = hermitic_filling_left_padding(complexdata, RODIM)
  print(f"Appliying {pad_removal_mode} padding removal")
  complexdata = padding_removal(complexdata, RODIM, pad_removal_mode) 
  complexdata = make_even_dim_at_axis(complexdata, RODIM)
  print(f"Processed complex data is{'' if is_partial_echo(complexdata) else ' not'} a partial echo. Shape {complexdata.shape}")
  
  # show_im(np.abs(complexdata)[0,10,:,:,0])
  # plot_array(np.abs(complexdata[0,10,:,0,0]))
  spokes_per_frame = complexdata.shape[SPKDIM]
  nframes          = complexdata.shape[TDIM]

  full_time_angles = generate_uniform_radial_angles(nframes, spokes_per_frame)
  full_acquisitions = harvard_format_to_standard_radial_acquisitions(complexdata)
  # sub_idxs = rotating_subsampling_indexs(nframes, spokes_per_frame, angle_jump)
  sub_idxs = pseudo_golden_subsampling_indexs(full_time_angles, sub_spokes_per_frame, tiny_number)

  time_angles = subsampling_framespoke_array_with_per_frame_pattern(full_time_angles, sub_idxs)
  subsampledcomplexdata = subsampling_standard_array_with_per_frame_pattern(complexdata, sub_idxs)
  undersamples_acquisitions = harvard_format_to_standard_radial_acquisitions(subsampledcomplexdata)

  rad_full_acq = radial_acquisition_from_angles(full_time_angles, full_acquisitions)
  rad_undersampled_acq = radial_acquisition_from_angles(time_angles, undersamples_acquisitions)

  return rad_full_acq, rad_undersampled_acq

#----------------------------------------------------------------------------#
# Interface with Neural Nets 
# necesito una función de la forma 
# train_X, train_Y, csmaps, im = load_data(config)
#----------------------------------------------------------------------------#

def cfl_exists(path): 
  return Path(path + '.cfl').is_file()

def make_grasp_name(grasp_param): 
    return f"lmbda{grasp_param['lmbda']}_lagrangian{grasp_param['lagrangian']}_iters{grasp_param['iters']}".replace('.', '-')


def get_db_config(config): 
  db_config = {
    'chosen_patient': config['chosen_patient'], 
    'pad_removal_mode': 'relaxed' if config['relaxed_pad_removal'] else 'aggresive', 
    'hermitic_fill': config['hermitic_fill'], 
    'sub_spokes_per_frame': config['sub_spokes_per_frame'],     
    'tiny_number': config['tiny_number']
  }
  return db_config 

def get_bart_dir(db_config):
  spin_echo_extradir = "" if db_config['hermitic_fill'] is None else "/hermit-fill"
  bart_dir = join(BART_DIRECTORY, f"{db_config['chosen_patient']}{spin_echo_extradir}")
  return bart_dir 

def get_bart_grasp_subacq_name(db_config):
  return f"sub-acq-tiny{db_config['tiny_number']}-spokes{db_config['sub_spokes_per_frame']}"

def get_csmaps_and_mask(bart_acquisition):
    coil_sens = bart_acquisition.calculate_coil_sens()
    csmaps = coil_sens.to_std_coil()
    hollow_mask = coil_sens.hollow_mask()
    return csmaps, hollow_mask

REFERENCE_GRASP_PARAMS = {'lmbda': 0.001, 'lagrangian': 0.0005, 'iters': 100 }

def get_reference_reco(bart_acquisition):
    gtpath = bart_acquisition.path + '/fullacq-gt-reco_' + make_grasp_name(REFERENCE_GRASP_PARAMS)
    if cfl_exists(gtpath): 
        reco_GT = bart_read(gtpath)
    else:
        reco_GT = bart_acquisition.calculate_bart_reco('gt-reco_' + make_grasp_name(REFERENCE_GRASP_PARAMS), **REFERENCE_GRASP_PARAMS)

    im = np.squeeze(reco_GT)
    im = normalize(im)
    return im 


def load_data(config): 
    db_config = get_db_config(config)  

    ra, rus = acquisition_from_harvard_data(**db_config)

    bart_dir = get_bart_dir(db_config)
    safe_mkdir(bart_dir)

    bart_acquisitions = ra.to_bart(bart_dir, "fullacq")  

    synth_csmaps, hollow_mask = get_csmaps_and_mask(bart_acquisitions)

    im = get_reference_reco(bart_acquisitions)

    train_X, train_Y = rus.generate_dataset()
    return train_X, train_Y, synth_csmaps, im, hollow_mask 

def load_data_without_bart(chosen_patient:str, sub_spokes_per_frame:int, csmap_folder:str): 
    """
    - chosen_patient: str. Cualquier key del diccionario HARVARD_DB_IDs.
    - sub_spokes_per_frame: usualmente 8, 12, 16
    """
    config_data = {
        'chosen_patient'        : chosen_patient, 
        'hermitic_fill'         : True, 
        'relaxed_pad_removal'   : False,
        'sub_spokes_per_frame'  : sub_spokes_per_frame, 
        'tiny_number'           : 1,
    }

    db_config = get_db_config(config_data)  

    ra, rus = acquisition_from_harvard_data(**db_config) # ra es la adquisición full, rus es la adquisicion submuestreada 
    train_X, train_Y = rus.generate_dataset()

    csmappath = join(csmap_folder, f'csmap-{chosen_patient}.npy')
    csmap = np.load(csmappath)
    hollow_mask = calculate_hollow_mask(csmap)

    im = None # TODO: real reference image 

    return train_X, train_Y, csmap, im, hollow_mask 

#----------------------------------------------------------------------#
# Reconstrucciones con GRASP 
#----------------------------------------------------------------------#

GRASP_BEST_PARAMS = {
    16: (0.001, 5.0),
    8 : (0.001, 5.0), # antes usaba (0.01, 5.0)
    4 : (0.01, 5.0), 
    2 : (0.1, 0.5), 
    1 : (10., 0.5), 
}

def get_grasp_subreco_name(grasp_params):
    grasp_reco_name =  'gt-subgrasp-reco_' + make_grasp_name(grasp_params)
    return grasp_reco_name

def load_grasp_undersampled_reco(config, grasp_sub_grasp_params):
    """
    - `grasp_sub_grasp_params`: dict de la forma 
        `{'lmbda': float, 'lagrangian': float, 'iters': int }`

    ## Notas 

    Ver `GRASP_BEST_PARAMS` para parámetros que dan resultados razonables
    según el número de spokes disponibles (`sub_spokes_per_frame`).
    """
    db_config = get_db_config(config)
    ra, rus = acquisition_from_harvard_data(**db_config)

    bart_dir = get_bart_dir(db_config)
    safe_mkdir(bart_dir)

    bart_acquisitions = ra.to_bart(bart_dir, "fullacq")  

    bart_grasp_name = get_bart_grasp_subacq_name(db_config)
    bart_sub_acquisitions = rus.to_bart(bart_dir, bart_grasp_name) 
    grasp_reco_name = get_grasp_subreco_name(grasp_sub_grasp_params)
    subreco = bart_sub_acquisitions.calculate_bart_reco_with_external_csmap(bart_acquisitions, grasp_reco_name, **grasp_sub_grasp_params)
    return subreco 

def load_undersampled_acquisition_from_bart_files(config):
    """
    Es una forma más rápida de cargar una adquisición. Debería
    arrojar error si no se ha usado previamente
    `load_grasp_undersampled_reco` con el mismo `config`. El uso 
    previo de esa función deja archivos de BART residuales que 
    pueden volver a usarse sin tener que cargar todos los datos y
    luego hacer el preprocesamiento.    
    """
    db_config = get_db_config(config)
    bart_dir = get_bart_dir(db_config)

    spokes = db_config['sub_spokes_per_frame']
    tiny_number = db_config['tiny_number']

    bart_sub_acquisitions = bart_acquisition_from_files(bart_dir, f'sub-acq-tiny{tiny_number}-spokes{spokes}')
    return bart_sub_acquisitions

def load_grasp_subreco_with_previous_bart_data(bart_sub_acquisitions, grasp_sub_grasp_params):
    """
    Permite hacer una reconstrucción con GRASP a partir de una
    `bart_sub_acquisitions` existente. Útil cuando se necesitan 
    hacer varias reconstrucciones a partir de la misma adquisición
    (usando distintos hiperparámetros de GRASP por ejemplo).

    - `grasp_sub_grasp_params`: ver `load_grasp_undersampled_reco`
    """
    bart_dir = bart_sub_acquisitions.path
    subreco = bart_sub_acquisitions.calculate_bart_reco_with_external_csmappath(bart_dir + 'fullacq-csmap', 'gt-subgrasp-reco_' + make_grasp_name(grasp_sub_grasp_params), **grasp_sub_grasp_params)
    return subreco 

#----------------------------------------------------------------------#
# Crops del corazón para calcular métricas
#----------------------------------------------------------------------#

def calculate_width(left, right, full_width): 
    return full_width - (left + right)

def calculate_new_lims(old_l, old_r, full_width, new_width): 
    old_width = calculate_width(old_l, old_r, full_width)
    center = old_l + 1 + old_width // 2 
    new_l = center - new_width // 2 
    new_r = full_width - new_l - new_width 
    return new_l, new_r 

def new_lims_dict(adict, full_width, new_size): 
    new_dict = {}
    for patientname, lims in adict.items():
        old_u, old_d, old_l, old_r = lims 
        new_width_v, new_width_h = new_size
        new_u, new_d = calculate_new_lims(old_u, old_d, full_width, new_width_v)
        new_l, new_r = calculate_new_lims(old_l, old_r, full_width, new_width_h)
        new_dict[patientname] = [new_u, new_d, new_l, new_r]
    return new_dict 

old_crops = {
    'P01': [60,80,50,75],
    'P02': [70,75,60,70], 
    'P03': [78,70,58,70],
    'P04': [70,80,80,70], 
    'P05': [65,75,60,60],
    'P07': [75,65,70,60],
    'P08': [40,105,55,80], 
    # 'P09': [70,75,60,70], 
    # 'P10': [60,80,50,75],
    'P11': [75,75,55,75],
    'P12': [80,70,55,75],
    'P13': [80,65,45,80],
    'P15': [75,70,60,75],
}

PATIENTS_HEART_LOCATION = new_lims_dict(old_crops, 208, (65, 78))

correctFlip = {
    'P01': False,
    'P02': False,
    'P03': True,
    'P04': False,
    'P05': False,
    'P06': False,
    'P07': False,
    'P08': False,
    'P09': False,
    'P11': False,
    'P12': False,
    'P13': False,
    'P15': False,
    'P18': True,
    'P20': False,
    'P21': False,
    'P22': False,
    'P23': False,
    'P24': True,
    'P25': False
}