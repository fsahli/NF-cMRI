"""
Radial acquisitions, guardadas como una trayectoria y un conjunto de datos.

Recibe una lista de angulos por tiempo, que transforma en una trayectoria. 

Es la principal interfaz con los datos que reciben la redes tipo INR,
mediante la funcion `generate_dataset` que entrega los datos de training X e Y. 
"""
from jax import numpy as np 
from jax import vmap 
from nfcmri.models.classical.fourier import get_points_in_angle
from nfcmri.models.classical.bart import bart_acquisition_from_arrays, BARTAcquisition
from nfcmri.models.classical.radon import calculate_angle 

def radial_trajs(angles, nsamples):
    """
    # Argumentos 
    - `frame_angles`: array(float), shape (nframes, spokes_per_frame)

    # Salida: 
    - array(float), shape (nframes, spokes_per_frame, nsamples, 2) (2 por real e imaginario)
    
    """
    return vmap(vmap(lambda angle: get_points_in_angle(angle, nsamples)))(angles)

def add_times_to_framestrajxy(framestrajxy): 
  """
  # Argumentos: 
  - `framestrajxy`: array(float), shape (nframes, spokes_per_frame, nsamples, 2)
  # Salida: 
  - array(float), shape (nframe * spokes_per_frame, nsamples + 1, 2)
  reduce la dimension pero agrega el tiempo asociado al frame como primer sample.
  """
  nframes, nspokes_per_frame, nsamples, dims = framestrajxy.shape 
  times = (np.linspace(0,1,nframes,endpoint = False)[:,None] * np.ones(nspokes_per_frame)[None,:]).ravel()[:,None]
  times = np.stack([times, times], axis = -1)
  return np.r_['1',times, np.reshape(framestrajxy, (nframes * nspokes_per_frame, nsamples, dims))]

def is_spoke_traj_centered(spoke) -> bool: 
  """
  Devuelve True si el punto central de la spoke es [0.,0.].
  El centro se calcula con la convención de la fft (readout_points // 2)

  - spoke: (readout_points,2)
  """
  assert spoke.ndim == 2
  assert spoke.shape[-1] == 2
  fft_center_index = spoke.shape[0]//2
  center = spoke[fft_center_index]
  is_center_close_to_0 = np.all(np.isclose(center, np.array([0.,0.]))) # bool 
  return is_center_close_to_0

def are_spoke_trajs_centered(spokes) -> bool:
    """
    - spokes: array (nspoke, readout_points, 2)
    """
    are_spokes_centered = vmap(is_spoke_traj_centered)(spokes) # array(bool)
    return np.all(are_spokes_centered)

def kFOV_limit_from_spoke_traj(spoke):
  """
  spoke: (readout_points,2)
  """
  assert spoke.ndim == 2
  assert spoke.shape[-1] == 2
  readout_points = spoke.shape[0]
  extreme_point = spoke[0]
  return np.linalg.norm(extreme_point) / readout_points

def radial_acquisition_from_angles(frame_angles, data):
    """
    # Argumentos 
    - `frame_angles`: array(float), shape (frames, spokes_per_frame)
    - `data`: array(complex), shape (ncoils, frames, spokes_per_frame, nsamples)
    """
    assert frame_angles.ndim == 2
    assert data.ndim == 4
    assert frame_angles.shape == data.shape[1:3]
    nsamples = data.shape[-1]
    trajs = radial_trajs(frame_angles, nsamples)
    return RadialAcquisitions(trajs, data)

class RadialAcquisitions: 
    """
    Guarda trayectorias y adquisiciones radiales. 
    Principal interfaz con el entrenamiento de la red y con BART. 
    Las fuentes de datos generan uno de estos.
    # Ejemplos 
    ## Guardar datos 
    Crear una instancia con una adquisicion y una lista de angulos 
    ```
    >>> radialdata = RadialAcquisitions(trajs, data) 
    >>> print(radialdata)
    Radial adquisition: 9 coils, 20 frames of 8 spokes each, 10 samples per spokes.
    ```
    ## Generar dataset para entrenar la red 
    ```
    >>> dataX, dataY = radialdata.generate_dataset_angletime()
    ```
    """

    def __init__(self, trajs, data): 
        """
        # Argumentos 
        - `trajs`: array(float), shape (frames, spokes_per_frame, nsamples, 2) (2 por real e imaginario)
            - `nsamples` es la dimensión de readout.
            - trajs.max() debería estar más o menos en `nsamples//2`
        - `data`: array(complex), shape (ncoils, frames, spokes_per_frame, nsamples)
        """
        assert trajs.ndim == 4
        assert trajs.shape[-1] == 2
        assert data.ndim == 4
        assert trajs.shape[:3] == data.shape[1:4]
        self.trajs = trajs 
        self.data = data 
    
    def get_number_of_coils(self): 
        return self.data.shape[0]

    def get_total_frames(self):
        return self.data.shape[1]
    
    def get_spokes_per_frame(self): 
        return self.data.shape[2]
    
    def get_read_out_dim(self): 
        return self.data.shape[3]

    def generate_dataset(self): 
        """
            DEPRECATED: Old interface to training neural networks with data from Radial Trajectories 
        """
        print("Warning: This function was deprecated!!!")
        fourier_radial_data = self.data.reshape(self.get_number_of_coils(), self.get_total_frames() * self.get_spokes_per_frame(), self.get_read_out_dim())
        fourier_radial_data = np.moveaxis(fourier_radial_data, 0,1)
        dataY = fourier_radial_data[...,None]
        dataX = add_times_to_framestrajxy(self.trajs)
        return dataX, dataY 
        
    def generate_dataset_angletime(self):
        """
            Interface to training neural networks with data from Radial Trajectories 
        - dataX: array float shape (nframes * spokes_per_frame, 2)
            dataX[i] contiene angle, bin index of the i-th spoke 
        - dataY: array complex shape (nframes * spokes_per_frame, ncoils, read out, 1)
            dataY[i] is the data acquired in i-th spoke 
        """
        angles = vmap(vmap(calculate_angle, in_axes=0), in_axes=0)(self.trajs) # (frames, spokesperframe)
        nframes, nspokesperframe = angles.shape
        ts = np.arange(nframes) 
        ts = np.broadcast_to(ts[:,None], (nframes, nspokesperframe))
        dataX = np.stack((angles.flatten(), ts.flatten()), axis=-1)

        fourier_radial_data = self.data.reshape(self.get_number_of_coils(), self.get_total_frames() * self.get_spokes_per_frame(), self.get_read_out_dim())
        fourier_radial_data = np.moveaxis(fourier_radial_data, 0,1)
        dataY = fourier_radial_data[...,None]
        return dataX, dataY

    def get_spclim(self):
        return kFOV_limit_from_spoke_traj(self.trajs[0,0])

    def __str__(self): 
        return f"Radial adquisition: {self.get_number_of_coils()} coils, {self.get_total_frames()} frames of {self.get_spokes_per_frame()} spokes each, {self.get_read_out_dim()} samples per spokes."
    
    def to_bart(self, datapath, name): 
        """
        Returns BARTAcquisition object 
        """
        return bart_acquisition_from_arrays(np.copy(self.trajs), np.copy(self.data), datapath, name) 

    def unbin_data(self):
        new_data_shape = (self.data.shape[0], -1, 1, self.data.shape[-1]) # preserve dims 0 and -1, put everthing in RO dim 
        new_traj_shape = (-1, 1, self.trajs.shape[-2], self.trajs.shape[-1])
        self.data = self.data.reshape(new_data_shape)
        self.trajs = self.trajs.reshape(new_traj_shape)

    def bin_data(self, new_spokes_per_frame):
        self.unbin_data()
        new_frames = self.data.shape[1] // new_spokes_per_frame
        new_total_spokes = new_frames * new_spokes_per_frame
        new_data_shape = (self.data.shape[0], new_frames, new_spokes_per_frame, self.data.shape[-1]) 
        self.data = self.data[:,:new_total_spokes, ...].reshape(new_data_shape)
        new_trajs_shape = (new_frames, new_spokes_per_frame, self.trajs.shape[-2], self.trajs.shape[-1]) 
        self.trajs  = self.trajs[:new_total_spokes, ...].reshape(new_trajs_shape)

    def selected_range(self, idt0:int, idtf:int):
        """
            Conserva solo los datos entre los frames [idt0, idtf)
        - idt0: índice inicial 
        - idtf: índice final 
        """
        assert idt0 >= 0
        assert idtf > idt0
        assert idtf <= self.trajs.shape[0]

        self.trajs = self.trajs[idt0:idtf]
        self.data = self.data[:,idt0:idtf]

def create_radial_acq_from_bart(bartacq:BARTAcquisition): 
    return radial_acquisition_from_angles(bartacq.calculate_frameangles(), bartacq.calculate_radial_acq_data())

def check_centered_train_X(train_X):
    """
    - return True si el train_X está correctamente centrado (la frecuencia asociada a 0 está en el lugar esperado)
    """
    spokes = train_X[:,1:,:]
    centered_spokes = are_spoke_trajs_centered(spokes)
    return centered_spokes 

def check_dataset_dims(train_X, train_Y):
    assert train_X.ndim == 3, f"train_X.ndim is {train_X.ndim}, should be 3: (batch, read out + 1, 2)"
    assert train_Y.ndim == 4, f"train_Y.ndim is {train_Y.ndim}, should be 4: (batch, ncoil, read out, 1)"
    assert train_Y.shape[-1] == 1, f"train_Y.shape[-1] is {train_Y.shape[-1]}, should be 1"
    assert train_X.shape[0] == train_Y.shape[0], f"train_X and train_Y have different batch size {train_X.shape[0]} and {train_Y.shape[0]}"
    assert train_X.shape[1] == train_Y.shape[2] + 1, f"train_X.shape[1] {train_X.shape[1]} and train_Y.shape[2]+1 ({train_Y.shape[2]+1}) should be the same"
    assert train_X.shape[-1] == 2, f"train_X.shape[-1] {train_X.shape[-1]} should be 2"

def check_correct_dataset(train_X):
    """
    train_X generado por RadialAcquisition.generate_dataset()
    """
    centered_spokes = check_centered_train_X(train_X)
    str_centered_spokes = 'centered' if centered_spokes else 'not centered!'
    print(f"The trajectories are {str_centered_spokes}")
    return centered_spokes 