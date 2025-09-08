"""
    bart.py

This file contains an interface for connecting with BART. 
- read and write cfl files 
- transforming data to a format useful for bart 
"""
from jax import numpy as np 

from jax import vmap 
from nfcmri.models.classical.radon import calculate_angle

import os
import sys
from os.path import join 
from pathlib import Path 
from typing import Tuple 

# from abc import ABC, abstractmethod

os.environ["TOOLBOX_PATH"] = "/home/tabita/ACIP-MRI/bart"
path = os.environ["TOOLBOX_PATH"] + "/python"; # para poder importar los paquetes de python de bart 
sys.path.append(path);

import cfl
#------------------------------------------------------------# 
# Common funcions
#------------------------------------------------------------# 

#BART = '/opt/bart/bart-0.8.00/bart' <----------- esto sirve en ih-condor 
BART = 'bart' # <-------------- esto sirve si puedes llamar a bart desde la terminal 

def bart_write(array, path):
  cfl.writecfl(path, array)

def bart_read(path): 
  return cfl.readcfl(path)

def cfl_exists(path): 
  return Path(path + '.cfl').is_file()

#------------------------------------------------------------# 
# Array transformations
#------------------------------------------------------------#

# , float 
def bart_trajectory(trajectory): 
  """
  # Argumentos: 
  - trajectory: array(float), shape (frames, spokesperframe, readout, 2 o 3)
  """
  assert trajectory.ndim == 4 
  dim = trajectory.shape[-1]
  assert dim in [2,3]
  measured_points = trajectory + 0j # (frames, spokesperframe, readout, dim), complex 
  measured_points = np.moveaxis(measured_points, -1, 0) # (dim, frames, spokesperframe, readout), complex 
  measured_points = np.moveaxis(measured_points, -1, 1) # (dim, readout, frames, spokesperframe), complex 
  measured_points = np.moveaxis(measured_points, 2, -1) # (dim, readout, spokesperframe, frames), complex 
  if dim == 2: # añadir dimension faltante de 0s 
    measured_points = np.concatenate((measured_points, np.zeros(measured_points[0:1].shape)), axis = 0) # (3, readout, spokesperframe, frames)
  measured_points = measured_points[:,:,:,None,None,None,None,None,None,None,:,None,None,None,None,None]
  return measured_points

def bart_acquisition(fourier_radial_data): 
  """
  fourier_radial_data: array(complex), shape (coils, frames, spokesperframe, readout)
  """
  fourier_radial_data = np.moveaxis(fourier_radial_data, -1,0) # (readout, coils, frames, spokesperframe)
  fourier_radial_data = np.moveaxis(fourier_radial_data, -1,1) # (readout, spokesperframe, coils, frames)
  fourier_radial_data = fourier_radial_data[None, :,:,:,None,None,None,None,None,None,:,None,None,None,None,None]
  return fourier_radial_data

def bart_sensitivity(csmaps): 
  """
  csmaps: array(complex), shape (coils, px, py)
  """
  sensibilities = csmaps # (coils, px, py)
  sensibilities = np.moveaxis(sensibilities, 0, -1) # (px, py, coils)
  sensibilities = sensibilities[:,:,None,:,None,None,None,None,None,None,None,None,None,None,None,None]
  return sensibilities

def inverse_nufft_reconstruction(trajpath, datapath, imgpath, dims:Tuple[int,int,int]=None):
  str_dim = '' if dims is None else f' -d{dims[0]}:{dims[1]}:{dims[2]}'
  os.system(f'{BART} nufft -i{str_dim} -t {trajpath} {datapath} {imgpath}')
  im = bart_read(imgpath)
  return im 

def sense_reconstruction(trajpath, datapath, csmappath, imgpath):
  os.system(f"{BART} pics -t {trajpath} {datapath} {csmappath} {imgpath}")
  im = bart_read(imgpath)
  return im 

def image_fft(imgpath, kspacepath):
  os.system(f'{BART} fft -u $(bart bitmask 0 1) {imgpath} {kspacepath}')
  ks = bart_read(kspacepath)
  return ks 

def espirit_coil_sensitivity_estimation(kspacepath, csmappath): 
  os.system(f'{BART} ecalib -m1 {kspacepath} {csmappath}')
  cs = bart_read(csmappath)
  return cs 
  
# show_im(np.abs(cs[:,:,0,0]))
# show_im(np.abs(np.squeeze(multich_img))[:,:,5,10], cmap = 'gray')
# inufftreco_sub.shape:  (240, 240, 1, 16, 1, 1, 1, 1, 1, 1, 25)
# coil_sens_sub.shape:  (240, 240, 1, 16)

class BARTObject: 

  def __init__(self, datapath, name): 
    self.path = datapath 
    self.name = name 

  def aux_path(self, strid): 
    return join(self.path, self.name + strid)
  
  def obj_path(self): 
    pass 

  # debo tener paths a los archivos 

class BartMulticoilImage(BARTObject): 
  """Multicoil xyt image"""
  def __init__(self,im, datapath, name): 
    """
    im: array, shape (px, py, 1, coils, 1, 1, 1, 1, 1, 1, frames)
    """
    super().__init__(datapath, name)
    assert im.ndim == 11 , "im deberia tener 11 dims"
    self.im = im
  
  def obj_path(self): 
    return super().aux_path('-img')
  
  def simple_reco(self):
    im = self.im #  (px, py, 1, coils, 1, 1, 1, 1, 1, 1, frames)
    im = np.squeeze(im) # (px, py, coils, frames)
    assert im.ndim == 4 
    simple_reco = np.abs(np.sqrt(np.sum(np.power(im, 2), axis = -2)))
    return simple_reco # (px, py, frames)

def read_multicoil_img(path, name): 
  im = bart_read(path)
  return BartMulticoilImage(im, path, name)
# -------------------------------------------------------

def calculate_hollow_mask(csmap):
  """
  csmap: array, shape (ncoils, px, py)
  """ 
  mask = np.sum(csmap == 0, axis = 0)
  normalized_mask = mask / np.max(mask)
  return normalized_mask

class BartCoilSensitivityMaps(BARTObject): 
  def __init__(self,csmap, datapath, name): 
    """
    csmap: array, shape (px, py, 1, ncoils)
    """
    super().__init__(datapath, name)
    assert csmap.ndim == 4 , "im deberia tener 4 dims"
    self.csmap = csmap
  
  def obj_path(self): 
    return super().aux_path('-csmap')
  
  def to_std_coil(self):
    cs = self.csmap #(px, py, 1, ncoils)
    cs = np.squeeze(cs, axis = -2) #(px, py, ncoils)
    cs = np.moveaxis(cs, -1, 0)  #(ncoils, px, py)
    return cs 
  
  def hollow_mask(self): 
    return calculate_hollow_mask(self.to_std_coil())
  
  def write_csmap(self): 
    cs = self.csmap # (px,py,1,ncoil)
    cs = np.moveaxis(cs, -1, 0) # (ncoil, px, py, 1)
    cs = cs[...,0]
    bart_cs = bart_sensitivity(cs)
    bart_write(bart_cs, self.obj_path())

def read_csmap(path, name): 
  cs = bart_read(join(path, name + '-csmap'))
  return BartCoilSensitivityMaps(cs, path, name)

def make_grasp_name(grasp_param): 
    return f"lmbda{grasp_param['lmbda']}_lagrangian{grasp_param['lagrangian']}_iters{grasp_param['iters']}".replace('.', '-')

def grasp_reco(trajpath, datapath, csmappath, recopath, lmbda, lagrangian, iters): 
  os.system(f'{BART} pics -R T:$(bart bitmask 10):0:{lmbda} -u{lagrangian} -i{iters} -t {trajpath} {datapath} {csmappath} {recopath}')

def bart_acquisition_from_files(path, name):
  datapath = join(path, name + '-data')
  trajpath = join(path, name + '-traj')
  if not cfl_exists(datapath): 
    print(f"{datapath}.cfl no existe")
    return 
  if not cfl_exists(trajpath): 
    print(f"{trajpath}.cfl no existe")
    return 
  data = bart_read(datapath)
  data = np.reshape(data, data.shape + (1,)* (16 - data.ndim))
  traj = bart_read(trajpath)
  traj = np.reshape(traj, traj.shape + (1,)* (16 - traj.ndim))
  return BARTAcquisition(traj, data, path, name)


def bart_acquisition_from_arrays(traj, data, datapath, name): 
  """
  - traj: array(float), shape (frames, spokesperframe, readout, 2 o 3)
  - data: array(complex), shape (coils, frames, spokesperframe, readout)
  """
  return BARTAcquisition(bart_trajectory(traj), bart_acquisition(data), datapath, name) 


class BARTAcquisition: 

  def __init__(self, traj, data, datapath, name): 
    """
    - traj: array(float), shape (3, readout, spokesperframe, 1, 1, 1, 1, 1, 1, 1, frames, 1, 1, 1, 1, 1)
    - data: array(complex), shape (1, readout, spokesperframe, coils, 1,1,1,1,1,1,frames,1,1,1,1,1)
    """
    assert traj.ndim == 16 
    assert data.ndim == 16 
    assert traj.shape[0] == 3 
    self.traj = traj
    self.data = data
    self.path = datapath 
    self.name = name 
    bart_write(self.traj, self.trajpath())
    bart_write(self.data, self.datapath())

  def aux_path(self, strid): 
    return join(self.path, self.name + strid)

  def trajpath(self): 
    return self.aux_path('-traj')

  def datapath(self): 
    return self.aux_path('-data')
  
  def imgpath(self): 
    return self.aux_path('-img')

  def kspacepath(self): 
    return self.aux_path('-kspace')

  def csmappath(self): 
    return self.aux_path('-csmap')

  def calculate_inufft_reco(self, name:str='', dims:Tuple[int,int,int]=None):
    trajpath = self.trajpath()
    datapath = self.datapath()
    imgpath = self.imgpath() + name 
    im = inverse_nufft_reconstruction(trajpath, datapath, imgpath, dims=dims)
    return im #BartMulticoilImage(im, imgpath, self.name)
  
  def calculate_coil_sens(self): 
    imgpath = self.imgpath()
    kspacepath = self.kspacepath()
    csmappath = self.csmappath()
    if not cfl_exists(csmappath): 
      if not cfl_exists(kspacepath):
        if not cfl_exists(imgpath):
          self.calculate_inufft_reco()
        ks = image_fft(imgpath, kspacepath)
      cs = espirit_coil_sensitivity_estimation(kspacepath, csmappath)
    return read_csmap(self.path, self.name) # BartCoilSensitivityMaps(cs, csmappath, self.name) 

  def calculate_bart_reco(self, reconame, lmbda = 0.01, lagrangian = 5., iters = 50): 
    csmappath = self.csmappath()
    trajpath = self.trajpath()
    datapath = self.datapath()
    recopath = self.aux_path(f'-{reconame}')
    if not cfl_exists(recopath): 
      if not cfl_exists(csmappath): 
        self.calculate_coil_sens()
      grasp_reco(trajpath, datapath, csmappath,recopath,lmbda,lagrangian,iters)

    reco = bart_read(recopath)
    return reco

  
  def calculate_bart_reco_with_external_csmap(self, anotherBARTAcq, reconame, lmbda = 0.01, lagrangian = 5., iters = 50): 
    """
    Calcula una reconstrucci+on usando las mapas de sensibilidad de otra adquisición. 
    anotherBARTAcq: BARTAcquisition
      Se usan sus mapas de sensibilidad. 
    """
    csmappath = anotherBARTAcq.csmappath()
    trajpath = self.trajpath()
    datapath = self.datapath()
    recopath = self.aux_path(f'-{reconame}')
    if not cfl_exists(recopath): 
      if not cfl_exists(csmappath): 
        anotherBARTAcq.calculate_coil_sens()
      grasp_reco(trajpath, datapath, csmappath,recopath,lmbda,lagrangian,iters)
    reco = bart_read(recopath)
    return reco
  
  def calculate_bart_reco_with_external_csmappath(self, csmappath, reconame, lmbda = 0.01, lagrangian = 5., iters = 50, force_reco = False): 
    """
    Calcula una reconstrucci+on usando las mapas de sensibilidad de otra adquisición. 
    anotherBARTAcq: BARTAcquisition
      Se usan sus mapas de sensibilidad. 
    """
    if not cfl_exists(csmappath): 
      print(f"{csmappath} no es un archivo .cfl. Abortando.")
      return None 
    trajpath = self.trajpath()
    datapath = self.datapath()
    recopath = self.aux_path(f'-{reconame}')
    if not cfl_exists(recopath) or force_reco: 
      grasp_reco(trajpath, datapath, csmappath,recopath,lmbda,lagrangian,iters)
    reco = bart_read(recopath)
    return reco

  def calculate_frameangles(self): 
    """
      Devuelve los angulos de los datos almacenados 
    
    ## Salida 
    - `frameangles`: array(float), shape (frame, spokes_per_frame)
    """
    traj = self.traj 
    squeezed_traj = np.squeeze(traj, axis=(3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15))
    compacted_traj = np.stack([np.real(squeezed_traj[0]), np.real(squeezed_traj[1])], axis=-1)
    frameangles = vmap(vmap(calculate_angle, in_axes=1), in_axes=2)(compacted_traj) 
    return frameangles

  def calculate_radial_acq_data(self):
    """
      Devuelve los datos almacenados en un formato adecuado para RadialAcquisitions 
    
    ## Salida 
    - `squeezed_data`: array(complex), shape (ncoils, frames, spokes_per_frame, nsamples)
    """ 
    data = self.data 
    squeezed_data = np.squeeze(data, axis=(0, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15))

    squeezed_data # (nsamples, spokes_per_frame, ncoils, frames)
    squeezed_data = np.moveaxis(squeezed_data, 2, 0) # (ncoils, nsamples, spokes_per_frame, frames)
    squeezed_data = np.moveaxis(squeezed_data, 1, -1) # (ncoils, spokes_per_frame, frames, nsamples)
    squeezed_data = np.moveaxis(squeezed_data, 1, 2) # (ncoils, frames, spokes_per_frame, nsamples)
    return squeezed_data
  
  def static_new_shape(shape): 
    newshape = list(shape)
    newshape[2] = shape[2] * shape[10]
    newshape[10] = 1 
    return tuple(newshape)
  
  def make_static_acquisition(self, name): 
    static_traj = np.reshape(self.traj, BARTAcquisition.static_new_shape(self.traj.shape))
    static_traj = np.repeat(static_traj, self.traj.shape[10], axis = 10)
    static_data = np.reshape(self.data, BARTAcquisition.static_new_shape(self.data.shape))
    static_data = np.repeat(static_data, self.data.shape[10], axis = 10)
    return BARTAcquisition(static_traj, static_data, self.path, f'{self.name}-{name}')
  
  def make_sense_reconstruction(self, csmappath, reconame): 
    trajpath = self.trajpath()
    datapath = self.datapath()
    recopath = self.aux_path(f'-{reconame}')
    im = sense_reconstruction(trajpath, datapath, csmappath, recopath)
    return im 