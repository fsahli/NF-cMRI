# NF-cMRI: Unsupervised reconstruction of accelerated cardiac cine MRI using neural fields <br> <span style="float: right"><sub><sup>Computers in Biology and Medicine</sup></sub></span> 

---

<!--[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](https://opensource.org/licenses/MIT)-->
[![Paper-Journal](https://img.shields.io/badge/Journal-Computers_in_Biology_and_Medicine-e43444.svg?style=flat-square)](https://www.sciencedirect.com/science/article/abs/pii/S001048252401552X)
[![Preprint](https://img.shields.io/badge/Preprint-arXiv-b41c1c.svg?style=flat-square)](https://arxiv.org/abs/2307.14363)
<!-- ihealth colors: azul oscuro #232453 green  #08c359
[![Institution](https://img.shields.io/badge/Institute-iHEALTH-08c359.svg?style=flat-square)](https://i-health.cl/)
--->
[![Institution](https://img.shields.io/badge/Institute-iHEALTH-232453.svg?style=flat-square)](https://i-health.cl/)

## Citation

```python
@article{catalanUnsupervisedReconstructionAccelerated2025,
	title = {Unsupervised reconstruction of accelerated cardiac cine {MRI} using neural fields},
	volume = {185},
	issn = {00104825},
	url = {https://linkinghub.elsevier.com/retrieve/pii/S001048252401552X},
	doi = {10.1016/j.compbiomed.2024.109467},
	language = {en},
	journal = {Computers in Biology and Medicine},
	author = {Catalán, Tabita and Courdurier, Matías and Osses, Axel and Fotaki, Anastasia and Botnar, René and Sahli-Costabal, Francisco and Prieto, Claudia},
	month = feb,
	year = {2025},
	pages = {109467},
}
```

## Introduction 

Cardiac cine MRI is the gold standard for cardiac functional assessment, but the inherently slow acquisition process creates the necessity of reconstruction approaches for accelerated undersampled acquisitions. Several regularization approaches that exploit spatial–temporal redundancy have been proposed to reconstruct undersampled cardiac cine MRI. More recently, methods based on supervised deep learning have been also proposed to further accelerate acquisition and reconstruction. However, these techniques rely on usually large dataset for training, which are not always available and might introduce biases.

In this work we propose NF-cMRI, an unsupervised approach based on implicit neural field representations for cardiac cine MRI. We evaluate our method in in-vivo undersampled golden-angle radial multi-coil acquisitions for undersampling factors of 13x, 17x and 26x.

![Intensity Net](assets/intensity-net.png)

![Training](assets/training.png)

## Instalation 

The easiest way of having everything running is installing the code as a pip package. The code uses [Jax](https://jax.readthedocs.io/en/latest/installation.html) and [Flax](https://flax.readthedocs.io/en/latest/), in the versions given by the `requirements.txt`.

### Installing as local pip package 

1. Clone the repo 
2. Create and activate a virtual environment 
3. Install requirements (Jax, Flax, etc)
4. Install `nfcmri` as editable libray 

```
$ git clone git@github.com:fsahli/NF-cMRI.git
$ cd NF-cMRI 
$ python -m venv venv
$ source venv/bin/activate 
(venv)$ pip install -r requirements.txt 
(venv)$ pip install -e . # install library in editable mode, allows import nfcmri 
```
