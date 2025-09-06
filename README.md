# 📄 NF-cMRI: Unsupervised reconstruction of accelerated cardiac cine MRI using neural fields

by *Tabita Catalán*, *Matías Courdurier*, *Axel Osses*, *Anastasia Fotaki*, *René Botnar*, *Francisco Sahli-Costabal*, *Claudia Prieto*

<!--[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](https://opensource.org/licenses/MIT)-->
[![Paper-Journal](https://img.shields.io/badge/Journal-Computers_in_Biology_and_Medicine-e43444.svg?style=flat-square)](https://www.sciencedirect.com/science/article/abs/pii/S001048252401552X)
[![DOI](https://img.shields.io/badge/DOI-10.1016/j.compbiomed.2024.109467-08c359.svg?style=flat-square)](https://doi.org/10.1016/j.compbiomed.2024.109467)
[![Preprint](https://img.shields.io/badge/Preprint-arXiv-b41c1c.svg?style=flat-square)](https://arxiv.org/abs/2307.14363)
[![Institution](https://img.shields.io/badge/Institute-iHEALTH-232453.svg?style=flat-square)](https://i-health.cl/)
<!-- ihealth colors: azul oscuro #232453 green  #08c359
[![Institution](https://img.shields.io/badge/Institute-iHEALTH-08c359.svg?style=flat-square)](https://i-health.cl/)
--->

## 📎 Citation

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

## 🧠 Overview

This repository contains the code for our paper:

> **"Unsupervised reconstruction of accelerated cardiac cine MRI using neural fields"**, published at Computers in Biology and Medicine. 
> We propose an unsupervised approach for highly undersampled cardiac cine reconstruction based on implicit neural field representations for cardiac cine MRI. We evaluate our method in in-vivo retrospectively undersampled golden-angle radial multi-coil acquisitions for undersampling factors of 13x, 17x and 26x.

Key features:

- We propose STiFF, a spatio-temporal Fourier Features based encoding for periodic signals 
- We use a simple Radon-Transform approach for radial data that avoids NUFFT.

<!--
![Intensity Net](assets/intensity-net.png)

![Training](assets/training.png)


## 🖼️ Examples

Here are some example reconstructions:

| Ground Truth      | Zero-filled       | Our Method        |
| ----------------- | ----------------- | ----------------- |
| ![](path_to_img1) | ![](path_to_img2) | ![](path_to_img3) |

-->

## 🚀 Getting Started

### 0. Install the BART Toolbox 

Because some things use BART from the terminal and others the Python interface (`cfl` package), `nfcmri.bart` needs boths. All the experiments were done with [`BART 0.7.00`](https://github.com/mrirecon/bart/releases/tag/v0.7.00) with the code from the [GitHub repository](https://github.com/mrirecon/bart) (at commit `e84f4ba7`). 

### 1. Clone repo and install dependencies 

The easiest way of having everything running is installing the code as a pip package. The code uses [Jax](https://jax.readthedocs.io/en/latest/installation.html) and [Flax](https://flax.readthedocs.io/en/latest/), and was tested in the versions given by the `requirements.txt`.

```
$ git clone git@github.com:fsahli/NF-cMRI.git
$ cd NF-cMRI 
$ python -m venv venv
$ source venv/bin/activate 
(venv)$ pip install -r requirements.txt 
```

### 2. Add PATHS to `nfcmri` code 

`nfcmri.bart` needs to access the cloned repository from BART. Please change the `os.environ["TOOLBOX_PATH"]` variable in the `nfcmri.bart` file.

```python

import os, sys

os.environ["TOOLBOX_PATH"] = "/home/tabita/ACIP-MRI/bart" # <-------- reemplace this with the directory where you cloned the BART Toolbox 
path = os.environ["TOOLBOX_PATH"] + "/python" # 
sys.path.append(path)

import cfl
```

We used the [`Replication Data for: Multi-Domain Convolutional Neural Network (MD-CNN) For Radial Reconstruction of Dynamic Cardiac MRI`](dataverse.harvard.edu/dataset.xhtml?persistentId=doi%3A10.7910%2FDVN%2FCI3WB6&version=2.0&q=&fileAccess=&fileTag=&fileSortField=&fileSortOrder=). The radial raw data can be downloaded with the code from `nfcmri.data_harvard`, but you need to set up the variables 

```python
HARVARD_FOLDER = '/home/tabita/ACIP-MRI/NF-cMRI/harvardDB' # <---------- replace with path to store the data
BART_DIRECTORY = '/home/tabita/ACIP-MRI/NF-cMRI/data_bart' # <---------- replace with path to store bart outputs
```

---

### 3. Install code as editable Python library 

Check that there is a `setup.py` file in the directory and run to install in editable mode. This allows using the code in any place in the repo and modifying it

```bash 
pip install -e .
```

You should be able to use the code in Python 

```python
import nfcmri 
```

### 4. Run the demo 

- `demos/nfcmri.py` has a Demo using the code and comparing against TD-DIP and GRASP. 

---

## 👀 Considerations 

We defined a custom interface to the BART library `nfcmri.bart`. The main class for working with radial data is `nfcmri.bart.BARTAcquisition`, which contains the trayectory and data of the acquisitions, and links results obtained from that acquisition such as coil sensitivity maps and reconstructions with different paramters. To avoid repeating calculations, the results are stored as `cfl` files. 

---

<!---
## 🤝 Acknowledgements

This codebase builds on:

- \[Link to any base code you used]
- \[Thank collaborators or funding agencies]
  - iHEALTH

---

## 🛠️ Troubleshooting

* [ ] GPU not found → Make sure CUDA is installed properly
* [ ] Model diverging → Check learning rate or dataset path
* For issues, feel free to open an [issue](https://github.com/yourusername/project/issues)

--

--->

## 📬 Contact

If you have questions, please contact:
- Tabita Catalán – [tcatalanm8@estudiante.uc.cl](mailto:tcatalanm8@estudiante.uc.cl) 
- Francisco Sahli - [fsc@ing.puc.cl](mailto:fsc@ing.puc.cl)

Or open an issue in this repo.
