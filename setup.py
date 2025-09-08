from setuptools import setup

setup(
    name='nfcmri',
    version='0.2.0',
    packages=['nfcmri'],
    author='Tabita Catalan',
    author_email='tabicm.nhg@gmail.com',
    install_requires=[ 
        'flax',
        'image-similarity-measures',
        'ipykernel',
        'matplotlib',
        'numpy',
        'optax',
        'scikit-image',
        'scipy',
        'tqdm',
        'wonderwords',
    ]
)