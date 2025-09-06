"""
Funciones y clases para cambiar el tamaño de las imágenes,
y prepararlas para el cálculo de métricas.
"""
import numpy as onp 
from abc import ABC, abstractmethod 
from skimage import morphology as skmorph 


def normalize(img): 
    return img/onp.max(onp.abs(img))

def half_FOV_range(m):
    #if m % 4 != 0: 
    #    print(f'CUIDADO: m = {m} no es divisible por 4')
    return m // 4, m - m// 4 

def reduce_FOV(im): 
    """
    Elimina más o menos la mitad de los píxeles tanto en
    la dim 0 como en la 1. Conserva los centrales.

    ## Notas 
    Las imagenes de MRI radiales suelen adquirirse con
    _oversampling_, por lo que al reconstruirlas se 
    obtienen imágenes con el cuerpo en el centro y una
    gran zona negra alrededor. Esta función permite cortar
    esas imágenes para obtener un FOV más "estándar"
    """
    x0, xf = half_FOV_range(im.shape[0])
    y0, yf = half_FOV_range(im.shape[1])
    return im[x0:xf, y0:yf]

def crop_2d_im(im, ns): 
    """
    Elimina pixeles de los primeros dos axis de un array `im`. 
    La cantidad de píxeles esliminados está dada por `ns`
    
    ## Argumentos 
    - `im`: numpy array, de ndim>=2
    - `ns`: lista[int], `(nl, nr, nu, nd)` de largo 4. Cantidad
        de píxeles a eliminar. `nl` píxeles se eliminan del comienzo
        y `nr` del final de la dimension 0; `nu` y `nd` píxeles se
        eliminan del comienzo y final respectivamente de la dimensión 1.
    
    ## Ejemplo 
    
    La función `crop_2d_im` solo elimina elementos de las primeras
    dos dimensiones, las demás no sufren cambios.

    ```python
    >>> a = onp.array([
            [[1],  [2],  [3],  [4]],
            [[5],  [6],  [7],  [8]],
            [[9], [10], [11], [12]]
        ])
    >>> a.shape
    (3, 4, 1)
    >>> cropped_a = crop_2d_im(a, [1,0,2,1])
    >>> cropped_a 
    array([
        [[ 7]],
        [[11]]
    ])
    >>> cropped_a.shape 
    (2, 1, 1)
    ```
    """
    N1, N2 = im.shape[:2]
    nl, nr, nu, nd = ns
    return im[nl:(N1-nr), nu:(N2-nd)]

def reduce_crop_abs(im, crop_ns):
    im = reduce_FOV(im)
    im = crop_2d_im(im, crop_ns)
    return onp.abs(im)

class ImageProcesor(ABC): 
    """
    Clase abstracta para procesar imágenes, especialmente
    antes de hacer gráficos y calcular métricas.

    Las clases hijas deben implementar el método `process`.
    """
    @abstractmethod 
    def process(self, im):
        pass 

class BeforeNormalizer(ImageProcesor): 
    """
    Reducice el FOV al tamaño estandar, normaliza por el
    valor máximo dentro del nuevo FOV y, finalmente elimina 
    píxeles.

    ## Notas 

    Usualmente util para calcular métricas en torno al corazón, 
    el parámetro `crop_ns` puede usarse para eliminar los pixeles
    innecesarios y conservar solo un zoom del corazón.    
    """
    def __init__(self, crop_ns) -> None:
        self.crop_ns = crop_ns 
    
    def process(self, img):
        img = reduce_FOV(img)
        img = normalize(img)
        img = onp.abs(crop_2d_im(img, self.crop_ns))
        return img 

class AfterNormalizer(ImageProcesor): 
    """Reducir, cropear, normalizar"""
    def __init__(self, crop_ns) -> None:
        self.crop_ns = crop_ns 
    
    def process(self, img):
        img = reduce_crop_abs(img, self.crop_ns)
        img = normalize(img)
        return onp.abs(img)

def compute_scaling_factor(reference_image, target_image):
    """
    Computes the optimal scalar multiplier that minimizes the difference between 
    the reference image and the scaled target image.

    Given two images of the same shape, this function finds a scaling factor 's' such that:
        s * target_image ≈ reference_image
    The factor is computed using least squares minimization:
        s = sum(reference_image * target_image) / sum(target_image^2)

    Parameters:
    - reference_image (ndarray): The reference image with real values.
    - target_image (ndarray): The image to be scaled, with the same shape as reference_image.

    Returns:
    - float: The optimal scaling factor.
    """
    numerator = onp.sum(reference_image * target_image)
    denominator = onp.sum(target_image * target_image)
    scaling_factor = numerator / denominator
    return scaling_factor

class BeforeLinRegNormalizer(ImageProcesor):
    """
    Similar a `BeforeNormalizer`, pero realizar la normalización
    de forma que maximiza el PSNR/minimiza el MSE con respecto a 
    una imagen de referencia.
    """
    def __init__(self, gtim, crop_ns): 
        """
        - `gtim`: imagen completa, sin preprocesamiento previo.
        """
        self.crop_ns = crop_ns 
        self.gtim = reduce_crop_abs(gtim, self.crop_ns)

    def process(self, im):
        im = reduce_crop_abs(im, self.crop_ns)
        scaling_factor = compute_scaling_factor(self.gtim, im) 
        return scaling_factor * im

class Flipper(ImageProcesor):
    """
    Se puede añadir a cualquier `ImageProcesor`. Añade un flip al final
    del procesamiento en torno al eje deseado.
    """
    def __init__(self, improc:ImageProcesor, axis=0):
        self.axis=axis
        self.improc = improc
    
    def process(self, im):
        return onp.flip(self.improc.process(im), axis=self.axis)

def threshold_convex_mask(im, level:float):
    """
    im: array (nx, ny, frames), se promedian los frames 
    level: float entre 0 y 1
    """
    mask = onp.mean(onp.abs(im), axis=-1)
    mask = mask > (level * mask).max() 

    mask = skmorph.convex_hull_image(mask)
    return mask