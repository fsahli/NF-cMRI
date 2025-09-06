# %%
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as onp
from nfcmri.evaluation.image_processing import crop_2d_im, reduce_FOV, compute_scaling_factor
from typing import Sequence

def make_cax(ax): 
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    return cax 

def make_annotated_heatmap(ax, matrix, xvals, yvals, rotatexlabels = False, vlims = (None,None), colorbar = True): 
    heatmap = ax.imshow(matrix, cmap = 'magma', vmin = vlims[0], vmax = vlims[1])
    if colorbar: 
        cbar = plt.gcf().colorbar(heatmap, cax=make_cax(ax), orientation='vertical')
    else: 
        cax = make_cax(ax)
        cax.set_axis_off()
        cbar = None

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(onp.arange(len(xvals)), labels=xvals)
    ax.set_yticks(onp.arange(len(yvals)), labels=yvals)

    if rotatexlabels:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    return cbar  


def desactivate_ticks(ax): 
    ax.tick_params(
        # axis='x',       # changes apply to x-axis and y-axis 
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False,
        right=False,
        labelbottom=False,
        labeltop=False,
        labelleft=False,
        labelright=False) # labels along the bottom edge are off

def disable_border(ax=None): 
    ax = ax or plt.gca()
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_visible(False)

def remove_ticks(ax=None):
    ax = ax or plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])


def make_rectangle(imshape, crop, linewidth, linecolor): 
    H, W = imshape[:2]
    n0, n1, n2, n3 = crop 
    return Rectangle((n2, n0), W - n2 - n3, H - n0 - n1, linewidth=linewidth, color = linecolor, fill=False)

def full_halph_FOV_space_time(reco_list, crop_ns, frame:int = 0,
                              saturation:float = 1.0,
                              scale:float=1.0,
                              dpi:int=100,
                              vmax:float=None,
                              xy_proportion=(3,7),
                              adjust_aspect_ratio:bool=False, 
                              adjust_frame:bool=False, 
                              nframes:int=None,
    ):
    """
        Hace 4 plots: full, crop (ambas en un frame), y perfiles temporales vertical y horizontal de crop 

    Está pensada para recibir una imagen luego de un reduce_FOV (es decir, sin oversampling).
    El crop_ns permite hacer un zoom a un area específica (ver PATIENT_HEART_LOCATION en data_harvard para
    encontrar crops ya calculados para ese dataset)

    ## Argumentos 

    - reco_list: List[array], cada array de shape (N,N,frames)
        usualmente uso aquí una reco con FOV sin oversampling 
    - frame: el frame que se muestra en los cortes
    - crop_ns: Tuple[int,int,int,int]
        valores positivo, cantidad de pixeles a eliminar arriba, abajo, a la derecha y a la izquierda 
    - `vmax` y `saturation`: controlan la saturación de las imagenes (a partir de qué valores los píxeles
        se muestran blancos). `vmax=None` por default, y ahí se ocupa `abs(reco_list[0]).max()` para todos.
        Saturation sirve para lo mismo, debería ser un valor >0 (usualmente <=1) que multiplica al `vmax`.
    - scale: escala del figsize (relacionado con el tamaño de los título, cuando hay)
    - `xy_proportion:Tuple[num,num]`: valores positivos, permiten controlar la proporción entre ancho y alto de la figura.
        es útil cuando se necesita un ajuste más fino (por ejemplo, achicar el tamaño de los bordes).
    - `adjust_aspect_ratio:bool=False`: True pone a todos los perfiles temporales de la misma altura (en
        base al primero), independiente del número de frames.
    - `adjust_frame:bool=False`: True muestra una frame en una posición equivalente, considerando el número
        de frames de cada imagen. Se indica el número con respecto a la primera imagen. 
    - `nframes:int`: when given, uses this number of frames for the aspect ratio. By default, it will use the number of frames of the first reco.
    """
    reco = reco_list[0]
    reco_crop = crop_2d_im(reco, crop_ns)
    vmax = vmax or onp.abs(reco).max()

    nrecos = len(reco_list)

    nx, ny = reco_crop.shape[:2]
    nframes = nframes or reco_crop.shape[2]

    ht_ratios = [1,nx/ny, nframes/ny, nframes/nx]
    fig, axs = plt.subplots(4,nrecos, figsize=(nrecos * scale*xy_proportion[0],scale*xy_proportion[1]), layout='constrained', height_ratios=ht_ratios, dpi=dpi) # 
    if nrecos == 1:
        axs = axs[:,None]
    axs = onp.transpose(axs)
        
    for reco, ax in zip(reco_list, axs): 
        reco_crop = crop_2d_im(reco, crop_ns)

        nframes_this_reco = reco_crop.shape[2]
        if adjust_frame: 
            adjusted_frame = onp.int32(onp.round((frame / nframes) * nframes_this_reco))
            print(frame)
        else: 
            adjusted_frame = frame 
    
        ax[0].imshow(onp.abs(reco)[...,adjusted_frame], cmap='bone', vmax=saturation*vmax)
        rec = make_rectangle(reco.shape, crop_ns, scale*2.5, 'red')
        ax[0].add_artist(rec)

        ax[1].imshow(onp.abs(reco_crop)[...,adjusted_frame], cmap='bone', vmax=saturation*vmax)
        aspect =  nframes/nframes_this_reco if adjust_aspect_ratio else 1
        ax[2].imshow(onp.abs(reco_crop[reco_crop.shape[0]//2,:,:]).transpose(), cmap='bone',vmax=saturation*vmax, aspect=aspect)
        ax[3].imshow(onp.abs(reco_crop[:,reco_crop.shape[1]//2,:]).transpose(), cmap='bone', vmax=saturation*vmax, aspect=aspect)

    for ax_ in axs.flatten():
        remove_ticks(ax_)
    
    return fig, axs 

class NamedReco: 
    def __init__(self, im, name:str):
        self.im   = im
        self.name = name

def scale_im(ref, im):
    factor = compute_scaling_factor(onp.mean(onp.abs(ref), axis=-1), onp.mean(onp.abs(im), axis=-1))
    im = factor * im
    return im 

def plot_named_recos(named_recos:Sequence[NamedReco], framelist, crop_ns, title='', ref=None):

    ref = ref or named_recos[0].im

    for frame in framelist:

        fig, axs = full_halph_FOV_space_time(
            [reduce_FOV(scale_im(ref, reco.im)) for reco in named_recos],
            crop_ns=crop_ns,
            adjust_aspect_ratio=True,
            saturation=0.4,
            nframes=30, 
            frame=frame, adjust_frame=True
        )

        for ax, reco in zip(axs, named_recos): 
            # metric_ssim = .append(METRIC_FUNCTIONS['ssim_3D'](
            #     reduce_crop_abs(im_reference, crop_ns),
            #     reduce_crop_abs(scale_im(cine_final), crop_ns)
            # ))
            ax[0].set_title(reco.name)
        plt.suptitle(title)
