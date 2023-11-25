from aicsimageio import AICSImage
import numpy as np
from functools import partial
from IPython.display import clear_output
from PIL import Image
import ipywidgets as widgets
from tqdm import tqdm
from skimage.segmentation import find_boundaries
import matplotlib.pyplot as plt
from scipy import signal
import numba as nb
import os


import numpy as np
from aicsimageio import AICSImage
import torch as ch
from tqdm import tqdm
from os import path

import argparse
from distutils.util import strtobool
import os

import ast

from cellpose import models

from .widgets import buttons, upper_range, int_range_v, lower_range, dropdown_soma, dropdown_nuc, buttons2, text, int_range_seg

def load_3D(img, N_channels):
    res = []
    for i in N_channels:
        res.append(img.get_image_data("ZXY", C=i))
    out = np.stack(res, axis=3)    
    return out

def load_2D(img, z_slice, N_channels):
    res = []
    for i in N_channels:
        res.append(img.get_image_data("ZXY", C=i)[z_slice])
    out = np.stack(res, axis=2)    
    return out

def update_bounds(bounds, buttons, new_value, higher=True):
    if higher:
        for i in buttons.value:
            bounds[i] = (bounds[i][0], new_value)
    else:
        for i in buttons.value:
            bounds[i] = (new_value, bounds[i][1])

def update_image(img, lower_percentile, upper_percentile, channel, zi, N_channels, bounds):
    
    out = load_2D(img, zi, N_channels)

    channel = set(channel)
    all_channels = range(len(N_channels))
    temp = [x for x in all_channels if x not in channel]
    
    mat = out.copy().astype('uint8')
    for i in bounds:
        mat[:,:,i] = threshold_im(out, bounds[i][0], bounds[i][1])[:,:,i]
    
    for i in temp:
        mat[:,:,i] = 0
        
    im = Image.fromarray(mat)

    return im

def on_value_change_slider_upper(img, output2, lower_range, buttons, int_range_v, N_channels, bounds, buttons2,text, change):
    update_bounds(bounds, buttons2, change['new'], higher=True)
    text.value = '<br>'.join(['Channel ' + str(i) + ' lower bound = ' + str(bounds[i][0]) + '; higher bound = ' + str(bounds[i][1])for i in range(len(bounds))])
    
    with output2:  
        clear_output(wait=True)
        display(update_image(img, lower_range.value, change['new'], buttons.value, int_range_v.value, N_channels, bounds))
        
def on_value_change_slider_lower(img, output2, upper_range, buttons, int_range_v, N_channels, bounds, buttons2, text, change):
    update_bounds(bounds, buttons2, change['new'], higher=False)
    text.value = '<br>'.join(['Channel ' + str(i) + ' lower bound = ' + str(bounds[i][0]) + '; higher bound = ' + str(bounds[i][1])for i in range(len(bounds))])
    
    with output2:  
        clear_output(wait=True)
        display(update_image(img, change['new'], upper_range.value, buttons.value, int_range_v.value, N_channels, bounds))

def on_value_change_button(img, output2, lower_range, upper_range, int_range_v, N_channels, bounds, change):
    with output2:  
        clear_output(wait=True)
        display(update_image(img, lower_range.value, upper_range.value, change['new'], int_range_v.value, N_channels, bounds))


def on_value_change_slider_vertical(img, output2, lower_range, upper_range, buttons, N_channels, bounds, change):
    with output2:  
        clear_output(wait=True)
        display(update_image(img, lower_range.value, upper_range.value, buttons.value, change['new'], N_channels, bounds))

        
def show_im(path, z_slice=10, N_channels=range(3)):
    img = AICSImage(path)
    
    opt = range(len(N_channels))
    
    bounds = dict(zip(opt, [(1,99),(1,99),(1,99)]))
    
    buttons.options = opt
    buttons2.options = opt

    # dropdown_soma.options = opt
    # dropdown_nuc.options = opt

    int_range_v.max = img.dims['Z'][0]
    
    output2 = widgets.Output()

    e = partial(on_value_change_slider_vertical, img, output2, lower_range, upper_range, buttons, N_channels, bounds)
    f = partial(on_value_change_slider_upper, img, output2, lower_range, buttons, int_range_v, N_channels, bounds, buttons2, text)
    g = partial(on_value_change_button, img, output2, lower_range, upper_range, int_range_v, N_channels, bounds)
    h = partial(on_value_change_slider_lower, img, output2, upper_range, buttons, int_range_v, N_channels, bounds, buttons2, text)

    upper_range.observe(f, names='value')
    lower_range.observe(h, names='value')
    buttons.observe(g, names='value')
    int_range_v.observe(e, names='value')
    return widgets.VBox([buttons, buttons2, upper_range, lower_range, int_range_v, text, output2]), bounds

def get_czi_files(directory): # this function is chatGPT3
    files = [file for file in os.listdir(directory) if file.endswith(".czi")]
    return sorted(files)

def extract_sbatch_parameters(file_path):
    parameters = {}

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()

            # Ignore comments
            if line.startswith("#"):
                continue

            # Extract key-value pairs or parameters in list form
            parts = line.split()
            if len(parts) >= 2:
                key, *values = parts
                if '\\' in values:
                    # Handle parameters in the form of "--key value1 value2 \"
                    values = values[:values.index('\\')]
                parameters[key] = values
            elif len(parts) == 1:
                # Handle parameters in list form
                parameters.setdefault('list_parameters', []).extend(parts)

    return parameters

def get_anisotropy(img):
    temp = img.physical_pixel_sizes
    return temp.Z/temp.X

def do_inference(mat, do_3D, model, progressbar=None, anisotropy=None, diameter=20, channels=[2,0], zi = 15, channel_axis = 3, z_axis = 0, min_size = 1000, normalize=False):
    if do_3D is False:
        masks, flows, styles, _ = model.eval(mat[zi], diameter=diameter, channels=channels, do_3D=do_3D, progress=progressbar, normalize = normalize)
    elif do_3D:
        masks, flows, styles, _ = model.eval(mat, diameter=diameter, channels=channels, anisotropy=anisotropy, channel_axis=channel_axis, z_axis=z_axis, do_3D=do_3D, min_size=min_size, progress=progressbar, normalize = normalize)        
    return masks, flows

def reduce_to_3D(out_float, is_4D=True):
    if is_4D:
        axis = (0,1,2)
    else:
        axis = (0,1)
        
    temp = np.max(out_float, axis=axis)
    keep = list(np.where(temp>0)[0])
    zeros = list(np.where(temp==0)[0])

    for i in range(3-len(keep)):
        keep.append(zeros[i])

    if is_4D:
        out_float = out_float[:,:,:,np.sort(keep)]
    else:
        out_float = out_float[:,:,np.sort(keep)]
    return out_float

def extract_channels(keep_channel, mat, is_4D=True):
    mat2 = mat.copy()

    if is_4D:
        channels = range(mat.shape[3])
        channels_discard = np.array(channels)[[x not in set(keep_channel) for x in channels]]
    
        for i in channels_discard:
            mat2[:,:,:,i] = 0
        mat2 = reduce_to_3D(mat2, is_4D=is_4D)
    else:
        channels = range(mat.shape[2])
        channels_discard = np.array(channels)[[x not in set(keep_channel) for x in channels]]
    
        for i in channels_discard:
            mat2[:,:,i] = 0
        mat2 = reduce_to_3D(mat2, is_4D=is_4D)
            
    return mat2

def apply_thresh_all_Z(out, bounds):
    mat = out.copy().astype('uint8')
    for i in bounds:
        for j in tqdm(range(mat.shape[0])):
            mat[j,:,:,i] = threshold_im(out[j], bounds[i][0], bounds[i][1])[:,:,i]
    return mat

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def threshold_im(array, lower_percentile, upper_percentile):
    if upper_percentile is None:
        upper = np.max(array, axis=(0,1))
    else:
        upper = np.percentile(array, upper_percentile, axis=(0,1))
    if lower_percentile is None:
        lower = 0
    else:
        lower = np.percentile(array, lower_percentile, axis=(0,1))
    return ((np.clip(array.astype(np.float64)-lower, 0, upper)/upper)*255).astype('uint8')

def load_3D(img, N_channels):
    res = []
    for i in N_channels:
        res.append(img.get_image_data("ZXY", C=i))
    out = np.stack(res, axis=3)    
    return out

def show_segmentation(loaded_mat2, M, i):
    
    masked = np.where(M[i])

    T = loaded_mat2[i]
    T[:,:,0][masked] = 255
    T[:,:,1][masked] = 255
    T[:,:,2][masked] = 255
    
    im = Image.fromarray(T)
    
    return im


def on_slide(output2, loaded_mat2, loaded_M, change):
    with output2:  
        clear_output(wait=True)
        display(show_segmentation(loaded_mat2, loaded_M, change['new']))
    
def toggle_segmentation(mat2, masks):    
    output2 = widgets.Output()
    int_range_seg.max = mat2.shape[0]-1
    o = [find_boundaries(masks[i], mode = 'outer', background = 0) for i in range(masks.shape[0])]
    M = np.stack(o, axis=0)

    e = partial(on_slide, output2, mat2, M)
    int_range_seg.observe(e, names='value')
    
    return widgets.VBox([int_range_seg, output2])

def show_maxproj_with_outlines(mat2, masks):
    max_proj = np.mean(mat2, axis=(0))

    for i in tqdm(range(masks.shape[0])):
        M = find_boundaries(masks[i], mode = 'outer', background = 0)
        max_proj[:,:,0][np.where(M)] = 255
        max_proj[:,:,1][np.where(M)] = 255
        max_proj[:,:,2][np.where(M)] = 255


    return max_proj
    
    
def hide_masks(Y, masks_copy, dictionary):
    hide_masks = np.where((Y[:,0]<dictionary[0]) | (Y[:,1]<dictionary[1]) | (Y[:,2]<dictionary[2]))[0]+1
    for i in hide_masks:
        masks_copy[np.where(masks_copy==i)]=False
        

def run_med_filter(out_float, kernel=3, is_4D = True):
    out_med = out_float.copy()

    if is_4D:
        for i in tqdm(range(out_med.shape[0])):
            for j in range(out_med.shape[-1]):
                out_med[i,:,:,j] = med_filter(matrix=out_float[i,:,:,j], kernel=kernel)
    else:
        for j in range(out_med.shape[-1]):
            out_med[:,:,j] = med_filter(matrix=out_float[:,:,j], kernel=kernel)
        
    return out_med
        

def show_meanproj_with_outlines(mat2, masks):
    max_proj = np.mean(mat2, axis=(0))

    for i in tqdm(range(masks.shape[0])):
        M = find_boundaries(masks[i], mode = 'outer', background = 0)
        max_proj[:,:,0][np.where(M)] = 255
        max_proj[:,:,1][np.where(M)] = 255
        max_proj[:,:,2][np.where(M)] = 255


    return max_proj

def gamma_correct_channel(image_float, gamma, lower, upper):
    
    # threshold 
    lower = np.percentile(image_float, lower)
    upper = np.percentile(image_float, upper)
    
    # Clip the values to be in the valid range [0, 1]
    image_float = np.clip((image_float-lower)/(upper-lower), a_min = 0, a_max = 1)
    
    # Apply gamma correction
    image_corrected = np.power(image_float, gamma)
    
    return image_corrected

def gamma_correct_image(im, gamma_dict, lower_dict, upper_dict, is_4D=True):
    im_corrected = im.copy()
    
    if is_4D:
        for i in range(len(gamma_dict)):
            im_corrected[:,:,:,i] = gamma_correct_channel(im[:,:,:,i], gamma_dict[i], lower_dict[i], upper_dict[i])
    else:
        for i in range(len(gamma_dict)):
            im_corrected[:,:,i] = gamma_correct_channel(im[:,:,i], gamma_dict[i], lower_dict[i], upper_dict[i])
    return im_corrected

def int_to_float(out):
    if out.dtype=='uint16':
        return out.astype(float)/((2**16)-1)
    elif out.dtype=='uint8':
        return out.astype(float)/((2**8)-1)

def float_to_int(out, dtype='uint8'):
    if dtype=='uint16':
        return (out*((2**16)-1)).astype('uint16')
    elif dtype=='uint8':
        return (out*((2**8)-1)).astype('uint8')

@nb.njit(parallel=True)
def bgrnd_subtract(matrix, percentile):
    res = []
    for i in range(matrix.shape[-1]):
        res.append(estimate_percentile(matrix, percentile[i], i))

    out = matrix-np.array(res)
    
    return  np.clip(out, 0, 1)

@nb.njit(parallel=True)
def estimate_percentile(matrix, percentile, channel):
    N = matrix.shape[0]
    x = 0
    for i in nb.prange(N):
        x+=np.percentile(matrix[i,:,:,channel], percentile)
    return x/N

def display_image(out_float, kernel_size, show, gamma_dict, background_dict, lower_dict, upper_dict, Zi):
    
    # do background subtraction
    out_float_subtract = bgrnd_subtract(out_float, np.array(list(background_dict.values())))

    matrix = out_float_subtract[Zi]
    
    # run med filter to remove noise
    out_med = run_med_filter(matrix, kernel = kernel_size, is_4D = False)

    # perform gamma correction for visualization
    out_med_gamma = gamma_correct_image(out_med, gamma_dict, lower_dict, upper_dict, is_4D = False)
    
    # only show selected channels
    out_med_gamma_bgrnd_sele = extract_channels(show,out_med_gamma, is_4D=False)   
    
    # return selected z channel
    im = Image.fromarray(float_to_int(out_med_gamma_bgrnd_sele))
    return im

def on_filter_change(widget_output, out_float, kernel_size, adjust, show, gamma_dict, background_dict, gamma_new_val, background_new_val, lower_dict, upper_dict, upper_new_val, lower_new_val, Zi,change):
    
    with widget_output:  
        clear_output(wait=True)

        update_dict(gamma_dict, adjust.value, gamma_new_val.value)
        update_dict(background_dict, adjust.value, background_new_val.value)
        update_dict(lower_dict, adjust.value, lower_new_val.value)
        update_dict(upper_dict, adjust.value, upper_new_val.value)
    
        # print(background_dict)
        # print(lower_dict)
        # print(upper_dict)
        # print(Zi)
        # print(show)
        
        display(display_image(out_float, kernel_size.value, show.value, gamma_dict, background_dict, lower_dict, upper_dict, Zi.value))


def create_init_dict(N_channels, val):
    empty_dict = {}
    for i in range(N_channels):
        empty_dict[i] = val
    return empty_dict
    
def toggle_filters(out_float):

    N_channels = out_float.shape[3]
    channel_adjust = create_buttons(range(N_channels), [0], 'Adjust:')
    channel_show = create_buttons(range(N_channels), [1], 'Show:')
    
    zi_slider = create_slider_int(10, 0, out_float.shape[0]-1, 1, 'Zi:')
    gamma_slider = create_slider_float(1, 0, 5, 0.01, 'gamma:')
    median_slider = create_slider_int(1, 1, 51, 2, 'median:')
    background_slider = create_slider_float(0, 0, 100, 0.01, 'background:')
    lower_slider = create_slider_float(0, 0, 100, 0.01, 'lower:')
    upper_slider = create_slider_float(100, 0, 100, 0.01, 'upper:')

    widget_output = widgets.Output()
    
    gamma_dict = create_init_dict(N_channels, 1)
    lower_dict = create_init_dict(N_channels, 0)
    upper_dict = create_init_dict(N_channels, 100)
    background_dict = create_init_dict(N_channels, 0)
    
    f = partial(on_filter_change, widget_output, out_float, median_slider, channel_adjust, channel_show, gamma_dict, background_dict, gamma_slider, background_slider, lower_dict, upper_dict, upper_slider, lower_slider,zi_slider)
    
    channel_show.observe(f, names='value')
    channel_adjust.observe(f, names='value')

    zi_slider.observe(f, names='value')
    gamma_slider.observe(f, names='value')
    median_slider.observe(f, names='value')
    background_slider.observe(f, names='value')
    lower_slider.observe(f, names='value')
    upper_slider.observe(f, names='value')
    
    return widgets.VBox([channel_show, channel_adjust, zi_slider, gamma_slider, median_slider, background_slider, lower_slider, upper_slider, widget_output]), median_slider.value, background_dict, gamma_dict, upper_dict, lower_dict

def update_dict(dictionary, adjust, new_val):
    for i in adjust:
        dictionary[i] = new_val
            
def create_slider_float(value, min, max, step, description):
    slider = widgets.FloatSlider(
    value=value,
    min=min,
    max=max,
    step=step,
    description=description,
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.1f',
)
    return slider


def create_slider_int(value, min, max, step, description):
    slider = widgets.IntSlider(
    value=value,
    min=min,
    max=max,
    step=step,
    description=description,
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.1f',
)
    return slider

def create_buttons(options, value, description):
    
    buttons = widgets.SelectMultiple(
    options=options,
    value=value,
    description=description,
    disabled=False
)
    return buttons

@nb.njit(parallel=True)
def med_filter(matrix, kernel):
    """
    Apply a 2D median filter to the input matrix.

    Parameters:
    - matrix (numpy.ndarray): The input 2D matrix.
    - kernel (int): The size of the square filter kernel. Should be an odd number.

    Returns:
    - numpy.ndarray: The filtered matrix of the same shape as the input.

    Note:
    The function uses the median value within a square window of size 'kernel'
    centered at each element in the input matrix to calculate the filtered result.

    Example:
    >>> input_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> filtered_matrix = med_filter(input_matrix, kernel=3)
    """
    out = matrix.copy()
    k = (kernel-1)/2
    Nrow, Ncol = matrix.shape
    for i in nb.prange(Nrow):
        for j in range(Ncol):
            i_lower = i-k
            i_upper = i+k+1
            j_lower = j-k
            j_upper = j+k+1

            if i_upper > Nrow:
                i_upper = Nrow
            if i_lower < 0:
                i_lower = 0
            if j_upper > Ncol:
                j_upper = Ncol
            if j_lower < 0:
                j_lower = 0 

            out[i,j] = np.median(matrix[int(i_lower):int(i_upper)][:,int(j_lower):int(j_upper)])
    return out



def update_image(img, lower_percentile, upper_percentile, channel, zi, N_channels, bounds):
    
    out = load_2D(img, zi, N_channels)

    channel = set(channel)
    all_channels = range(len(N_channels))
    temp = [x for x in all_channels if x not in channel]
    
    mat = out.copy().astype('uint8')
    for i in bounds:
        mat[:,:,i] = threshold_im(out, bounds[i][0], bounds[i][1])[:,:,i]
    
    for i in temp:
        mat[:,:,i] = 0
        
    im = Image.fromarray(mat)

    return im

    
    
    
        
def show_im(path, z_slice=10, N_channels=range(3)):
    img = AICSImage(path)
    
    opt = range(len(N_channels))
    
    bounds = dict(zip(opt, [(1,99),(1,99),(1,99)]))
    
    buttons.options = opt
    buttons2.options = opt

    # dropdown_soma.options = opt
    # dropdown_nuc.options = opt

    int_range_v.max = img.dims['Z'][0]
    
    output2 = widgets.Output()

    e = partial(on_value_change_slider_vertical, img, output2, lower_range, upper_range, buttons, N_channels, bounds)
    f = partial(on_value_change_slider_upper, img, output2, lower_range, buttons, int_range_v, N_channels, bounds, buttons2, text)
    g = partial(on_value_change_button, img, output2, lower_range, upper_range, int_range_v, N_channels, bounds)
    h = partial(on_value_change_slider_lower, img, output2, upper_range, buttons, int_range_v, N_channels, bounds, buttons2, text)

    upper_range.observe(f, names='value')
    lower_range.observe(h, names='value')
    buttons.observe(g, names='value')
    int_range_v.observe(e, names='value')
    return widgets.VBox([buttons, buttons2, upper_range, lower_range, int_range_v, text, output2]), bounds

