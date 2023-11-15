from aicsimageio import AICSImage
import numpy as np
from functools import partial
from IPython.display import clear_output
from PIL import Image
import ipywidgets as widgets

from widgets import buttons, upper_range, int_range_v, lower_range

def load_2D(img, z_slice, N_channels):
    res = []
    for i in N_channels:
        res.append(img.get_image_data("ZXY", C=i)[z_slice])
    out = np.stack(res, axis=2)    
    return out


def update_image(img, lower_percentile, upper_percentile, channel, zi, N_channels):
    
    out = load_2D(img, zi, N_channels)

    channel = set(channel)
    all_channels = range(len(N_channels))
    temp = [x for x in all_channels if x not in channel]
    
    mat = threshold_im(out, lower_percentile, upper_percentile)
    
    for i in temp:
        mat[:,:,i] = 0
        
    im = Image.fromarray(mat)

    return im

def on_value_change_slider_upper(img, output2, lower_range, buttons, int_range_v, N_channels, change):
    with output2:  
        clear_output(wait=True)
        display(update_image(img, lower_range.value, change['new'], buttons.value, int_range_v.value, N_channels))
        
def on_value_change_slider_lower(img, output2, upper_range, buttons, int_range_v, N_channels, change):
    with output2:  
        clear_output(wait=True)
        display(update_image(img, change['new'], upper_range.value, buttons.value, int_range_v.value, N_channels))

def on_value_change_button(img, output2, lower_range, upper_range, int_range_v, N_channels, change):
    with output2:  
        clear_output(wait=True)
        display(update_image(img, lower_range.value, upper_range.value, change['new'], int_range_v.value, N_channels))


def on_value_change_slider_vertical(img, output2, lower_range, upper_range, buttons, N_channels, change):
    with output2:  
        clear_output(wait=True)
        display(update_image(img, lower_range.value, upper_range.value, buttons.value, change['new'], N_channels))

        
def show_im(path, z_slice=10, N_channels=range(3)):
    img = AICSImage(path)
    
    buttons.options = range(len(N_channels))
    int_range_v.max = img.dims['Z'][0]
    
    output2 = widgets.Output()

    e = partial(on_value_change_slider_vertical, img, output2, lower_range, upper_range, buttons, N_channels)
    f = partial(on_value_change_slider_upper, img, output2, lower_range, buttons, int_range_v, N_channels)
    g = partial(on_value_change_button, img, output2, lower_range, upper_range, int_range_v, N_channels)
    h = partial(on_value_change_slider_lower, img, output2, upper_range, buttons, int_range_v, N_channels)

    upper_range.observe(f, names='value')
    lower_range.observe(h, names='value')
    buttons.observe(g, names='value')
    int_range_v.observe(e, names='value')
    return widgets.VBox([buttons, upper_range, lower_range, int_range_v, output2])


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