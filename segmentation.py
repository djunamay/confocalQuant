from aicsimageio import AICSImage
import numpy as np
from functools import partial
from IPython.display import clear_output
from PIL import Image
import ipywidgets as widgets

from widgets import buttons, int_range, int_range_v

def load_2D(img, z_slice):
    res = []
    for i in range(3):
        res.append(img.get_image_data("ZXY", C=i)[z_slice])
    out = np.stack(res, axis=2)    
    return out


def update_image(img, percent, channel, zi):
    
    out = load_2D(img, zi)
    
    normalize = np.percentile(out, percent, axis=(0,1))

    channel = set(channel)
    all_channels = range(3)
    temp = [x for x in all_channels if x not in channel]
    for i in temp:
        normalize[i] = 1e1000
    im = Image.fromarray(np.clip(out.astype(np.float64)/normalize[None,None]*255.0, 0, 255).astype('uint8'))

    return im

def on_value_change_slider(img, output2, buttons, int_range_v, change):
    with output2:  
        clear_output(wait=True)
        display(update_image(img, change['new'], buttons.value, int_range_v.value))

def on_value_change_button(img, output2, int_range, int_range_v, change):
    with output2:  
        clear_output(wait=True)
        display(update_image(img, int_range.value, change['new'], int_range_v.value))


def on_value_change_slider_vertical(img, output2, int_range, buttons, change):
    with output2:  
        clear_output(wait=True)
        display(update_image(img, int_range.value, buttons.value, change['new']))

        
def show_im(path, z_slice=10):
    img = AICSImage(path)
    out = load_2D(img, z_slice)
    output2 = widgets.Output()

    e = partial(on_value_change_slider_vertical, img, output2, int_range, buttons)
    f = partial(on_value_change_slider, img, output2, buttons, int_range_v)
    g = partial(on_value_change_button, img, output2, int_range, int_range_v)
    
    int_range.observe(f, names='value')
    buttons.observe(g, names='value')
    int_range_v.observe(e, names='value')
    return widgets.VBox([buttons, int_range, int_range_v, output2])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def threshold_im(array, upper_percentile, lower_percentile):
    upper = np.percentile(array, upper_percentile)
    lower = np.percentile(array, lower_percentile)
    return (np.clip(array-lower, 0, upper)/upper)*255
