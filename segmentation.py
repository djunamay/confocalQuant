from aicsimageio import AICSImage
import numpy as np
from functools import partial
from IPython.display import clear_output
from PIL import Image
import ipywidgets as widgets


from widgets import buttons, upper_range, int_range_v, lower_range, dropdown_soma, dropdown_nuc, buttons2, text, int_range_seg

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

    dropdown_soma.options = opt
    dropdown_nuc.options = opt

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
    return widgets.VBox([buttons, buttons2, upper_range, lower_range, int_range_v, text, output2])


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

def show_segmentation(loaded_mat2, loaded_M, i):

    T = loaded_mat2[i]
    T[:,:,0][np.invert(loaded_M[i].mask)] = 255
    T[:,:,1][np.invert(loaded_M[i].mask)] = 255
    T[:,:,2][np.invert(loaded_M[i].mask)] = 255
    
    im = Image.fromarray(T)
    
    return im


def on_slide(output2, loaded_mat2, loaded_M, change):
    with output2:  
        clear_output(wait=True)
        display(show_segmentation(loaded_mat2, loaded_M, change['new']))
    
def toggle_segmentation(loaded_mat2, loaded_M):    
    output2 = widgets.Output()
    int_range_seg.max = loaded_mat2.shape[0]-1
    e = partial(on_slide, output2, loaded_mat2, loaded_M)
    int_range_seg.observe(e, names='value')
    
    return widgets.VBox([int_range_seg, output2])
