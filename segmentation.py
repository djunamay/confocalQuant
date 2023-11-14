from aicsimageio import AICSImage
import numpy as np
import ipywidgets as widgets
from functools import partial
from IPython.display import clear_output
from PIL import Image

buttons = widgets.ToggleButtons(
    options=[0,1,2, 'All'],
    description='Speed:',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltips=['Description of slow', 'Description of regular', 'Description of fast'],
)



int_range = widgets.FloatSlider(
    value=99,
    min=85,
    max=100.0,
    step=0.1,
    description='Normalization:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.1f',
)

def load_2D(path, z_slice):
    img = AICSImage(path)
    res = []
    for i in range(3):
        res.append(img.get_image_data("ZXY", C=i)[z_slice])
    out = np.stack(res, axis=2)    
    return out

#@nb.njit(parallel=False)
def update_image(out, percent, channel):
    
    normalize = np.percentile(out, percent, axis=(0,1))
    if channel==['All']:
        im = Image.fromarray(np.clip(out.astype(np.float64)/normalize[None,None]*255.0, 0, 255).astype('uint8'))
    else:
        channel = set(channel)
        all_channels = range(3)
        temp = [x for x in all_channels if x not in channel]
        for i in temp:
            normalize[i] = 1e1000
        im = Image.fromarray(np.clip(out.astype(np.float64)/normalize[None,None]*255.0, 0, 255).astype('uint8'))

    return im

def on_value_change_slider(out, output2, change):
    with output2:  
        clear_output(wait=True)
        display(update_image(out, change['new'], [buttons.value]))

def on_value_change_button(out, output2, change):
    with output2:  
        clear_output(wait=True)
        display(update_image(out, int_range.value, [change['new']]))

def show_im(path, z_slice):
    out = load_2D(path, z_slice)
    output2 = widgets.Output()

    f = partial(on_value_change_slider, out, output2)
    g = partial(on_value_change_button, out, output2)
    
    int_range.observe(f, names='value')
    buttons.observe(g, names='value')
    return widgets.VBox([buttons, int_range, output2])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
