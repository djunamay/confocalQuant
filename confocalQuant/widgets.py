import ipywidgets as widgets
import numpy as np
from ipywidgets import interact, interactive, fixed, interact_manual
from IPython.display import display, clear_output
from PIL import Image
from tqdm import tqdm
from functools import partial

from .segmentation import bgrnd_subtract, gamma_correct_image, run_med_filter, extract_channels, float_to_int, import_im

def toggle_filters(all_files, parent_path, channels, out_float=None):

    """
    Create interactive widgets for adjusting and visualizing image filters.

    Parameters:
    - all_files (List[str]): List of file names.
    - parent_path (str): Parent path where image files are located.
    - channels (List[int]): List of channel indices.
    - out_float (List[np.ndarray], optional): List of precomputed image data arrays. Defaults to None.

    Returns:
    - widgets.VBox: Interactive widget container.

    This function creates a set of interactive widgets for adjusting and visualizing image filters. It allows users
    to adjust parameters such as file ID, Z-plane (Zi), gamma correction, median filter, background subtraction, and more.
    The function returns a widget container (`VBox`) that can be displayed in a Jupyter Notebook.

    Example:
    ```
    all_files = [...]  # List of file names
    parent_path = "/path/to/images/"
    channels = [0, 1, 2]  # List of channel indices

    # Create and display the interactive widget
    interactive_widget = toggle_filters(all_files, parent_path, channels)
    display(interactive_widget)
    ```
    """
    if out_float is None:
        out_float = []
        for i in tqdm(all_files):
            out_float.append(import_im(parent_path+i, channels))

    N_channels = out_float[0].shape[3]
    channel_adjust = create_buttons(range(N_channels), [0], 'Adjust:')
    channel_show = create_buttons(range(N_channels), [1], 'Show:')
    
    im_slider = create_slider_int(0, 0, len(all_files)-1, 1, 'File ID:')
    zi_slider = create_slider_int(10, 0, out_float[0].shape[0]-1, 1, 'Zi:')
    gamma_slider = create_slider_float(1, 0, 5, 0.01, 'gamma:')
    median_slider = create_slider_int(1, 1, 51, 2, 'median:')
    background_slider = create_slider_float(0, 0, 100, 0.01, 'background:')
    lower_slider = create_slider_float(0, 0, 100, 0.01, 'lower:')
    upper_slider = create_slider_float(100, 90, 100, 0.01, 'upper:')

    widget_output = widgets.Output()
    
    gamma_dict = create_init_dict(N_channels, 1)
    lower_dict = create_init_dict(N_channels, 0)
    upper_dict = create_init_dict(N_channels, 100)
    background_dict = create_init_dict(N_channels, 0)
    
    f = partial(on_filter_change, widget_output, out_float, median_slider, channel_adjust, channel_show, gamma_dict, background_dict, gamma_slider, background_slider, lower_dict, upper_dict, upper_slider, lower_slider,zi_slider,parent_path, all_files, im_slider, channels)
    
    channel_show.observe(f, names='value')
    channel_adjust.observe(f, names='value')

    zi_slider.observe(f, names='value')
    gamma_slider.observe(f, names='value')
    median_slider.observe(f, names='value')
    background_slider.observe(f, names='value')
    lower_slider.observe(f, names='value')
    upper_slider.observe(f, names='value')

    im_slider.observe(f, names='value')

    return widgets.VBox([channel_show, channel_adjust, zi_slider, gamma_slider, median_slider, background_slider, lower_slider, upper_slider, im_slider, widget_output])

def display_image(out_float, kernel_size, show, gamma_dict, background_dict, lower_dict, upper_dict, Zi):
    
    """
    Display an image after applying background subtraction, gamma correction, median filter, and channel extraction.

    Parameters:
    - out_float (np.ndarray): 4D array representing the input image data.
    - kernel_size (int): Size of the kernel for median filtering.
    - show (List[int]): List of channel indices to display.
    - gamma_dict (dict): Dictionary containing gamma correction values for each channel.
    - background_dict (dict): Dictionary containing background subtraction values for each channel.
    - lower_dict (dict): Dictionary containing lower intensity values for gamma correction.
    - upper_dict (dict): Dictionary containing upper intensity values for gamma correction.
    - Zi (int): Z-plane index to display.

    Returns:
    - Image.Image: PIL Image object representing the displayed image.

    This function applies background subtraction, gamma correction, median filtering, and channel extraction
    to the input image data. It then returns the resulting image as a PIL Image object.

    Example:
    ```
    out_float = ...  # 4D array representing image data
    kernel_size = 3
    show = [0, 1, 2]  # List of channel indices to display
    gamma_dict = {...}  # Dictionary containing gamma correction values
    background_dict = {...}  # Dictionary containing background subtraction values
    lower_dict = {...}  # Dictionary containing lower intensity values
    upper_dict = {...}  # Dictionary containing upper intensity values
    Zi = 5  # Z-plane index to display

    im = display_image(out_float, kernel_size, show, gamma_dict, background_dict, lower_dict, upper_dict, Zi)
    im.show()
    ```
    """
    # do background subtraction
    out_float_subtract = bgrnd_subtract(out_float, np.array(list(background_dict.values())))

    # perform gamma correction for visualization
    out_med_gamma = gamma_correct_image(out_float_subtract, gamma_dict, lower_dict, upper_dict, is_4D = True)
    
    matrix = out_med_gamma[Zi]
    
    # run med filter to remove noise
    out_med = run_med_filter(matrix, kernel = kernel_size, is_4D = False)

    # only show selected channels
    out_med_gamma_bgrnd_sele = extract_channels(show,out_med, is_4D=False)   
    
    # return selected z channel
    im = Image.fromarray(float_to_int(out_med_gamma_bgrnd_sele))
    return im

def on_filter_change(widget_output, out_float, kernel_size, adjust, show, gamma_dict, background_dict, gamma_new_val, background_new_val, lower_dict, upper_dict, upper_new_val, lower_new_val, Zi,parent_path, all_files, im_slider, channels, change):
    """
    Handle the change in filter parameters and update the displayed image.

    Parameters:
    - widget_output: Output widget for displaying content.
    - out_float (List[np.ndarray]): List of 4D arrays representing image data.
    - kernel_size (IntSlider): Slider widget for adjusting the median filter kernel size.
    - adjust (Dropdown): Dropdown widget for selecting the channel to adjust.
    - show (SelectMultiple): SelectMultiple widget for choosing channels to display.
    - gamma_dict (dict): Dictionary containing gamma correction values for each channel.
    - background_dict (dict): Dictionary containing background subtraction values for each channel.
    - gamma_new_val (FloatSlider): Slider widget for adjusting the new gamma value.
    - background_new_val (FloatSlider): Slider widget for adjusting the new background value.
    - lower_dict (dict): Dictionary containing lower intensity values for gamma correction.
    - upper_dict (dict): Dictionary containing upper intensity values for gamma correction.
    - upper_new_val (FloatSlider): Slider widget for adjusting the new upper intensity value.
    - lower_new_val (FloatSlider): Slider widget for adjusting the new lower intensity value.
    - Zi (IntSlider): Slider widget for adjusting the Z-plane index.
    - parent_path (str): Parent path to the image files.
    - all_files (List[str]): List of file names.
    - im_slider (IntSlider): Slider widget for adjusting the file ID.
    - channels (List[int]): List of channel indices.
    - change: Button widget for triggering the filter change.

    Returns:
    - None

    This function handles the change in filter parameters, updates the relevant dictionaries, and displays
    the updated image based on the selected parameters.
    """
    with widget_output:  
        clear_output(wait=True)

        update_dict(gamma_dict, adjust.value, gamma_new_val.value)
        update_dict(background_dict, adjust.value, background_new_val.value)
        update_dict(lower_dict, adjust.value, lower_new_val.value)
        update_dict(upper_dict, adjust.value, upper_new_val.value)
    
        temp = out_float[im_slider.value]
        
        display(display_image(temp, kernel_size.value, show.value, gamma_dict, background_dict, lower_dict, upper_dict, Zi.value))

def create_init_dict(N_channels, val):
    """
    Create an initial dictionary with specified values for each channel.

    Parameters:
    - N_channels (int): Number of channels.
    - val: Initial value to be assigned to each channel.

    Returns:
    - dict: Initial dictionary with channel indices as keys and specified values as values.

    This function creates a dictionary with keys representing channel indices (0 to N_channels-1)
    and values set to the specified initial value.
    """
    empty_dict = {}
    for i in range(N_channels):
        empty_dict[i] = val
    return empty_dict

def update_dict(dictionary, adjust, new_val):
    """
    Update values in a dictionary based on the specified keys.

    Parameters:
    - dictionary (dict): The dictionary to be updated.
    - adjust (list): List of keys to be updated.
    - new_val: The new value to set for the specified keys.

    This function updates the values in the dictionary for the specified keys
    with the new value provided.
    """
    for i in adjust:
        dictionary[i] = new_val
            
def create_slider_float(value, min, max, step, description):
    """
    Create a FloatSlider widget with specified parameters.

    Parameters:
    - value: The initial value of the slider.
    - min: The minimum value of the slider.
    - max: The maximum value of the slider.
    - step: The step size of the slider.
    - description: The description to be displayed next to the slider.

    Returns:
    - widgets.FloatSlider: The created FloatSlider widget.

    This function creates a FloatSlider widget with specified parameters
    for value, minimum, maximum, step size, and description.
    """
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
    """
    Create an IntSlider widget with specified parameters.

    Parameters:
    - value: The initial value of the slider.
    - min: The minimum value of the slider.
    - max: The maximum value of the slider.
    - step: The step size of the slider.
    - description: The description to be displayed next to the slider.

    Returns:
    - widgets.IntSlider: The created IntSlider widget.

    This function creates an IntSlider widget with specified parameters
    for value, minimum, maximum, step size, and description.
    """
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
    """
    Create a SelectMultiple widget with specified parameters.

    Parameters:
    - options: The list of options for the widget.
    - value: The initial selected values from the options.
    - description: The description to be displayed next to the widget.

    Returns:
    - widgets.SelectMultiple: The created SelectMultiple widget.

    This function creates a SelectMultiple widget with specified parameters
    for options, initial selected values, and description.
    """
    buttons = widgets.SelectMultiple(
    options=options,
    value=value,
    description=description,
    disabled=False
)
    return buttons

#### widgets

text = widgets.HTML(value='')

dropdown_soma = widgets.Dropdown(
    options=[1],
    value=1,
    description='Soma:',
    disabled=False,
)

dropdown_nuc = widgets.Dropdown(
    options=[1],
    value=1,
    description='Nucleus:',
    disabled=False,
)

buttons = widgets.SelectMultiple(
    options=[0, 1, 2, 3],
    value=[0],
    #rows=10,
    description='Show:',
    disabled=False
)

buttons2 = widgets.SelectMultiple(
    options=[0, 1, 2, 3],
    value=[0],
    #rows=10,
    description='Adjust:',
    disabled=False
)

upper_range = widgets.FloatSlider(
    value=99,
    min=0,
    max=100.0,
    step=0.01,
    description='Upper:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.1f',
)

lower_range = widgets.FloatSlider(
    value=1,
    min=0,
    max=100.0,
    step=0.01,
    description='Lower:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.1f',
)

int_range_v = widgets.IntSlider(
    value=7,
    min=0,
    max=10,
    step=1,
    description='Zi',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d'
)

int_range_seg = widgets.IntSlider(
    value=7,
    min=0,
    max=10,
    step=1,
    description='Zi',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d'
)

