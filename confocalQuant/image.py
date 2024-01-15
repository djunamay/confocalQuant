import numpy as np
from PIL import Image
from skimage.segmentation import find_boundaries
from matplotlib.patches import Rectangle
from .segmentation import hide_masks, gamma_correct_image, extract_channels, float_to_int
from .plotting import get_id_data

def save_mean_proj(mat_sele, mask_sele, outdir):
    """
    Calculate the mean projection of a 3D matrix, perform gamma correction,
    extract specific channels, apply a binary mask specifying cell boundaries, and save the resulting image.

    Parameters:
    - mat_sele (np.ndarray): 3D matrix for mean projection.
    - mask_sele (np.ndarray): Binary mask for channel extraction.
    - outdir (str): Output directory to save the processed image.

    Returns:
    - int: Randomly generated ID for the saved image.
    """
    mat_proj = np.mean(mat_sele, axis=(0))
    gamma_dict={0: 1, 1: 1, 2: 1, 3:1}
    lower_dict={0: 0, 1: 0, 2: 0, 3:0}
    upper_dict={0: 100, 1: 99.5, 2: 100, 3:100}

    mat_g = gamma_correct_image(mat_proj, gamma_dict, lower_dict, upper_dict, is_4D=False)
    show = extract_channels([1], mat_g, is_4D=False)
    show_copy = show.copy()

    index = mask_sele>0
    index = np.max(index, axis=(0))
    index = index.astype(int)
    index = find_boundaries(index, mode = 'outer', background = 0) 
    for i in range(show.shape[-1]):
        show_copy[:,:,i][index]=1

    temp = np.zeros((mat_proj.shape[0],10,3))
    test = np.concatenate((show, temp, show_copy), axis = (1))
    image = Image.fromarray(float_to_int(test))
    random_ID = np.random.randint(10**9, 10**10) 
    path = outdir +str(random_ID)+'.png'
    image.save(path)

    return random_ID

def add_inset(axes, j, i, plt, x1=200,x2=400,y1=400,y2=600):
    """
    Add an inset image to a specified subplot in a set of axes.

    Parameters:
    - axes (array-like): The set of axes to which the inset will be added.
    - j (int): The row index of the subplot in the axes array.
    - i (int): The column index of the subplot in the axes array.
    - plt: The image to be inset.
    - x1, x2, y1, y2 (int, optional): The limits of the inset area. Default values are provided.

    Returns:
    None
    """
    axin = axes[j,i].inset_axes([.4, .4, 0.6, 0.6])
    axin.set_xlim(x1, x2)
    axin.set_ylim(y1, y2)
    axin.imshow(plt)
    axes[j,i].indicate_inset_zoom(axin)
    axin.set_xticks([])
    axin.set_yticks([])
    border = Rectangle((0, 0), 5, 5, color='white', linewidth=5, fill=False, transform=axin.transAxes)
    axin.add_patch(border)
    
def add_scale_bar(size, img, plt):
    """
    Add a scale bar to an image plot.

    Parameters:
    - size (float): The size of the scale bar in physical units.
    - img: The image object with physical pixel size information.
    - plt: The image plot to which the scale bar will be added.

    Returns:
    None
    """
    end = np.round(size/img.physical_pixel_sizes[2])
    for i in range(3):
        #plt[10:(10+int(end)),190:195,i] = 1
        plt[10:15,150:(150+int(end)),i] = 1
        
def plot_axis(axes, plt, j, i, size, img, collabs, rowlabs):
    """
    Plot an axis with an image, scale bar, and labels.

    Parameters:
    - axes: The array of subplots.
    - plt: The image plot to be displayed.
    - j (int): The row index of the subplot.
    - i (int): The column index of the subplot.
    - size (float): The size of the scale bar in physical units.
    - img: The image object with physical pixel size information.
    - collabs: The labels for the columns.
    - rowlabs: The labels for the rows.

    Returns:
    None
    """
    add_scale_bar(size, img, plt)
    axes[j,i].imshow(plt, origin = 'lower')
    axes[j,i].set_xticks([])
    axes[j,i].set_yticks([])
    if j==0:
        axes[j,i].xaxis.set_label_position('top')
        axes[j,i].set_xlabel(collabs[i], fontsize=23)
    if i==0:
        axes[j,i].set_ylabel(rowlabs[j], fontsize=23)
        
def get_mean_projections(mat, mask, background_dict, gamma_dict, lower_dict, upper_dict, channels, order, mask_channel, maskit=True, percentile=True, mean=True):
    """
    Generate mean or max projections from a 4D matrix, considering background subtraction, gamma correction, and channel extraction.

    Parameters:
    - mat: The 4D matrix containing image data.
    - mask: The binary mask for masking image data.
    - background_dict: A dictionary of background values for each channel.
    - gamma_dict: A dictionary of gamma values for each channel.
    - lower_dict: A dictionary of lower thresholds for each channel.
    - upper_dict: A dictionary of upper thresholds for each channel.
    - channels: The list of channel indices to extract.
    - order: The order of channels in the output.
    - mask_channel: The channels to consider when applying the mask.
    - maskit (bool): Flag to apply the mask.
    - percentile (bool): Flag to use percentile thresholds.
    - mean (bool): Flag to compute mean or max projections.

    Returns:
    - show_ordered: The resulting mean or max projection with adjusted properties.
    """
    mat_sub = bgrnd_subtract(mat, np.array(list(background_dict.values())))
    if maskit:
        mat_sub_masked = mat_sub.copy()
        for x in mask_channel:
            mat_sub_masked[mask==0,x]=0
        if mean:
            mat_proj = np.mean(mat_sub_masked, axis = (0))
        else:
            mat_proj = np.max(mat_sub_masked, axis = (0))

    else:
        if mean:
            mat_proj = np.mean(mat_sub, axis = (0))
        else:
            mat_proj = np.max(mat_sub, axis = (0))

    mat_g = gamma_correct_image(mat_proj, gamma_dict, lower_dict, upper_dict, is_4D=False, percentile=percentile)
    show = extract_channels(channels, mat_g, is_4D=False)
    show_ordered = show.copy()
    for i in range(show_ordered.shape[-1]):
        show_ordered[:,:,i] = show[:,:,order[i]]
    return show_ordered