from .plotting import get_id_data
from .segmentation import hide_masks, gamma_correct_image, extract_channels, float_to_int
import numpy as np
from PIL import Image 
from skimage.segmentation import find_boundaries

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

