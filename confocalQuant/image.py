from .plotting import get_id_data
from .segmentation import hide_masks, gamma_correct_image, extract_channels, float_to_int

def save_mean_proj(ID, zi_per_job, Nzi, mat, masks, Y_extracted, Y_filtered, id_d):
    
    # add start and end info
    mat_sele, mask_sele = get_id_data(ID, zi_per_job, Nzi, mat, masks)
    mask_sele2 = hide_masks(Y_extracted, ID, Y_filtered, mask_sele)

    mat_proj = np.mean(mat_sele, axis=(0))
    gamma_dict={0: .5, 1: 1, 2: .5}
    lower_dict={0: 0, 1: 0, 2: 0}
    upper_dict={0: 100, 1: 99.9, 2: 100}

    mat_g = gamma_correct_image(mat_proj, gamma_dict, lower_dict, upper_dict, is_4D=False)
    show = extract_channels([1], mat_g, is_4D=False)

    gamma_dict={0: .5, 1: 1, 2: .5}
    lower_dict={0: 0, 1: 0, 2: 0}
    upper_dict={0: 100, 1: 99.25, 2: 100}

    mat_g = gamma_correct_image(mat_proj, gamma_dict, lower_dict, upper_dict, is_4D=False)
    show_copy = extract_channels([1], mat_g, is_4D=False)


    index = mask_sele2>0
    index = np.max(index, axis=(0))
    index = index.astype(int)
    index = find_boundaries(index, mode = 'outer', background = 0) 
    for i in range(show.shape[-1]):
        show_copy[:,:,i][index]=1

    temp = np.zeros((1024,10,3))
    test = np.concatenate((show, temp, show_copy), axis = (1))
    image = Image.fromarray(float_to_int(test))
    path = './segmentations/'+str(id_d[ID])+'.png'
    image.save(path)
