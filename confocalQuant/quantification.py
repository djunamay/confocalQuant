import numpy.ma as ma
from tqdm import tqdm
import numpy as np

# def compute_single_expectation_per_cell(Y, P, N):
#     return ma.sum(ma.multiply(Y,P))*(1/N)

# def compute_all_expectations_per_cell(MASK, mat, probs, N):
#     res = []
#     for i in range(mat.shape[3]):
#         Y = ma.masked_array(mat[:,:,:,i], MASK)
#         P = ma.masked_array(probs, MASK)
#         res.append(compute_single_expectation_per_cell(Y,P,N))
#     return res

# def get_im_stats(masks, mat, probs):
#     res = []
#     for cell in tqdm(np.unique(masks)[1:]):
#         MASK = masks!=cell
#         N = np.sum(np.invert(MASK))
#         res.append(np.concatenate((compute_all_expectations_per_cell(MASK, mat, probs, N), [N])))
#     Y = np.vstack(res)
#     return Y

def get_per_cell_expectation(matrix, probs, masks, maskID, volume_per_voxel):
    index = masks==maskID
    cell_volume = volume_per_voxel * np.sum(index)
    return np.dot(matrix[index], probs[index])/cell_volume

def get_all_expectations(matrix, probs, masks, volume_per_voxel):
    nchannels = matrix.shape[3]
    unique_masks = np.unique(masks)[1:]
    N = len(unique_masks)
    out = np.empty((N,nchannels))

    masks = masks.ravel()
    probs = probs.ravel()
    
    for j in range(nchannels):
        for i in tqdm(range(N)):
            out[i,j] = get_per_cell_expectation(matrix[:,:,:,j].ravel(), probs, masks, unique_masks[i], volume_per_voxel)
    return out