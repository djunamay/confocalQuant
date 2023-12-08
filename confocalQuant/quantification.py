import numpy.ma as ma
from tqdm import tqdm
import numpy as np
import pandas as pd
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

def get_per_cell_expectation(matrix, probs, index):
    return np.dot(matrix[index], probs[index])

def get_all_expectations(matrix, probs, masks, volume_per_voxel):
    nchannels = matrix.shape[3]
    unique_masks = np.unique(masks)[1:]
    N = len(unique_masks)
    out = np.empty((N,nchannels+2))

    masks = masks.ravel()
    probs = probs.ravel()
    
    for i in tqdm(range(N)):
        maskID = unique_masks[i]
        index = masks==maskID
        N_points = np.sum(index)
        cell_volume = volume_per_voxel * N_points
        for j in range(nchannels):
            out[i,j] = get_per_cell_expectation(matrix[:,:,:,j].ravel(), probs, index)
        out[i,-2] = cell_volume
        out[i,-1] = N_points
    return out



def concatenate_Y(Nfiles, all_Y, cells_per_job, Ncells_per_job, colnames):
    res = []
    for ID in range(Nfiles):
        start = ID*cells_per_job
        end = start + Ncells_per_job[ID][0]
        temp = all_Y[start:end]
        res.append(temp)
        
    data = pd.DataFrame(np.vstack(res))
    data.columns = colnames 
    
    return data
