"""
The main file, which downloads WGS data from synapse and returns variant annotation and genotype data for specified genes
"""

import numpy as np
from aicsimageio import AICSImage
import torch as ch
from tqdm import tqdm

from confocalQuant.segmentation import load_3D, apply_thresh_all_Z, do_inference, get_anisotropy, sigmoid
from confocalQuant.quantification import get_im_stats

import argparse
from distutils.util import strtobool
import os


def process_image(folder, im_path, start, end, ID, model, channels, bounds, filter_dict, diameter, inf_channels, min_size, Ncells, NZi, start_Y, end_Y, start_zi, end_zi, xi_per_job, yi_per_job, Njobs):
    
    mmap_1 = path.join(folder, 'mat.npy')

    if not path.exists(mmap_1):
        mode = 'w+'
    else:
        mode = 'r+'

    all_mat = np.lib.format.open_memmap(mmap_1, shape=(NZi, xi_per_job, yi_per_job), dtype='uint8', mode=mode)
    all_masks = np.lib.format.open_memmap(path.join(folder, 'masks.npy'), shape=(NZi, xi_per_job, yi_per_job), dtype='uint16', mode=mode)
    all_Y_filtered = np.lib.format.open_memmap(path.join(folder, 'Y_filtered.npy'), shape=(Ncells, len(channels)+1), dtype=float, mode=mode)
    Ncells_per_job = np.lib.format.open_memmap(path.join(folder, 'Ncells_per_job.npy'), shape=(1, Njobs), dtype=int, mode=mode)

    # load image with channels from above
    img = AICSImage(im_path)
    out = load_3D(img, channels)

    # thresholding
    mat = apply_thresh_all_Z(out, bounds)

    # get anisotropy
    anisotropy = get_anisotropy(img)
        
    # do inference
    masks, flows = do_inference(mat, do_3D=True, model=model, anisotropy=anisotropy, diameter=diameter, channels=inf_channels, min_size = min_size)
        
    # get the image stats
    Y = get_im_stats(masks, mat2, probs)

    # filter on pre-specified values
    hide_masks(Y, masks, filter_dict)
        
    # get final filtered values
    Y_filtered = get_im_stats(masks, out, probs)
    
    # add image ID
    Y_filtered = np.concatenate((Y_filtered, np.repeat(ID, Y_filtered.shape[0]).reshape(-1,1)), axis=1)
            
    # save mat, masks, and Y_filtered
    all_mat[start_zi:end_zi] = mat
    all_masks[start_zi:end_zi] = masks
    all_Y_filtered[start_Y:end_Y] = Y_filtered
    Ncells_per_job[ID] = Y_filtered.shape[0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=float, required=False, default=0.5)
    parser.add_argument('--im_path', type=lambda x:bool(strtobool(x)), required=False, default=False)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--channels', type=int, required=True)
    parser.add_argument('--bounds', type=int, required=True)
    parser.add_argument('--filter_dict', type=int, required=True)
    parser.add_argument('--diameter', type=int, required=True)
    parser.add_argument('--inf_channels', type=int, required=True)
    parser.add_argument('--min_size', type=int, required=True)
    parser.add_argument('--Ncells', type=int, required=True)
    parser.add_argument('--cells_per_job', type=int, required=True)
    parser.add_argument('--NZi', type=int, required=True)
    parser.add_argument('--zi_per_job', type=int, required=True)
    parser.add_argument('--xi_per_job', type=int, required=True)
    parser.add_argument('--yi_per_job', type=int, required=True)
    parser.add_argument('--Njobs', type=int, required=True)

    args = parser.parse_args()
    cells_per_job = args.cells_per_job
    zi_per_job = args.zi_per_job

    model = ch.load(args.model_path)
    
    ID = int(os.environ['SLURM_ARRAY_TASK_ID'])
    start_Y = cells_per_job * ID
    end_Y = start_Y + cells_per_job
      
    start_zi = zi_per_job * ID
    end_zi = start_zi + zi_per_job
                                               
    process_image(args.folder, args.im_path, start, end, ID, model, args.channels, args.bounds, args.filter_dict, args.diameter, args.inf_channels, args.min_size, args.Ncells, args.NZi, start_Y, end_Y, start_zi, end_zi, args.xi_per_job, args.yi_per_job, args.Njobs)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    