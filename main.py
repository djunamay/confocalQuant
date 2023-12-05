"""
The main file, which downloads WGS data from synapse and returns variant annotation and genotype data for specified genes
"""

import numpy as np
from aicsimageio import AICSImage
import torch as ch
from tqdm import tqdm
from os import path

from confocalQuant.segmentation import load_3D, int_to_float, run_med_filter, bgrnd_subtract, get_anisotropy, do_inference,  sigmoid, gamma_correct_image
from confocalQuant.quantification import get_all_expectations

import argparse
from distutils.util import strtobool
import os

import ast

from cellpose import models

def parse_dict(arg):
    try:
        # Safely evaluate the string as a Python literal expression
        return ast.literal_eval(arg)
    except (SyntaxError, ValueError) as e:
        raise argparse.ArgumentTypeError(f"Invalid dictionary format: {arg}")

        
def get_czi_files(directory): # this function is chatGPT3
    files = [file for file in os.listdir(directory) if file.endswith(".czi")]
    return sorted(files)

def process_image(folder, im_path, ID, model, channels, y_channel, kernel, per_channel_subtraction_vals, diameter, inf_channels, min_size, Ncells, NZi, start_Y, start_zi, xi_per_job, yi_per_job, Njobs, gamma_dict, lower_thresh_dict, upper_thresh_dict):
    mmap_1 = path.join(folder, 'mat.npy')

    if not path.exists(mmap_1):
        mode = 'w+'
    else:
        mode = 'r+'

    all_mat = np.lib.format.open_memmap(mmap_1, shape=(NZi, xi_per_job, yi_per_job, len(channels)), dtype=float, mode=mode)
    all_masks = np.lib.format.open_memmap(path.join(folder, 'masks.npy'), shape=(NZi, xi_per_job, yi_per_job), dtype='uint16', mode=mode)
    all_Y = np.lib.format.open_memmap(path.join(folder, 'Y.npy'), shape=(Ncells, len(channels)+len(y_channel)+3), dtype=float, mode=mode)
    Ncells_per_job = np.lib.format.open_memmap(path.join(folder, 'Ncells_per_job.npy'), shape=(Njobs,1), dtype=int, mode=mode)
    Nzi_per_job = np.lib.format.open_memmap(path.join(folder, 'Nzi_per_job.npy'), shape=(Njobs,1), dtype=int, mode=mode)

    # load image with channels from above
    print('loading image')
    img = AICSImage(im_path)
    out = load_3D(img, channels)
    
    # convert to float
    out_float = int_to_float(out)

    # run med filter to remove noise
    print('runing median filter')
    out_med = run_med_filter(out_float, kernel = kernel)

    # run background subtraction 
    print('running background subtraction')
    out_float_subtract = bgrnd_subtract(out_med, np.array(per_channel_subtraction_vals))

    # run upper thresholding
    print('upper threshold')
    
    out_float_subtract_correct = gamma_correct_image(out_float_subtract, gamma_dict, lower_thresh_dict, upper_thresh_dict)

    # run inference
    print('doing inference')
    anisotropy = get_anisotropy(img)
    print('Anisotropy: ' + str(anisotropy))
    
    masks, flows = do_inference(out_float_subtract_correct, do_3D=True, model=model, anisotropy=anisotropy, diameter=diameter, channels=inf_channels, channel_axis=3, z_axis=0, min_size=min_size, normalize = False)
        
    # get expectations
    print('computing expectations')
    probs = sigmoid(flows[2])
    volume_per_voxel = np.prod(img.physical_pixel_sizes)
    Y = get_all_expectations(out_float, probs, masks, volume_per_voxel)
  
    # remove background mean for channel of interest
    print('processing Y')
    for i in range(len(y_channel)):
        channel_zero_bgrnd_mean = np.mean(out_float[:,:,:,y_channel[i]][masks==0])
        Y = np.concatenate((Y, (Y[:,y_channel[i]]-channel_zero_bgrnd_mean).reshape(-1,1)), axis=1)

    # add image ID
    Y = np.concatenate((Y, np.repeat(ID, Y.shape[0]).reshape(-1,1)), axis=1)
    
    # save mat, masks, and Y_filtered
    print('saving..')
    actual_NZi = out_float.shape[0]
    actual_Ncells = Y.shape[0]
    
    end_zi = start_zi + actual_NZi
    all_mat[start_zi:end_zi] = out_float
    all_masks[start_zi:end_zi] = masks
    end_Y = start_Y + actual_Ncells
    all_Y[start_Y:end_Y] = Y
    Ncells_per_job[ID] = actual_Ncells
    Nzi_per_job[ID] = actual_NZi

    print('done')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, required=True)
    parser.add_argument('--impath', type=str, required=True)
    parser.add_argument('--model_type', type=str, required=True, default='cyto2')
    parser.add_argument('--channels', type=int, nargs="+", required=True)
    parser.add_argument('--y_channel', type=int, nargs="+", required=True)
    parser.add_argument('--kernel', type=int, required=True)
    parser.add_argument('--bgrnd_subtraction_vals', type=int, nargs="+", required=True)
    parser.add_argument('--diameter', type=int, required=True)
    parser.add_argument('--inf_channels', type=int, nargs="+", required=True)
    parser.add_argument('--min_size', type=int, required=True)
    parser.add_argument('--Ncells', type=int, required=True)
    parser.add_argument('--cells_per_job', type=int, required=True)
    parser.add_argument('--NZi', type=int, required=True)
    parser.add_argument('--zi_per_job', type=int, required=True)
    parser.add_argument('--xi_per_job', type=int, required=True)
    parser.add_argument('--yi_per_job', type=int, required=True)
    parser.add_argument('--Njobs', type=int, required=True)
    parser.add_argument('--gamma_dict', type=parse_dict, required=True)
    parser.add_argument('--lower_thresh_dict', type=parse_dict, required=True)
    parser.add_argument('--upper_thresh_dict', type=parse_dict, required=True)

    args = parser.parse_args()
    cells_per_job = args.cells_per_job
    zi_per_job = args.zi_per_job

    print('loading model')
    model = models.Cellpose(gpu = True, model_type=args.model_type)
    
    ID = int(os.environ['SLURM_ARRAY_TASK_ID'])
    start_Y = cells_per_job * ID
      
    start_zi = zi_per_job * ID
    print(ID)
    im_path_root = args.impath
    im_path = im_path_root + get_czi_files(im_path_root)[ID]
    print(im_path)
    
    print('starting processing')
    process_image(args.folder, im_path, ID, model, args.channels, args.y_channel, args.kernel, args.bgrnd_subtraction_vals, args.diameter, args.inf_channels, args.min_size, args.Ncells, args.NZi, start_Y, start_zi, args.xi_per_job, args.yi_per_job, args.Njobs, args.gamma_dict, args.lower_thresh_dict, args.upper_thresh_dict)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    