"""
The main file, which downloads WGS data from synapse and returns variant annotation and genotype data for specified genes
"""

import numpy as np
from aicsimageio import AICSImage
import torch as ch
from tqdm import tqdm
from os import path

from confocalQuant.segmentation import load_3D, int_to_float, run_med_filter, bgrnd_subtract, get_anisotropy, do_inference,  sigmoid
from confocalQuant.quantification import get_all_expectations

import argparse
from distutils.util import strtobool
import os

import ast

def parse_dict(arg):
    try:
        # Safely evaluate the string as a Python literal expression
        return ast.literal_eval(arg)
    except (SyntaxError, ValueError) as e:
        raise argparse.ArgumentTypeError(f"Invalid dictionary format: {arg}")

        
def get_czi_files(directory): # this function is chatGPT3
    files = [file for file in os.listdir(directory) if file.endswith(".czi")]
    return sorted(files)

def process_image(folder, im_path, ID, model, channels, bounds, filter_dict, diameter, inf_channels, min_size, Ncells, NZi, start_Y, start_zi, xi_per_job, yi_per_job, Njobs):
    print(bounds)
    mmap_1 = path.join(folder, 'mat.npy')

    if not path.exists(mmap_1):
        mode = 'w+'
    else:
        mode = 'r+'

    all_mat = np.lib.format.open_memmap(mmap_1, shape=(NZi, xi_per_job, yi_per_job, len(channels)), dtype='uint8', mode=mode)
    all_masks = np.lib.format.open_memmap(path.join(folder, 'masks.npy'), shape=(NZi, xi_per_job, yi_per_job), dtype='uint16', mode=mode)
    all_Y_filtered = np.lib.format.open_memmap(path.join(folder, 'Y_filtered.npy'), shape=(Ncells, len(channels)+2), dtype=float, mode=mode)
    Ncells_per_job = np.lib.format.open_memmap(path.join(folder, 'Ncells_per_job.npy'), shape=(Njobs,1), dtype=int, mode=mode)
    Nzi_per_job = np.lib.format.open_memmap(path.join(folder, 'Nzi_per_job.npy'), shape=(Njobs,1), dtype=int, mode=mode)

    # load image with channels from above
    img = AICSImage(im_path)
    out = load_3D(img, channels)
    
    # convert to float
    out_float = int_to_float(out)

    # run med filter to remove noise
    out_med = run_med_filter(out_float, kernel = kernel)

    # run background subtraction 
    out_float_subtract = bgrnd_subtract(out_med, np.array(list(background_dict.values())))

    # load model
    model = models.Cellpose(gpu = True, model_type='cyto2')

    # run inference
    masks, flows = do_inference(out_float_subtract, do_3D=True, model=model, anisotropy=anisotropy, diameter=20, channels=[2,3], channel_axis=3, z_axis=0, min_size=1100, normalize = False)
        
    # get expectations
    probs = sigmoid(flows[2])
    Y = get_all_expectations(out_float_subtract, probs, masks)
  
    # filter
    channel_zero_bgrnd_mean = np.mean(out_float_subtract[:,:,:,0][masks==0])
    Y_filtered = Y[Y[:,1]>0.025]
    Y_filtered = np.concatenate((Y_filtered, (Y_filtered[:,0]-channel_zero_bgrnd_mean).reshape(-1,1)), axis=1)
    
    # add image ID
    Y_filtered = np.concatenate((Y_filtered, np.repeat(ID, Y_filtered.shape[0]).reshape(-1,1)), axis=1)
    
    # save mat, masks, and Y_filtered
    print('saving..')
    end_zi = start_zi + mat.shape[0]
    all_mat[start_zi:end_zi] = mat
    all_masks[start_zi:end_zi] = masks
    end_Y = start_Y + Y_filtered.shape[0]
    all_Y_filtered[start_Y:end_Y] = Y_filtered
    Ncells_per_job[ID] = Y_filtered.shape[0]
    Nzi_per_job[ID] = mat.shape[0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, required=True)
    parser.add_argument('--impath', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--channels', type=int, nargs="+", required=True)
    parser.add_argument('--bounds', type=parse_dict, required=True)
    parser.add_argument('--filter_dict', type=parse_dict, required=True)
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

    args = parser.parse_args()
    cells_per_job = args.cells_per_job
    zi_per_job = args.zi_per_job

    print('loading model')
    model = ch.load(args.model_path)
    
    ID = int(os.environ['SLURM_ARRAY_TASK_ID'])
    start_Y = cells_per_job * ID
      
    start_zi = zi_per_job * ID
    print(ID)
    im_path_root = args.impath
    im_path = im_path_root + get_czi_files(im_path_root)[ID]
    print(im_path)
    
    print('starting processing')
    process_image(args.folder, im_path, ID, model, args.channels, args.bounds, args.filter_dict, args.diameter, args.inf_channels, args.min_size, args.Ncells, args.NZi, start_Y, start_zi, args.xi_per_job, args.yi_per_job, args.Njobs)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    