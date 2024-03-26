"""
Confocal Image Processing Script

This script processes confocal microscopy images using the Cellpose model. It performs segmentation and analysis on specified channels, applying preprocessing steps if required. The results, including projections, masks, and processed data, are saved in the specified output folder.

Parameters:
- --folder (str): Path to the folder where the results will be stored.
- --impath (str): Path to the microscopy image file.
- --model_type (str): Type of Cellpose model to be used (default is 'cyto2').
- --channels (list): List of channel indices to load from the image.
- --y_channel (list): Variable indicating the channel to be plotted.
- --kernel (int): Size of the median filter kernel for noise removal.
- --bgrnd_subtraction_vals (list): List of values for per-channel background subtraction.
- --diameter (int): Estimated diameter of objects in the image.
- --inf_channels (list): List of channel indices to use in inference.
- --min_size (int): Minimum size of objects to consider in segmentation.
- --Ncells (int): Number of cells.
- --cells_per_job (int): Number of cells per job.
- --NZi (int): Number of Z slices.
- --zi_per_job (int): Number of Z slices per job.
- --xi_per_job (int): Number of X indices per job.
- --yi_per_job (int): Number of Y indices per job.
- --Njobs (int): Total number of jobs.
- --gamma_dict (dict): Dictionary mapping channel indices to gamma correction parameters.
- --lower_thresh_dict (dict): Dictionary mapping channel indices to lower thresholds for percentile-based adjustment.
- --upper_thresh_dict (dict): Dictionary mapping channel indices to upper thresholds for percentile-based adjustment.
- --outdir (str): Output directory for saving results.
- --preprocess (bool): Enable preprocessing steps (median filter, background subtraction, thresholding).
- --normalize (bool): Enable normalization of input data before inference.

Usage:
$ python main_script.py --folder path/to/results --impath path/to/image --channels 0 1 2 --y_channel 0 --kernel 3 --bgrnd_subtraction_vals 10 20 30 --diameter 50 --inf_channels 0 1 --min_size 100 --Ncells 500 --cells_per_job 50 --NZi 10 --zi_per_job 2 --xi_per_job 512 --yi_per_job 512 --Njobs 10 --gamma_dict {0: 1.0, 1: 1.2} --lower_thresh_dict {0: 10, 1: 20} --upper_thresh_dict {0: 90, 1: 95} --outdir path/to/output --preprocess --normalize
"""

import os
import argparse
from distutils.util import strtobool
import ast

import numpy as np
import torch as ch
from tqdm import tqdm
from os import path

from aicsimageio import AICSImage
from cellpose import models

from confocalQuant.segmentation import get_czi_files, load_3D, int_to_float, run_med_filter, bgrnd_subtract, get_anisotropy, do_inference,  sigmoid, gamma_correct_image
from confocalQuant.data_handling import parse_dict
from confocalQuant.image import save_mean_proj
    
    
def process_image(folder, im_path, ID, model, channels, y_channel, kernel, per_channel_subtraction_vals, diameter, inf_channels, min_size, Ncells, NZi, start_Y, start_zi, xi_per_job, yi_per_job, Njobs, gamma_dict, lower_thresh_dict, upper_thresh_dict, outdir, preprocess, normalize, stitch):
    """
    Process confocal microscopy images and return cell segmentations using cellpose models.

    Parameters:
    - folder (str): Path to the folder where the results will be stored.
    - im_path (str): Path to the microscopy image file.
    - ID (int): Job ID.
    - model: Cellpose model for inference.
    - channels (list): List of channel indices to load from the image.
    - y_channel (str): Variable indicating the channel to be plotted.
    - kernel (int): Size of the median filter kernel for noise removal.
    - per_channel_subtraction_vals (list): List of values for per-channel background subtraction.
    - diameter (int): Estimated diameter of objects in the image.
    - inf_channels (list): List of channel indices to use in inference.
    - min_size (int): Minimum size of objects to consider in segmentation.
    - Ncells (int): Number of cells.
    - NZi (int): Number of Z slices.
    - start_Y (int): Starting Y index for processing.
    - start_zi (int): Starting Z index for processing.
    - xi_per_job (int): Number of X indices per job.
    - yi_per_job (int): Number of Y indices per job.
    - Njobs (int): Total number of jobs.
    - gamma_dict (dict): Dictionary mapping channel indices to gamma correction parameters.
    - lower_thresh_dict (dict): Dictionary mapping channel indices to lower thresholds for percentile-based adjustment.
    - upper_thresh_dict (dict): Dictionary mapping channel indices to upper thresholds for percentile-based adjustment.
    - outdir (str): Output directory for saving results.
    - preprocess (bool): Whether to apply preprocessing steps (median filter, background subtraction, thresholding) before inference.
    - normalize (bool): Whether to normalize the input data before inference.

    Returns:
    - None: Saves results, including projections, masks, and other processed data, in the specified folder.
    """
        
    mmap_1 = path.join(folder, 'mat.npy')

    if not path.exists(mmap_1):
        mode = 'w+'
    else:
        mode = 'r+'

    all_mat = np.lib.format.open_memmap(mmap_1, shape=(NZi, xi_per_job, yi_per_job, len(channels)), dtype=float, mode=mode)
    all_masks = np.lib.format.open_memmap(path.join(folder, 'masks.npy'), shape=(NZi, xi_per_job, yi_per_job), dtype='uint16', mode=mode)
    all_probs = np.lib.format.open_memmap(path.join(folder, 'probs.npy'), shape=(NZi, xi_per_job, yi_per_job), dtype=float, mode=mode)
    Nzi_per_job = np.lib.format.open_memmap(path.join(folder, 'Nzi_per_job.npy'), shape=(Njobs,1), dtype=int, mode=mode)
    randomID_per_job = np.lib.format.open_memmap(path.join(folder, 'randomID_per_job.npy'), shape=(Njobs,1), dtype=int, mode=mode)

    # load image with channels from above
    print('loading image')
    img = AICSImage(im_path)
    out = load_3D(img, channels)
    
    # convert to float
    out_float = int_to_float(out)

    # run med filter to remove noise
    if preprocess:
        print('preprocess is True')
        print('runing median filter')
        if kernel!=1:
            out_med = run_med_filter(out_float, kernel = kernel)
        else:
            out_med = out_float

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

        masks, flows = do_inference(out_float_subtract_correct, do_3D=True, model=model, anisotropy=anisotropy, diameter=diameter, channels=inf_channels, channel_axis=3, z_axis=0, min_size=min_size, normalize = normalize, stitch=stitch)

    else:
        print('preprocess is False')
        # run inference
        print('doing inference')
        anisotropy = get_anisotropy(img)
        print('Anisotropy: ' + str(anisotropy))
        print('normalize' + str(normalize))
        masks, flows = do_inference(out_float, do_3D=True, model=model, anisotropy=anisotropy, diameter=diameter, channels=inf_channels, channel_axis=3, z_axis=0, min_size=min_size, normalize = normalize, stitch=stitch)

    # save projection 
    print('saving projection')
    np.random.seed(ID)
    if not path.exists(outdir):
        os.makedirs(outdir)
    randomID = save_mean_proj(out_float, masks, outdir)
    print(randomID)
    
    # save mat, masks, and Y_filtered
    print('saving..')
    actual_NZi = out_float.shape[0]    
    end_zi = start_zi + actual_NZi
    all_mat[start_zi:end_zi] = out_float
    all_masks[start_zi:end_zi] = masks
    all_probs[start_zi:end_zi] = sigmoid(flows[2])
    Nzi_per_job[ID] = actual_NZi
    randomID_per_job[ID] = randomID
    
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
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--preprocess', action='store_true', help='Enable preprocessing', default=False)
    parser.add_argument('--normalize', action='store_true', help='Enable preprocessing', default=False)
    parser.add_argument('--stitch', action='store_true', help='Enable preprocessing', default=False)

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
    process_image(args.folder, im_path, ID, model, args.channels, args.y_channel, args.kernel, args.bgrnd_subtraction_vals, args.diameter, args.inf_channels, args.min_size, args.Ncells, args.NZi, start_Y, start_zi, args.xi_per_job, args.yi_per_job, args.Njobs, args.gamma_dict, args.lower_thresh_dict, args.upper_thresh_dict, args.outdir, args.preprocess, args.normalize, args.stitch)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    