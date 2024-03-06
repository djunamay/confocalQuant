# End-to-end Confocal Image Processing 
### using pre-trained Cellpose models

- This repository provides a number of functions for image processing (loading, processing, and segmentation using inference by pre-trained Cellpose[^1] models), viewing (a number of notebook widgets are implemented to facilitate pre-processing and segmentation assessment), and plotting (normalized - intensities per segmented region by experimental category of interest, quantification, and representative images). 
- `do_inference` function could easily be replaced with other pre-trained models (e.g. [^2], or from other sources)
- The repository also provides a custom droplet / particle analysis 

## Prerequisites

Basic requirements:
- Python 3
- Access to a GPU

Cellpose installation:
- [Cellpose](https://github.com/MouseLand/cellpose)
***install the models of interest**

Other (available by `pip install`):
- [argparse](https://pypi.org/project/argparse/)
- [ast]
- [numpy](https://numpy.org/install/)
- [torch]
- [tqdm] 
- [aicsimageio] 

Install repo:
```bash
git clone
```

Run tests:
```bash
```

## Quickstart

1. The `example_segmentation.ipynb` notebook will help you run a couple of quick experiments on your data to find segmentation parameters that work for you with your selected Cellpose model. 

2. Once you have identified parameters that work for your segmentation purposes, you can update the following line of code and run that in your terminal or update the `run_jobs.sh` file to submit to a job scheduler and process multiple images in parallel. 

```bash
python main_script.py --folder path/to/results --impath path/to/image --channels 0 1 2 --y_channel 0 --kernel 3 --bgrnd_subtraction_vals 10 20 30 --diameter 50 --inf_channels 0 1 --min_size 100 --Ncells 500 --cells_per_job 50 --NZi 10 --zi_per_job 2 --xi_per_job 512 --yi_per_job 512 --Njobs 10 --gamma_dict {0: 1.0, 1: 1.2} --lower_thresh_dict {0: 10, 1: 20} --upper_thresh_dict {0: 90, 1: 95} --outdir path/to/output --preprocess --normalize
```

```
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
```

3. As results are processing, check the `--outdir` folder to monitor segmentation results (projections) as they come in #TODO: update this to be the projection of the segmentations, not projection, then segmentations

4. Finally, go back to the `example_segmentation.ipynb` notebook to view the results, perform some simple sanity checks, and plot signal intensity quantifications for segmented regions.

## Droplet / Particle analysis


## Repository overview

- `example_segmentation.ipynb` notebook guiding user through the pipeline
- `main.py` performs image processing and segmentation with user-defined parameters
- `run_jobs.sh` example file for submission with job scheduler 
- `./models/` save ***cellpose** models here
- `./data/` save raw data here (in ***czi*** file format; example `./data/experiment_1`)
- `./outs/` outputs of `main_script.py` run will be saved here (including `masks.npy`, `mat.npy`, `Nzi_per_job.npy`, `probs.npy`, `randomID_per_job.npy`; example `./outs/experiment_1_out`)
- `./confocalQuant/` functions called in `main_script.py` for processing and segmentation and in `example_segmentation.ipynb` for viewing and plotting

## Methods
If the `--preprocess` flag is added, the following preprocessing steps are performed
- background subtraction
- median filtering
- gamma correction
- thresholding

Per-segmented region intensities are computed 
        P = temp_probs/np.sum(temp_probs)
        E[M] = np.dot(temp_vals, P)

## References
[^1]: @article{Stringer2020,
  title = {Cellpose: a generalist algorithm for cellular segmentation},
  volume = {18},
  ISSN = {1548-7105},
  url = {http://dx.doi.org/10.1038/s41592-020-01018-x},
  DOI = {10.1038/s41592-020-01018-x},
  number = {1},
  journal = {Nature Methods},
  publisher = {Springer Science and Business Media LLC},
  author = {Stringer,  Carsen and Wang,  Tim and Michaelos,  Michalis and Pachitariu,  Marius},
  year = {2020},
  month = dec,
  pages = {100–106}
}

[^2]: @article{Pachitariu2022,
  title = {Cellpose 2.0: how to train your own model},
  volume = {19},
  ISSN = {1548-7105},
  url = {http://dx.doi.org/10.1038/s41592-022-01663-4},
  DOI = {10.1038/s41592-022-01663-4},
  number = {12},
  journal = {Nature Methods},
  publisher = {Springer Science and Business Media LLC},
  author = {Pachitariu,  Marius and Stringer,  Carsen},
  year = {2022},
  month = nov,
  pages = {1634–1641}
}