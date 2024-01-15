# Confocal Image Processing using Cellpose

This script processes confocal microscopy images using the Cellpose model. It performs segmentation and analysis on specified channels, applying preprocessing steps if required. The results, including projections, masks, and processed data, are saved in the specified output folder.

## Prerequisites

Before running the script, ensure you have the following dependencies installed:

- Python 3
- [Cellpose](https://github.com/MouseLand/cellpose)
- Other required Python packages (install via `pip install -r requirements.txt`)

## Usage

```bash
python main_script.py --folder path/to/results --impath path/to/image --channels 0 1 2 --y_channel 0 --kernel 3 --bgrnd_subtraction_vals 10 20 30 --diameter 50 --inf_channels 0 1 --min_size 100 --Ncells 500 --cells_per_job 50 --NZi 10 --zi_per_job 2 --xi_per_job 512 --yi_per_job 512 --Njobs 10 --gamma_dict {0: 1.0, 1: 1.2} --lower_thresh_dict {0: 10, 1: 20} --upper_thresh_dict {0: 90, 1: 95} --outdir path/to/output --preprocess --normalize