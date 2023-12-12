#!/bin/bash
#SBATCH --job-name=102523_rox    # Job name
#SBATCH --mem=15G                     # Job memory request
#SBATCH --time=00:30:00               # Time limit hrs:min:sec
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:volta:1
#SBATCH --array 0-50

module unload anaconda
module load anaconda/2022a

cd /home/gridsan/djuna/homer/github/confocalQuant

python main.py \
--folder './outs/neuronbatch102523_cellrox_out/' \
--impath './data/neuronbatch102523_cellrox/' \
--model_type 'cyto2' \
--channels 0 1 2 \
--y_channel 2 \
--kernel 1 \
--bgrnd_subtraction_vals 0 0 0 \
--diameter 15 \
--inf_channels 2 0 \
--min_size 300 \
--Ncells 15300 \
--cells_per_job 300 \
--NZi 2040 \
--zi_per_job 40 \
--xi_per_job 1024 \
--yi_per_job 1024 \
--Njobs 51 \
--gamma_dict '{0: 1, 1: 1, 2: 1}' \
--lower_thresh_dict '{0: 0, 1: 0, 2: 0}' \
--upper_thresh_dict '{0: 100, 1: 100, 2: 100}' \
--outdir './outs/all_outs/segmentations/1025_cellrox/'

