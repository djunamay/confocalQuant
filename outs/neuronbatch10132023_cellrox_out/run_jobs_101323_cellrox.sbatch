#!/bin/bash
#SBATCH --job-name=101323_rox    # Job name
#SBATCH --mem=15G                     # Job memory request
#SBATCH --time=00:30:00               # Time limit hrs:min:sec
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:volta:1
#SBATCH --array 0-19

module unload anaconda
module load anaconda/2022a

cd /home/gridsan/djuna/homer/github/confocalQuant

python main.py \
--folder './outs/neuronbatch10132023_cellrox_out/' \
--impath './data/neuronbatch10132023_cellrox/' \
--model_type 'cyto2' \
--channels 0 2 1 3 \
--y_channel 2 3 \
--kernel 1 \
--bgrnd_subtraction_vals 0 95 0 0 \
--diameter 20 \
--inf_channels 2 0 \
--min_size 500 \
--Ncells 12000 \
--cells_per_job 300 \
--NZi 1600 \
--zi_per_job 40 \
--xi_per_job 1024 \
--yi_per_job 1024 \
--Njobs 20 \
--gamma_dict '{0: 1, 1: 1, 2: 1, 3:1}' \
--lower_thresh_dict '{0: 0, 1: 0, 2: 0, 3:0}' \
--upper_thresh_dict '{0: 100, 1: 99.9, 2: 100, 3:100}'
