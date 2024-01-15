#!/bin/bash
#SBATCH --job-name=110123_rox    # Job name
#SBATCH --mem=15G                     # Job memory request
#SBATCH --time=01:00:00               # Time limit hrs:min:sec
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:volta:1
#SBATCH --array 0-23

module unload anaconda
module load anaconda/2022a

cd /home/gridsan/djuna/homer/github/confocalQuant

python main.py \
--folder './outs/neuronbatch11012023_cellrox_bodipy_out/' \
--impath './data/neuronbatch11012023_cellrox_bodipy/' \
--model_type 'cyto2' \
--channels 0 2 1 3 \
--y_channel 2 3 \
--kernel 1 \
--bgrnd_subtraction_vals 0 0 0 0 \
--diameter 15 \
--inf_channels 2 0 \
--min_size 300 \
--Ncells 7200 \
--cells_per_job 300 \
--NZi 960 \
--zi_per_job 40 \
--xi_per_job 1024 \
--yi_per_job 1024 \
--Njobs 24 \
--gamma_dict '{0: 1, 1: 1, 2: 1, 3: 1}' \
--lower_thresh_dict '{0: 0, 1: 0, 2: 0, 3:0}' \
--upper_thresh_dict '{0: 100, 1: 100, 2: 100, 3:100}' \
--outdir './outs/all_outs/segmentations/1101_cellrox_bodipy/' \
--normalize
