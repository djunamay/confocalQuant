#!/bin/bash
#SBATCH --job-name=srebp    # Job name
#SBATCH --mem=15G                     # Job memory request
#SBATCH --time=00:30:00               # Time limit hrs:min:sec
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:volta:1
#SBATCH --array 0-83

module unload anaconda
module load anaconda/2022a

cd /home/gridsan/djuna/homer/github/confocalQuant

python main.py \
--folder './outs/10252023_SREBP2_matchedcohort_out/' \
--impath './data/10252023_SREBP2_matchedcohort/' \
--model_type 'cyto2' \
--channels 0 1 2 \
--y_channel 0 \
--kernel 1 \
--bgrnd_subtraction_vals 0 0 0 \
--diameter 15 \
--inf_channels 2 0 \
--min_size 300 \
--Ncells 25000 \
--cells_per_job 300 \
--NZi 4000 \
--zi_per_job 40 \
--xi_per_job 1024 \
--yi_per_job 1024 \
--Njobs 84 \
--gamma_dict '{0: 1, 1: 1, 2: 1}' \
--lower_thresh_dict '{0: 0, 1: 0, 2: 0}' \
--upper_thresh_dict '{0: 100, 1: 100, 2: 100}' \
--outdir './outs/all_outs/segmentations/srebp/'