#!/bin/bash
#SBATCH --job-name=ad_dm    # Job name
#SBATCH --mem=50G                     # Job memory request
#SBATCH --time=48:00:00               # Time limit hrs:min:sec
#SBATCH -c 20 
#SBATCH -p xeon-p8
#SBATCH --array 0-99

module unload anaconda
module load anaconda/2022a

cd /home/gridsan/djuna/homer/github/confocalQuant

python main.py --folder './outs/' \
--impath '../image_analysis2/' \
--model_path './models/cyto2' \
--channels 0 1 2 3 \
--bounds '{0: (1,99), 1: (48.89, 99.84), 2: (1, 99), 3: (1, 99)}' \
--filter_dict '{0: 0, 1: 0, 2: 0, 3: 0, 4: 0}' \
--diameter 20 \
--inf_channels 2 0 \
--min_size 1000 \
--Ncells 500 \
--cells_per_job 100 \
--NZi 60 \
--zi_per_job 20 \
--xi_per_job 1024 \
--yi_per_job 1024 \
--Njobs=5

