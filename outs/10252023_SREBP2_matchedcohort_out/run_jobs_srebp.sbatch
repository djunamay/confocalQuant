#!/bin/bash
#SBATCH --job-name=srebp    # Job name
#SBATCH --mem=15G                     # Job memory request
#SBATCH --time=01:00:00               # Time limit hrs:min:sec
#SBATCH -p xeon-p8
#SBATCH --array 0-83

module unload anaconda
module load anaconda/2022a

cd /home/gridsan/djuna/homer/github/confocalQuant

python main.py --folder './outs/10252023_SREBP2_matchedcohort_out/' \
--impath './data/10252023_SREBP2_matchedcohort/' \
--model_path './models/cyto2' \
--channels 0 1 2 \
--bounds '{0: (51.56, 99.52), 1: (50.32, 99.52), 2: (1, 99)}' \
--filter_dict '{0: 0, 1: 99.96838280643931, 2: 15.733808083293999, 3: 1477.8}' \
--diameter 20 \
--inf_channels 2 0 \
--min_size 1000 \
--Ncells 16800 \
--cells_per_job 200 \
--NZi 3360 \
--zi_per_job 40 \
--xi_per_job 1024 \
--yi_per_job 1024 \
--Njobs=84

