#!/bin/bash
#SBATCH --job-name=ad_dm    # Job name
#SBATCH --mem=50G                     # Job memory request
#SBATCH --time=48:00:00               # Time limit hrs:min:sec
#SBATCH -c 20 
#SBATCH -p xeon-p8
#SBATCH --array 0-99

module load anaconda/2022a

directory='./xgb_runs/alpha'
directory+=$ALPHA
directory+='_'
directory+=$NMODELS
directory+='_'

if [ $RANDOMIZE == "True" ]
then 
    directory+='random/'
elif [ $RANDOMIZE != "True" ]
then 
    directory+='fixed/'
fi

if [ ! -d "$directory" ]
then
    echo "making new directory"
    mkdir $directory
else echo "directory exists"
fi

cd /data1/groups/TsaiMadry/AD_subtypes
python train_xgb.py --output_dir=$directory --alpha=$ALPHA --randomize_alpha=$RANDOMIZE --models-per-job=500 --N=$NMODELS

python main.py --folder=
--im_path=
--model_path=
--channels=
--bounds=
--filter_dict=
--diameter=
--inf_channels=
--min_size=
--Ncells=
--cells_per_job=
--NZi=
--zi_per_job=
--xi_per_job=
--yi_per_job=
--Njobs=