# syntax: bash test_learn_config.sh <path_to_dir_containing_config> <num_procs>
# need to load bio3.6
#module load anaconda; source activate bio3.6


# outdir is in /N/project/phyloML/deep_ils/results/ms_learned
echo processing dir: $1...;
cd ../pylib
PYTHONWARNINGS=ignore python train_and_test_new.py --procs $2 \
       --data $1 \
       --predict \
       --classify \
       --folds 10
       #--config $1/config.json \
