# syntax: bash test_learn_config.sh <path_to_dir_containing_config> <num_procs>
# need to load bio3.6
#module load anaconda; source activate bio3.6


# outdir is in /N/dc2/projects/bkrosenz/deep_ils/results/ms_learned
echo processing dir: $1...;
PYTHONWARNINGS=ignore python learn_covs3.py --procs $2 \
       --outdir $1 \
       --config $1/config.json \
       --predict \
       --classify
