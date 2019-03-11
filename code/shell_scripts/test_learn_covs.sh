# input: num procs
# need to load bio3.6
module load anaconda; source activate bio3.6


nloci=500
nfolds=5
# outdir is in /N/dc2/projects/bkrosenz/deep_ils/results/ms_learned
python learn_covs.py --procs $2 --balanced --folds $nfolds \
       --outdir $1 \
       --ils 0.9 \
       --data /N/dc2/projects/bkrosenz/deep_ils/results/ms1000aa-n$nloci.hdf5 \
       --config $1/config.json
