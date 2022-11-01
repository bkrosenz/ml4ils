CODEDIR=/N/project/phyloML/deep_ils/code
OUTDIR=/N/project/phyloML/deep_ils/results/bo_torch_class
DATADIR=/N/project/phyloML/deep_ils/results/train_data/nonrec

mkdir -p $OUTDIR
PYTHONWARNINGS=ignore python  $CODEDIR/bo_learn_torch_topo.py --outdir $OUTDIR \
	--trials 2000 \
        --preprocessor $DATADIR/preprocessor.joblib \
	--data_files $DATADIR/*d30*hdf5 $DATADIR/*d40*hdf5 $DATADIR/*d60*hdf5 \
	>  $OUTDIR/bo_learn_torch.log

#-i /N/project/phyloML/deep_ils/results/train_data/*d1000*hdf5 \
