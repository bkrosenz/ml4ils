CODEDIR=/N/project/phyloML/deep_ils/code
OUTDIR=/N/project/phyloML/deep_ils/results/final_trained
mkdir $OUTDIR
python  $CODEDIR/test_config.py --outdir $OUTDIR \
	--config $OUTDIR/final_model.config \
	--data_files /N/project/phyloML/deep_ils/results/train_data/recomb/*hdf5  /N/project/phyloML/deep_ils/results/train_data/all*hdf5 >  $OUTDIR/learn_torch.log

	#--overwrite_trainfile \
#-i /N/project/phyloML/deep_ils/results/train_data/*d1000*hdf5 \
