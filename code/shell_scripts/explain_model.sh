CODEDIR=/N/project/phyloML/deep_ils/code
# OUTDIR=/N/project/phyloML/deep_ils/results/bo_final_classify
OUTDIR=$1 #/N/project/phyloML/deep_ils/results/bo_final_small

python  $CODEDIR/explain_model.py --outdir $OUTDIR \
	--data_files /N/project/phyloML/deep_ils/results/train_data/nonrec/allLengths_*nonrec.hdf5  \
	--preprocessor /N/project/phyloML/deep_ils/results/train_data/nonrec/preprocessor.joblib \
	--config $OUTDIR/model.config \
	>  $OUTDIR/explain_torch_$(date +%F).log
