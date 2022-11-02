CODEDIR=../code
OUTDIR=$1
DATADIR=$2 # DATADIR contains the hdf5 files on which to run DeepSHAP.

python  $CODEDIR/explain_model.py --outdir $OUTDIR \
	--data_files $DATADIR/*.hdf5  \
	--preprocessor ../models/preprocessor.joblib \
	--config $OUTDIR/model.config \
	>  $OUTDIR/explain_torch_$(date +%F).log
