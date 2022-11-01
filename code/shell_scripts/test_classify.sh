OUTDIR=/N/project/phyloML/deep_ils/results/bo_small_binary 


CODEDIR=/N/project/phyloML/deep_ils/code

python -m pdb  $CODEDIR/test_config.py         --config $OUTDIR/model.config         --model_dir $OUTDIR         --classify         --data_files /N/project/phyloML/deep_ils/results/test_data/g500_l500_f0.0_20.hdf5         --preprocessor /N/project/phyloML/deep_ils/results/train_data/nonrec/preprocessor.joblib         --outdir $OUTDIR/test/g500_l500_f0.0_20

