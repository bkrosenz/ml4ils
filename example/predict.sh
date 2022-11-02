##### predict ILS
OUTDIR=./predict_p
MODELDIR=../models/DNN-Pred
THRESHOLD=.9511
python ../code/test_config.py       \
      --config $MODELDIR/model.config  \
          --model_dir $MODELDIR        \
          --classify --threshold $THRESHOLD \
          --data_files ../data/summary_stats.hdf5 \
          --preprocessor ../models/preprocessor.joblib \
          --outdir $OUTDIR

##### predict the species tree topology
OUTDIR=./predict_topology
MODELDIR=../models/DNN-Top
python ../code/test_config.py       \
      --config $MODELDIR/model.config  \
          --model_dir $MODELDIR        \
          --topology \
          --data_files ../data/summary_stats.hdf5 \
          --preprocessor ../models/preprocessor.joblib \
          --outdir $OUTDIR