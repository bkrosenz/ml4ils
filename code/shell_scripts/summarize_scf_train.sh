d=10000

parallel --lb -j $1 python /N/project/phyloML/deep_ils/code/pylib/get_summary_stats_frac_scf.py \
    -p $2 -d $d \
	--overwrite --train \
    --outfile allLengths_d${d}_rep{1}.hdf5 \
	--outdir /N/project/phyloML/deep_ils/results/train_data/nonrec/ \
    ::: `seq 1 50`
