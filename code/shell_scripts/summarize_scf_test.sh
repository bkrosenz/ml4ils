
# parallel --shuf -j $1 python /N/project/phyloML/deep_ils/code/pylib/get_summary_stats_model_scf.py -p $2 -d 100 -g {1} -l {2} --outfile g{1}_l{2}_d100.hdf5 --outdir /N/project/phyloML/deep_ils/results/test_data/ ::: 50 100 250 500 1000 ::: 50 100 200 500 1000 


parallel --lb --shuf -j $1 python /N/project/phyloML/deep_ils/code/pylib/get_summary_stats_frac_scf.py \
    -p $2 -d 100 -g {1} -l {2} --outfile g{1}_l{2}_d100.hdf5 \
    --outdir /N/project/phyloML/deep_ils/results/test_data/ \
    ::: 50 100 250 500 1000 ::: 50 100 200 500 1000 

#parallel --shuf -j $1 python /N/project/phyloML/deep_ils/code/pylib/get_summary_stats_frac_scf.py -p $2 -d 100 -g {1} -l {2} --overwrite --outfile g{1}_l{2}_b{3}_d100.hdf5 --outdir /N/project/phyloML/deep_ils/results/test_data/ --nblocks {3} ::: 50 100 250 500 1000 ::: 50 100 200 500 1000 ::: 2 3 4 
#
#parallel -j $1 python /N/project/phyloML/deep_ils/code/pylib/get_summary_stats_frac_scf.py -p $2 -d 100 -g {1} -l {2} --rfrac {3} --overwrite --outfile g{1}_l{2}_f{3}_d100.hdf5 --outdir /N/project/phyloML/deep_ils/results/test_data/ ::: 50 100 250 500 1000 ::: 50 100 200 500 1000 ::: `seq 0.0 0.1 1.0` 

