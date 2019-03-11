# need X environment, and bio3.6
find /N/dc2/projects/bkrosenz/deep_ils/results/ms_learned/ -path '*theta*.csv.gz' | grep -v pred | parallel -j$1 python plot_ml_results.py {}
find /N/dc2/projects/bkrosenz/deep_ils/results/ms_learned -name 'theta*' -type d | parallel -j$1 python plot_by_param.py {}
