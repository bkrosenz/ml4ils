# need X environment, and bio3.6
find $1 -path '*theta*.csv.gz' | grep -v pred | parallel -j$2 python plot_ml_results.py {}
find $1 -name 'theta*' -type d | parallel -j$2 python plot_by_param.py {}
find $1 -name 'theta*' -type d | parallel -j$2 python plot_by_param_classify.py {}
