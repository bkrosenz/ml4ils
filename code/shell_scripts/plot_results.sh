#source activate pycarb
python plot_features.py ../results/short/results.regress.features.csv \
    && python plot_features.py ../results/short/results.classify.features.csv
python plot_ml_results.py ../results/short/results.classify.csv \
    && python plot_ml_results.py ../results/short/results.regress.csv 

