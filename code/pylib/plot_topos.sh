parallel -j4 python plot_topo_freqs.py {} ::: `ls /N/dc2/projects/bkrosenz/deep_ils/results/ms1000aa-theta0.01-*.hdf5`
