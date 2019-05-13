DATADIR=/N/dc2/projects/bkrosenz/deep_ils/sims/seqgen-aa-1000bp-theta0.01
find $DATADIR -type f -name '*.trees' |grep rooted -v | parallel -j$1 python $PROJ/deep_ils/code/pylib/root_trees.py {} {.}.rooted.trees 4 overwrite
