DATADIR=$1
find $DATADIR -type f -name '*.trees' |grep rooted -v | parallel -j$2 python $PROJ/deep_ils/code/pylib/root_trees.py {} {.}.rooted.trees 4 overwrite
