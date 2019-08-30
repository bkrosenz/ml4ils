BASEDIR=/N/dc2/projects/bkrosenz/deep_ils;
for dir in short_sims/inferred_trees short_sims/trees top1/inferred_trees top2/inferred_trees ils1/inferred_trees ils2/inferred_trees top1/trees top2/trees ils1/trees ils2/trees;
do
    echo parsing $dir...;
    python $BASEDIR/code/utils/distance_matrix.py $BASEDIR/sims/$dir/;
    python $BASEDIR/code/utils/summarize_sims.py $BASEDIR/sims/$dir/;
   done
