
BASEDIR=/N/dc2/projects/bkrosenz/deep_ils;

loci=2000;

for dir in short_sims/inferred_trees short_sims/trees;
do
    echo parsing $dir...;
    python $BASEDIR/code/utils/distance_matrix.py --treedir $BASEDIR/sims/$dir/ \
           --outdir $BASEDIR/sims/$dir \
           --jobs 4 \
           --limit $loci;
    python $BASEDIR/code/utils/summarize_sims.py $BASEDIR/sims/$dir;
   done
