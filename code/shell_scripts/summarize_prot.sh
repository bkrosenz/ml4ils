
BASEDIR=/N/dc2/projects/bkrosenz/deep_ils;

loci=2000;

for dir in prot/inferred_trees trees;
do
    echo parsing $dir...;
    python $BASEDIR/code/utils/distance_matrix.py --treedir $BASEDIR/sims/short_sims/$dir/ \
           --outdir $BASEDIR/sims/short_sims/$dir \
           --jobs 4 \
           --limit $loci;
    python $BASEDIR/code/utils/summarize_sims.py $BASEDIR/sims/short_sims/$dir;
   done
