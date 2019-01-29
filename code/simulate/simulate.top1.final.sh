#/usr/bin/bash
#### arguments: <T_a> <Tb-T_a> <T_c-T_b> <data_dir> <raxml_binary_name> <suffix>
### where T_a = T(a,b), T_b = T((a,b),c), T_c = T(((a,b),c),o)


PROG=/N/dc2/projects/bio_hahn_lab/soft/bin
CODEDIR=/N/dc2/projects/bkrosenz/deep_ils/code

nreps=10000 # simulate a lot, then subsample
nsamps=1
seqlen=1000
scale=1

ta=$1 #1.0
ibl=$2
tb=`echo print $ta+$ibl | perl` 
tc=`echo print $tb+$3 | perl` 

DATADIR=$4
raxml_binary=$5
suffix=$6
echo simulating with t_ab=$ta, t_abc=$tb, t_abco=$tc, 
echo using raxml: $raxml_binary

total_samps=$((1+3*$nsamps))
prefix=ta_${ta}_tb_${tb}_tc_${tc}.$suffix

# generate trees with ms
$PROG/ms $total_samps $nreps -T \
   -I 4 $nsamps $nsamps $nsamps 1 \
   -ej $ta 2 1 -ej $tb 3 1 -ej $tc 4 1 \
   > $DATADIR/trees/$prefix.trees

# generate seqs with seq-gen
# discrete gamma rate with 5 categories
$PROG/seq-gen -mHKY \
     -g5 \
     -s$scale -l$seqlen < $DATADIR/trees/$prefix.trees \
    | split -l$((1+$total_samps)) - $DATADIR/seqs/$prefix. \
#             -a3

# infer trees with raxml
# is it ok to use same seed for each gene? raxml requires one to be specified
ls $DATADIR/seqs/$prefix.* \
    | perl -pe "s/.*(${prefix}.*)/\$1/" \
    | xargs -I{} \
            $PROG/$raxml_binary -s $DATADIR/seqs/{} -w $DATADIR/raxml -n {}.raxml \
            -m GTRCAT --HKY85 \
            -p 12345

# concatenate all the trees
cat $DATADIR/raxml/RAxML_bestTree.${prefix}*.raxml > $DATADIR/inferred_trees/$prefix.raxml.trees

# remove raxml tmp files
rm $DATADIR/raxml/RAxML*.${prefix}*.raxml
# remove seq files
rm $DATADIR/seqs/$prefix.*

# python $CODEDIR/utils/distance_matrix.py $DATADIR/trees/$prefix
# python $CODEDIR/utils/distance_matrix.py $DATADIR/inferred_trees/$prefix.raxml.trees

echo done...
#exit $?
