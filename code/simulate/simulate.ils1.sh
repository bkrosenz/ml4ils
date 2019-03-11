#/usr/bin/bash
#### arguments:
### T(a,b) T((a,b),c) T(((a,b),c),o) scale raxml_binary_name

#source ~/.bash_profile
PROG=/N/dc2/projects/bio_hahn_lab/soft/bin
nreps=2000
DATADIR=/N/dc2/projects/bkrosenz/deep_ils/sims/ils1
CODEDIR=/N/dc2/projects/bkrosenz/deep_ils/code

nsamps=1
seqlen=1000
scale=$4

# total tree height is 6, seqgen (and raxml tree) scaled by scale

ibl=$2
ta=$1 #1.0
tb=`echo print $ta+$ibl | perl` 
tc=`echo print $tb+$3 | perl` 

echo params $1, $2, $3, $4
echo simulating with t0=$ta, t1=$tb, t_MRCA=$tc, scale=$4
echo using raxml: $5
raxml_binary=$5
suffix=$6

total_samps=$((1+3*$nsamps))
prefix=ta_${ta}_tb_${tb}_tc_${tc}_s$scale.$suffix

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
            
# infer trees with raxml - must remove previous raxml runs
rm $DATADIR/raxml/*${prefix}*.raxml
            
# is it ok to use same seed for each gene? raxml requires one to be specified.  Keep the same, since we're going to delete the seqs generated
ls $DATADIR/seqs/$prefix.* \
    | perl -pe "s/.*(${prefix}.*)/\$1/" \
    | xargs -I{} \
            $PROG/$raxml_binary -s $DATADIR/seqs/{} -w $DATADIR/raxml -n {}.raxml \
            -m GTRCAT --HKY85 \
            -p 12345

# concatenate all the trees
cat $DATADIR/raxml/RAxML_bestTree.${prefix}*.raxml > $DATADIR/inferred_trees/$prefix.raxml.trees

# remove raxml tmp files
rm $DATADIR/raxml/RAxML_bestTree.${prefix}*.raxml
# remove seq files
rm $DATADIR/seqs/$prefix.*

python $CODEDIR/utils/distance_matrix.py $DATADIR/trees/$prefix
python $CODEDIR/utils/distance_matrix.py $DATADIR/inferred_trees/$prefix.raxml

echo done...
#exit $?
