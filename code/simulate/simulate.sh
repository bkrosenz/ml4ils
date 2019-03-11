#/usr/bin/bash

#source ~/.bash_profile
PROG=/N/dc2/projects/bio_hahn_lab/soft/bin
nreps=2000
DATADIR=/N/dc2/projects/bkrosenz/deep_ils/sims

nsamps=1
seqlen=1000
scale=$4

# total tree height is 6, seqgen (and raxml tree) scaled by scale
tc=$3 #4.0
tb=$2
ta=$1 #1.0
ibl=`echo print $tb-$ta | perl` #2.0

echo params $1, $2, $3, $4
echo simulating with t0=$ta, t1=$tb, t_MRCA=$tc, scale=$4

total_samps=$((1+3*$nsamps))
prefix=ta_${ta}_tb_${ibl}_tc_${tc}_s$scale

# # generate trees
$PROG/ms $total_samps $nreps -T \
   -I 4 $nsamps $nsamps $nsamps 1 \
   -ej $tc 2 1 -ej $tb 3 1 -ej $tc 4 1 \
   > $DATADIR/trees/$prefix.tree

# generate seqs
# discrete gamma rate with 5 categories
$PROG/seq-gen -mHKY \
     -g5 \
     -s$scale -l$seqlen < $DATADIR/trees/$prefix.tree \
     | split -l$((1+$total_samps)) - $DATADIR/seqs/$prefix. --additional-suffix=.seq

# infer trees
# is it ok to use same seed for each gene? raxml requires one to be specified
ls $DATADIR/seqs/$prefix.*.seq \
    | perl -pe "s/.*(${prefix}.*).seq/\$1/" \
    | xargs -I{} \
            $PROG/raxmlHPC -s $DATADIR/seqs/{}.seq -w $DATADIR/raxml -n {}.raxml \
            -m GTRCAT --HKY85 \
            -p 12345

# concatenate all the trees
cat $DATADIR/raxml/RAxML_bestTree.${prefix}*.raxml > $DATADIR/inferred_trees/$prefix.raxml

# remove raxml tmp files
rm $DATADIR/raxml/RAxML_bestTree.${prefix}*.raxml
# remove seq files
rm $DATADIR/seqs/$prefix.*


echo done...
exit $?
