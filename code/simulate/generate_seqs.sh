#/usr/bin/bash

#source ~/.bash_profile
PROG=/N/dc2/projects/bio_hahn_lab/soft/bin
DATADIR=/N/dc2/projects/bkrosenz/deep_ils/sims

nreps=100
nsamps=10
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
prefix=ta_${ta}_tb_${ibl}_tc_${tc}

# generate seqs
# discrete gamma rate with 5 categories
$PROG/seq-gen -mHKY \
     -g5 \
     -s$scale -l$seqlen < $DATADIR/trees/$prefix.tree \
     | split -l$((1+$total_samps)) - $DATADIR/seqs/${prefix}_s$scale. --additional-suffix=.seq

# infer trees
# is it ok to use same seed for each gene? raxml requires one to be specified
ls $DATADIR/seqs/${prefix}_s$scale.*.seq \
    | perl -pe "s/.*(${prefix}.*).seq/\$1/" \
    | xargs -I{} \
            $PROG/raxmlHPC -s $DATADIR/seqs/{}.seq -w $DATADIR/raxml -n {}.raxml \
            -m GTRCAT --HKY85 \
            -p 12345

echo done...
exit $?
