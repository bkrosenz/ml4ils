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

echo params $1, $2, $3
echo simulating with t0=$ta, t1=$tb, t_MRCA=$tc

total_samps=$((1+3*$nsamps))
prefix=ta_${ta}_tb_${ibl}_tc_${tc}

# generate trees
$PROG/ms $total_samps $nreps -T \
   -I 4 $nsamps $nsamps $nsamps 1 \
   -ej $tc 2 1 -ej $tb 3 1 -ej $tc 4 1 \
   > $DATADIR/trees/$prefix.tree
