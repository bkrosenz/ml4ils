#/usr/bin/bash
#### arguments: <T_a> <Tb-T_a> <T_c-T_b> <data_dir> <raxml_binary_name> <suffix>
### where T_a = T(a,b), T_b = T((a,b),c), T_c = T(((a,b),c),o)

PROG=/N/dc2/projects/bio_hahn_lab/soft/bin
CODEDIR=/N/dc2/projects/bkrosenz/deep_ils/code

nreps=5000 # simulate a lot, then subsample
nsamps=1
seqlen=1000
scale=1

ta=$1 #1.0
ibl=$2
tb=`echo print $ta+$ibl | perl` 
tc=`echo print $tb+$3 | perl` 

DATADIR=$4
raxml_binary=$5
sim_model=$6
infer_model=$7
echo simulating with t_ab=$ta, t_abc=$tb, t_abco=$tc, 
echo using raxml: $raxml_binary

total_samps=$((1+3*$nsamps))

prefix=t_${ta}_${tb}_${tc}

# generate trees with ms

if [ ! -f $DATADIR/trees/$prefix.trees ] ; then
$PROG/ms $total_samps $nreps -T \
         -I 4 $nsamps $nsamps $nsamps 1 \
         -ej $ta 2 1 -ej $tb 3 1 -ej $tc 4 1 \
         > $DATADIR/trees/$prefix.trees
fi

for sim_model in WAG LG;
do
    simPrefix=${prefix}_${sim_model}
    if [ ! -f $DATADIR/inferred_trees/${simPrefix}_PROTCATWAG.raxml.trees ] ||  [ ! -f $DATADIR/inferred_trees/${simPrefix}_PROTCATLG.raxml.trees ] ; then
        echo simulating for $DATADIR/inferred_trees/${simPrefix}_PROTCATWAG.raxml.trees;
        # generate seqs with seq-gen
        # discrete gamma rate with 5 categories
        $PROG/seq-gen -m$sim_model \
                      -g5 \
                      -s$scale -l$seqlen < $DATADIR/trees/$prefix.trees \
            | split -l$((1+$total_samps)) - $DATADIR/seqs/${simPrefix}_
        #             -a3

        for infer_model in PROTCATWAG PROTCATLG;
        do
            inferPrefix=${simPrefix}_${infer_model}
            
            # infer trees with raxml
            # is it ok to use same seed for each gene? raxml requires one to be specified
            ls $DATADIR/seqs/${simPrefix}* \
                | perl -pe "s/.*(${simPrefix}.*)/\$1/" \
                | xargs -I{} \
                        $PROG/$raxml_binary \
                        -s $DATADIR/seqs/{} \
                        -w $DATADIR/raxml \
                        -n {}_${infer_model}.raxml \
                        -m $infer_model \
                        -p 12345

            # concatenate all the trees
            cat $DATADIR/raxml/RAxML_bestTree.${simPrefix}*$infer_model.raxml > $DATADIR/inferred_trees/$inferPrefix.raxml.trees
        done;
        # remove raxml tmp files
        rm $DATADIR/raxml/RAxML*.${simPrefix}*.raxml
        # remove seq files
        tar czf  $DATADIR/seqs/$simPrefix.tar.gz $DATADIR/seqs/$simPrefix*
        rm $DATADIR/seqs/$simPrefix*
    fi
    echo done...
done
