#! /bin/bash

fn=$1

iqtree -s  ${fn}.nex -nt 8 \
    --scf 10000   -t ${fn}.LG+F.treefile  \
    --cf-quartet --prefix $fn.tmp

echo 'ID      QuartID Seq1    Seq2    Seq3    Seq4    qCF     qCF_N   qDF1    qDF1_N  qDF2    qDF2_N  qN' > $fn.scf 
cut  -f2 --complement $fn.tmp.cf.quartet \
    | grep -v '^#' | sort | uniq | head -n-1 >> $fn.scf \
    && rm $fn.tmp.*
