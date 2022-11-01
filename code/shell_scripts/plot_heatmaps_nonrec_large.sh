cd /N/project/phyloML/deep_ils/code
OUTDIR=/N/project/phyloML/deep_ils/results

RESULTDIR=$OUTDIR/bo_final_2 

python -W ignore plotting/plot_test_results.py \
    --recdir $RESULTDIR/test/ \
    --outdir $RESULTDIR/test/plots -p 8;

find $RESULTDIR/test \
    -mindepth 1 -type d -name 'g500_l500_f0*' \
    | parallel -j4 python -W ignore plot_heatmap.py --resultdir {};

RESULTDIR=$OUTDIR/bo_2_binary 
find $RESULTDIR/test \
    -mindepth 1 -type d -name 'g500_l500_f0*' \
    | parallel -j4 python -W ignore plot_heatmap.py --resultdir {} --classify;
python -W ignore plotting/plot_test_results.py \
    --recdir $RESULTDIR/test/ \
    --outdir $RESULTDIR/test/plots -p 8;


RESULTDIR=$OUTDIR/bo_final_topology 
find $RESULTDIR/test \
    -mindepth 1 -type d  -name 'g500_l500_f0*' \
    | parallel -j4 python -W ignore plot_heatmap.py --resultdir {} --topology;
