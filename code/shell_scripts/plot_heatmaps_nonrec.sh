cd /N/project/phyloML/deep_ils/code
OUTDIR=/N/project/phyloML/deep_ils/results


for RESULTDIR in $OUTDIR/bo_2_binary $OUTDIR/bo_small_binary ;
do
    python -W ignore plotting/plot_test_results.py \
        --recdir $RESULTDIR/test/ \
        --outdir $RESULTDIR/test/plots -p 8;
    # find /N/project/phyloML/deep_ils/results/bo_final_small/test \
    find $RESULTDIR/test \
        -mindepth 1 -type d  \
        | parallel -j4 python -W ignore plot_heatmap.py --resultdir {} --classify;
done
for RESULTDIR in  $OUTDIR/bo_final_2 $OUTDIR/bo_final_small ;
do
    python -W ignore plotting/plot_test_results.py \
        --recdir $RESULTDIR/test/ \
        --outdir $RESULTDIR/test/plots -p 8;
    # find /N/project/phyloML/deep_ils/results/bo_final_small/test \
    find $RESULTDIR/test \
        -mindepth 1 -type d  \
        | parallel -j4 python -W ignore plot_heatmap.py --resultdir {};
done


for RESULTDIR in $OUTDIR/bo_2_topology $OUTDIR/bo_small_topology ;
do
    find $RESULTDIR/test \
        -mindepth 1 -type d  \
        | parallel -j4 python -W ignore plot_heatmap.py --resultdir {} --topology;
done