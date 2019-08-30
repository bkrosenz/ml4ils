source activate bio
SIMDIR=/N/dc2/projects/bkrosenz/deep_ils/sims
cd /N/dc2/projects/bkrosenz/deep_ils/code
OUTDIR=/N/dc2/projects/bkrosenz/deep_ils/databases

python utils/sims2sql.py \
       --treedir $SIMDIR/short_sims/prot/inferred_trees \
       -w --type protein \
       --model GAMMAWAG \
       --sim WAG \
       --out $OUTDIR/sim_db.sql

python utils/sims2sql.py \
       --treedir $SIMDIR/short_sims/inferred_trees \
       -w --type dna \
       --model GTRCAT HKY85 \
       --sim HKY \
       --out $OUTDIR/sim_db.sql

python utils/sims2sql.py \
       --treedir $SIMDIR/short_sims/trees \
       -w --type true \
       --out $OUTDIR/sim_db.sql
