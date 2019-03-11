#source activate bio3.6
#
#if test `find "logfile" -mmin +10` ;then postgres -D /N/dc2/projects/bkrosenz/deep_ils/databases/pgsql/data >logfile 2>&1 &;fi;
#
#psql -d sim4

# python ms2sql2.py --sim LG --infer PROTCATLG --simengine seqgen --infengine raxml --seqtype protein --seqlength 1000 \
#        --out /N/dc2/projects/bkrosenz/deep_ils/databases/pgsql/data \
#        -i /N/dc2/projects/bkrosenz/deep_ils/sims/seqgen-aa-1000bp/ --overwrite

parallel -j4 python ms2sql2.py --sim {1} --infer {2} --simengine seqgen --infengine raxml --seqtype protein --seqlength 1000 --theta 0.01 \
       --out /N/dc2/projects/bkrosenz/deep_ils/databases/pgsql/data \
       -i /N/dc2/projects/bkrosenz/deep_ils/sims/seqgen-aa-1000bp-theta0.01/  ::: LG WAG ::: PROTCATLG PROTCATWAG;
