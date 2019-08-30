#if test `find "logfile" -mmin +10` ;then postgres -D /N/dc2/projects/bkrosenz/deep_ils/databases/pgsql/data >logfile 2>&1 & fi;
#sleep 10s

# pg_ctl -D /N/dc2/projects/bkrosenz/deep_ils/databases/pgsql/data -l logfile start


#
#psql -d sim4

# python ms2sql2.py --sim LG --infer PROTCATLG --simengine seqgen --infengine raxml --seqtype protein --seqlength 1000 \
#        --out /N/dc2/projects/bkrosenz/deep_ils/databases/pgsql/data \
#        -i /N/dc2/projects/bkrosenz/deep_ils/sims/seqgen-aa-1000bp/ --overwrite

# pg_dump -j4 -Fd -Z9 -f/N/dc2/projects/bkrosenz/ml4ils/databases/sim4.postgres.backup.old sim4
# echo finished dumping db...

for length in 1000 500 200 100; do
    parallel -j4 python ms2sql3.py --sim {1} --infer {2} --simengine seqgen --infengine raxml \
             --seqtype aa --seqlength $length --theta 0.01 \
         -i /N/dc2/projects/bkrosenz/deep_ils/sims/seqgen-aa-theta0.01/${length}bp --overwrite \
         ::: LG WAG ::: PROTCATLG PROTCATWAG;
    echo finished updating db $length...
done

psql -d sim4 -a -f create_views.sql
psql -d sim4 -a -f refresh_views.sql

# dir shouldn't exist yet, but just in case...
backup_dir=/N/dc2/projects/bkrosenz/ml4ils/databases/sim4.postgres.backup.dir.$$
if [ -d $backup_dir ]; then rm -rf $backup_dir; fi
pg_dump -j4 -Fd sim4 -Z9 -f$backup_dir 
