# syntax: bash test_summarize_sql.sh <nloci> <nprocs>
# need to load bio3.6
date;

# if test `find "logfile" -mmin +10`
# then postgres -D /N/dc2/projects/bkrosenz/deep_ils/databases/pgsql/data >logfile 2>&1 &
# fi
nloci=$1 # 500, 1000, etc

python get_summary_stats_sql.py --procs $2 \
       --outdir /N/dc2/projects/bkrosenz/deep_ils/results/ \
       --outfile ms1000aa-theta0.01-n$nloci.hdf5 \
       --ntaxa 4 \
       --splitsize $nloci \
       --overwrite #  --verbose 

date;
