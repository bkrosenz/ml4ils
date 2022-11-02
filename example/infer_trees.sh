######  Job commands go below this line #####
. deactivate
. activate py38

SEQDIR=../data/Ryan2013_est.genes/
THREADS=4
SUBPROCESSES=6

find $SEQDIR -name '*.nex' | parallel --shuf -j $SUBPROCESSES iqtree -s {} -m LG+F -nt $THREADS --prefix {.}.LG+F
find $SEQDIR -name '*.nex' | parallel --shuf -j $SUBPROCESSES ./compute_scf.sh {.}
 
cd ../code/pylib

python meta_to_hdf.py --csize 200 --threads $THREADS --procs $SUBPROCESSES --seqdir $SEQDIR



