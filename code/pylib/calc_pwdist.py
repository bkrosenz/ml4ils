from Bio import AlignIO
import sequtils
import pandas as pd
from sys import argv
from joblib import Parallel,delayed
NJOBS=4

trios = pd.read_csv(argv[1])

species2clade =  {}
for tup in trios.itertuples(index=False):
    for k,v in tup._asdict().items():
        species2clade[v]=k

def rename(x):
    for a in x:
        a.name = species2clade.get(a.name,a.name)
    return x

align = AlignIO.read(argv[2],'phylip-relaxed')
rename(align)

records = Parallel(n_jobs=NJOBS)(delayed(sequtils.pwdist) \
               ([a for a in align if a.id in s])
               for s in trios.itertuples(index=False, name=None))

d=pd.DataFrame.from_records(records)

print(d.mean())
d.to_csv('pwdists.csv.gz')
