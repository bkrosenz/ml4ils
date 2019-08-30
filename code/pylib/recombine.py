from time import time
from Bio import AlignIO
from sys import stdin,argv
from os import path
from io import TextIOWrapper
import tarfile
import numpy as np
from sequtils import ichunked

nblocks, tfilename, outdir = argv[1:]
nblocks = int(nblocks)

print(l,tfilename,outdir)

t1=time()
with tarfile.open(tfilename,'r:gz') as tfile:
    print('opening',tfile)
    seqs = []
    for t in tfile:        
        seqfile = tfile.extractfile(t)
        try:
            align = AlignIO.read( TextIOWrapper(seqfile, encoding='utf-8'), "phylip-relaxed" )
            seqs.append(align)
        except Exception as e:
            print(tfilename,t.name,seqfile,'is not readable')
            print(dir(seqfile))

L = seqs[0].get_alignment_length()
step = L//nblocks
permutation_matrix = [np.random.permutation(len(seqs)) for _ in range(nblocks)]
slices = [slice(i,min(i+step,L)) for i in range(0,L+step,step) if i<L]
 
recomb = [ reduce( add,
                   (seqs[s][:,slices[i]] for i,s in enumerate(seq_inds)) ) \
           for seq_inds in zip(*map(iter,permutation_matrix),)
]


# WARNING: this reuses names from the input files
for s in recomb:
    AlignIO.write( align[:,:int(l)],
                   path.join(outdir,path.basename(t.name)),
                   "phylip-relaxed" )
       
print('finished in',time()-t1)
