from time import time
from Bio import AlignIO
from sys import stdin,argv
from os import path
from io import TextIOWrapper
import tarfile

l, tfilename, outdir = argv[1:]
print(l,tfilename,outdir)
t1=time()
with tarfile.open(tfilename,'r:gz') as tfile:
    print('opening',tfile)
    for t in tfile:        
        seqfile = tfile.extractfile(t)
        try:
            align = AlignIO.read( TextIOWrapper(seqfile, encoding='utf-8'), "phylip-relaxed" )
            AlignIO.write( align[:,:int(l)],
                           path.join(outdir,path.basename(t.name)),
                           "phylip-relaxed" )
        except Exception as e:
            print(tfilename,t.name,seqfile,'is not readable')
            print(dir(seqfile))
print('finished in',time()-t1)
