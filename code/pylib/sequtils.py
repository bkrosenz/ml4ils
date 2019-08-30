import numpy as np
from sys import argv
from itertools import *
from Bio import AlignIO
import numba
from itertools import chain, islice

def ichunked(seq, chunksize):
    it = iter(seq)
    while True:
        try:
            yield chain([next(it)], islice(it, chunksize - 1))
        except StopIteration:
            return
                                    
@numba.njit(parallel=True)
def diff(s1,s2):
    """operates on numberic or bytes iterable.  unnormalized."""
    d = 0.
    for a,b in zip(s1,s2):
        d+=a!=b
    return d

@numba.njit(parallel=True)
def ham(s1,s2,gap=-1):
    """operates on numeric or bytes iterable.  normalized. gap must be numeric (ASCII). returns nan if no nongaps"""
    d = 0.
    k = 0
    for a,b in zip(s1,s2):
        if gap==-1 or ( a!=gap and b!=gap) :
            d+=a!=b
            k+=1
    return k and d / k or np.nan

seq2bytes = lambda seq: bytearray(map(ord, seq))

def pwdist(alignment, ignore_gaps = True):
    """calculate pairwise hamming dist on Bio.Align object, or tuple of seqs"""
    d = {}
    for seqpair in combinations(alignment,2):
        if ignore_gaps:
            hamming = ham(*map(seq2bytes,seqpair), gap=ord('-'))
        else:
            hamming = ham(*map(seq2bytes,seqpair))
        key = ','.join(sorted(x.name for x in seqpair))
        d[key] = hamming
    return d 

if __name__=="__main__":
    alignment = AlignIO.read(argv[1],'phylip-relaxed')
    d = pwdist(alignment)
    for k in d:
        print( k, d[k] )
