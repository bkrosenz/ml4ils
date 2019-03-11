from itertools import *
from Bio import AlignIO
import numba

@numba.njit(parallel=True)
def diff(s1,s2):
    """operates on numberic or bytes iterable.  unnormalized."""
    d = 0.
    for a,b in zip(s1,s2):
        d+=a!=b
    return d

seq2bytes = lambda seq: bytearray(map(ord, seq))

def pwdist(alignment):
    align_length = alignment.get_alignment_length()
    d = {}
    for seqpair in combinations(alignment,2):
        hamming = diff(*map(seq2bytes,seqpair)) / align_length
        key = ','.join(x.name for x in seqpair)
        d[key] = hamming
    return d 

if __name__=="__main__":
    alignment = AlignIO.read(s,'phylip')
    d = pwdist(alignment)
    for k in d:
        print( k, d[k] )

