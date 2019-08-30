from scipy.spatial import distance_matrix
import bitstring
from sys import argv
import re

def myreadlines(f, newline):
  buf = ""
  while True:
    while newline in buf:
      pos = buf.index(newline)
      yield buf[:pos]
      buf = buf[pos + len(newline):]
    chunk = f.read(4096)
    if not chunk:
      yield buf
      break
    buf += chunk

sep = '//'
with open(argv[1]) as f:
    for sample in myreadlines(f,sep):
        if not sample: continue
        z = [bitstring.ConstBitArray('0b'+line.strip()) for line in sample.strip().replace(sep,'').split('\n')]
        print distance_matrix(z,z,p=1) #/ float(len(z[0]))
