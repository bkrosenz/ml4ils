"""separate  PartitionFinder/raxml partitions file"""
from sys import argv
from os import path
fn = argv[1]
fprefix = path.splitext(fn)[0]
with open (fn,'r') as fin, open(fprefix+'.split.phy','w') as fout:
    for line in fin:
        print (line)
        mn, parts = [s.strip() for s in line.split('=')]
        for i,p in enumerate(parts.split(',')):
            fout.write( '%s.%d = %s\n' % (mn,i,p) )
