
# find . -type f -name '*trees' |grep rooted -v | parallel -j4 python $PROJ/deep_ils/code/pylib/root_trees.py {} {.}.rooted.trees 4
from ete3 import Tree
from sys import argv
from os import path
# <in> <out> <outgroup>

# does not overwrite
#print(argv)
if ( len(argv)<5 or argv[4]!='overwrite' ) and path.exists(argv[2]):
    print('refusing to overwrite', argv[2])
    exit
if path.exists(argv[2]):
    
# 1 MiB buffer
with open(argv[1],'r') as fin, open(argv[2],'w',buffering=1048576) as fout:
    for line in fin:
        try:
            t=Tree(line)
            t.set_outgroup(argv[3])
            fout.write(t.write()+'\n')
        except:
            fout.write(line)
