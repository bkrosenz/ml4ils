# find . -type f -name '*trees' |grep rooted -v | parallel -j4 python $PROJ/deep_ils/code/pylib/root_trees.py {} {.}.rooted.trees 4
from ete3 import Tree
from sys import argv
from os import path
# <in> <out> <outgroup>

# does not overwrite
if path.exists(argv[2]):
    exit

with open(argv[1],'r') as fin, open(argv[2],'w',buffering=1000) as fout:
    for line in fin:
        try:
            t=Tree(line)
            t.set_outgroup(argv[3])
            fout.write(t.write()+'\n')
        except:
            fout.write(line)
        
                           
