import cProfile, pstats
import dendropy
from io import BytesIO, StringIO

from dendropy.calculate import treecompare
from glob import glob
from sys import argv
from os import path
from utils import enumerate_trees
import re


def read_nw(fn):
    with open(fn) as f:
        nwstr = suffix.sub('', f.read() )
    return nwstr

suffix = re.compile('_\w+')

sp_ind = re.compile('(\d+):\d+\.')

rename = dict(zip('1234','ABCD'))

data_folder = argv[1]

pr = cProfile.Profile()
pr.enable()

sp_tree = dendropy.Tree.get_from_path(path.join(data_folder,'s_tree.trees'),
                                      schema='newick')
# TODO: need to enumerate all topos and count the frequencies

ns = sp_tree.taxon_namespace

# TODO: use TreeList instead 
topo_counts = {
    dendropy.Tree.get_from_string(str(t)+';',
                                  schema='newick',
                                  taxon_namespace = ns,
                                  rooting='force-rooted'
    ) : 0
    for t in enumerate_trees.enum_unordered(ns)
}


gtrees = [read_nw(gt) for gt in glob(path.join(data_folder,'g*.trees'))]

for nwstr in gtrees:
    t1 = dendropy.Tree.get(data = nwstr,
                           schema='newick',
                           taxon_namespace = ns,
                           rooting='force-rooted'
    )
    outgroup_node = t1.find_node_with_taxon_label('D')
    t1.to_outgroup_position(outgroup_node, update_bipartitions=False)
    
#    print(t1)

    for topo in topo_counts:
        if treecompare.symmetric_difference(topo,t1)==0:
            topo_counts[topo]+=1
#break

print('simphy')
for k,v in topo_counts.items():
    print(str(k),v)

ms_counts = {k:0 for k in topo_counts}

with open(path.join(data_folder,'ms.trees')) as f:
    line_no=0
    for line in f:
        if line.startswith('('):
            line_no+=1
                    
            t = dendropy.Tree.get(data = line,
                                  schema='newick',
                                  edge_length_type=float,
                                  rooting='force-rooted'
            )
            for tn in t.taxon_namespace:
                tn.label = rename[tn.label]
            t.migrate_taxon_namespace(ns)
            outgroup_node = t.find_node_with_taxon_label('D')
            t.to_outgroup_position(outgroup_node, update_bipartitions=True)
            for topo in ms_counts:
                if treecompare.symmetric_difference(topo,t,is_bipartitions_updated=True)==0:
                    ms_counts[topo]+=1
#                    print(t,topo)

print('ms')
for k,v in ms_counts.items():
    print(str(k),v)
        
s = StringIO()
ps = pstats.Stats(pr, stream=s)
ps.sort_stats('cumulative').print_stats(20)
#print(s.getvalue())

