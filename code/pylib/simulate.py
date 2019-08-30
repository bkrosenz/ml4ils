from __future__ import print_function
from __future__ import division

# import argparse, msprime
# import utils as u
# import matplotlib
# # Force matplotlib to not use any Xwindows backend.
# matplotlib.use('Agg')
# import matplotlib.pyplot as pyplot

import numpy as np
from time import time
from sys import argv
from functools import partial
from itertools import product, groupby
from multiprocess import Pool
from contextlib import contextmanager
import dendropy
from dendropy.interop import raxml, seqgen
from dendropy.simulate import treesim
import os

@contextmanager
def poolcontext(*args, **kwargs):
    pool = Pool(*args, **kwargs)
    yield pool
    pool.terminate()
    pool.join()
                
class Tree(dendropy.Tree):
    def set_edge_lengths(self,length):
        for e in self.postorder_edge_iter():
            e.length = length
        return self

    def set_tip_lengths(self,length):
        for e in self.leaf_edge_iter():
            e.length = length
        return self

    def set_internal_lengths(self,length):
        for e in self.preorder_internal_edge_iter(exclude_seed_edge=True):
            e.length = length
        return self

    def extend_tip_lengths(self,length):
        for e in self.leaf_edge_iter():
            e.length += length
        return self

    def extend_internal_lengths(self,length):
        for e in self.preorder_internal_edge_iter(exclude_seed_edge=True):
            e.length += length
        return self

def worker(sp_tree, outfile, num_reps = 1000, seed=None):

    gene_to_species_map = dendropy.TaxonNamespaceMapping.create_contained_taxon_mapping(
            containing_taxon_namespace=sp_tree.taxon_namespace,
            num_contained=1)

    if not os.path.exists(outfile):
        os.makedirs(outfile)

    def label_pairs(tree):
        """returns node objs"""
        ns=sorted(tree.taxon_namespace)
        for i, t1 in enumerate(ns[:-1]):
            for t2 in ns[i+1:]:
                yield t1,t2
        
    def cov_mat(tree):
        """uses namespace of sp_tree"""
        pdc = tree.phylogenetic_distance_matrix()
        try:
            cov = np.array([ tree.mrca(taxa=(t1,t2)).distance_from_root() for t1,t2 in label_pairs(tree) ])
        except:
            print(tree,tree.as_string('newick'),list(label_pairs(tree)))
            raise
        return cov 

    def cov_mat_ultrametric(tree):
        """uses namespace of sp_tree. only works for ultrametric tree"""
        height = tree.max_distance_from_root()
        pdc = tree.phylogenetic_distance_matrix()
#        print(height,np.array([ pdc(t1,t2) for i, t1 in enumerate(tree.taxon_namespace[:-1]) for t2 in tree.taxon_namespace[i+1:] ]))
        return height - np.array([ pdc(t1,t2) for t1,t2 in label_pairs(tree) ])/2
    
    #### generate gene trees
    gene_trees = dendropy.TreeList(
        treesim.contained_coalescent_tree(containing_tree = sp_tree,
                                                    gene_to_containing_taxon_map = gene_to_species_map,
                                                    default_pop_size=0.5) for _ in range(num_reps)
        )
    # is this necessary?
#     for t in gene_trees:
# #        print('gene tree b4',t)
#         t.to_outgroup_position(t.find_node_with_taxon_label('D 1'),
#                                update_bipartitions = True
#         )
#        t.reroot_at_midpoint(update_bipartitions=True) # breaks at polytomies
            # except:
        #     from dendropy.calculate.phylogeneticdistance import PhylogeneticDistanceMatrix
        #     pdm = PhylogeneticDistanceMatrix.from_tree(t)
        #     print (pdm.max_pairwise_distance_taxa())
        #     print('gene tree',t)
        #     raise

    
    ##### generate sequences
    s = seqgen.SeqGen()

#    s.scale_branch_lens = 0.5 # don't need

    #s.char_model = seqgen.SeqGen.GTR
    # s.state_freqs = [0.4, 0.4, 0.1, 0.1]
    # s.general_rates = [0.8, 0.4, 0.4, 0.2, 0.2, 0.1]

    # if char_model not specified, defaults to JC
    data = s.generate(gene_trees)
    
    ##### estimate trees

    tree_ests = []
    rx_args=[] #['-o','T3'] #"--no-bfgs"]
    
    for i,dna_char_mat in enumerate(data.char_matrices):
        rx = raxml.RaxmlRunner() # TODO: rewrite raxml.py in dendropy so _check_overwrite is disabled
        t = rx.estimate_tree(
            char_matrix=dna_char_mat#,raxml_args=rx_args
        )
        # pdc = t.phylogenetic_distance_matrix()
        # newlen = pdc.patristic_distance( * t.taxon_namespace.get_taxa(('D 1','A 1'))) / 2

        # t.reroot_at_edge(outgroup.edge,
        #                  length2 = 10,#outgroup.edge_length - newlen, # can't just do midpoint rooting b/c we KNOW that some of the bl of C is real... should be 10
        #                  length1 = outgroup.edge_length - 2*newlen,
        #                  # assume we know that root is 10 coal units above (A,B,C) clade.  TODO: make variable
        #                  update_bipartitions = True
        #                  #new len of edge to outgroup = length1+length2
        # )
        #       print(t.as_string('newick'))
        #        t.reroot_at_midpoint(update_bipartitions=True)
        outgroup = t.find_node_with_taxon_label('D 1')
        t.to_outgroup_position(outgroup,
                               update_bipartitions = True
        )

        tree_ests.append(t)

    # combs=list(t.phylogenetic_distance_matrix().taxon_iter())
    # for t in tree_ests:
    #     is_same = combs==list(t.phylogenetic_distance_matrix().taxon_iter())
    #     if not is_same:
    #         print('diff order',zip(combs,list(t.phylogenetic_distance_matrix().taxon_iter())))
    #         exit(1)

    #### calculate var-covar matrix = tree_height - patristic_distance(xi,x_j)

    C = np.mean([
        cov_mat(t) for t in tree_ests
    ], axis=0)
    np.savetxt(os.path.join(outfile,'cov.raxml.txt'),C) # assumes all trees will have common namespace order
    
    C_true = np.mean([
        cov_mat(t) for t in gene_trees
    ], axis=0)
    np.savetxt(os.path.join(outfile,'cov.genes.txt'),C_true) # assumes all trees will have common namespace order
    
    C_species = cov_mat(sp_tree)
    np.savetxt(os.path.join(outfile,'cov.species.txt'),C_species) # assumes all trees will have common namespace order

    # sanity check
    for t in tree_ests:
        with open(os.path.join(outfile,'pairs.txt'),'w') as f:
            f.writelines('%s\t%s'%(l1,l2)+'\n' for l1,l2 in label_pairs(t))
            
    gene_trees.write(path = os.path.join(outfile,'trees.true'),
                     schema = 'newick')
            
    with open(os.path.join(outfile,'trees.raxml'),'w') as f:
        f.writelines(t.as_string(schema = 'newick')+'\n' for t in tree_ests)

    return C

    
def main(args):
    start_time = time()
    
    nprocs = int(args[0])
    nreps = int(args[1])
    
    #### initialize sp tree
    sp_tree_str = '[&R] (((A:%f,B:%f):%f,C:%f):%f,D:%f)Root:0.0;' # need outgroup since RAxML imputes unrooted trees
    taxa_labels = list('ABCD')
        
    #### set simulation params ####
    min_bl = 0.5
    step = 0.5
    internal_branches = np.arange(min_bl,10,step)
    external_branches = np.arange(min_bl,5,step)
    obl = 10 #TODO make this variable

    #sptree = sptree.clone(depth=1) # 1 copy per proc shouldnt be necessary, since map passes by value

    #### run sims
    with poolcontext(processes=nprocs) as pool:
        for i,(ibl,ebl) in enumerate( product( internal_branches, external_branches ) ):
            if not i%20:
                print('%d sims complete'%i)
                print('walltime:',time()-start_time)
            outpath="results/sim_i%.3f_e%.3f"%(ibl,ebl)
            obl = 2*(ebl+ibl)
            tree = Tree.get(data=sp_tree_str % (ebl,ebl,ibl,ibl+ebl,obl,obl+ibl+ebl),
                            schema="newick",
                            rooting='force-rooted') # is order of namespace ALWAYS consistent w/order of newick str?
            # tree.to_outgroup_position(tree.find_node_with_taxon_label('D'),
            #                           update_bipartitions = True
            # )
            worker( tree, seed=i*time(), num_reps=nreps, outfile=outpath )
            pool.map_async(partial(worker, seed=i*time(), num_reps=nreps, outfile=outpath), tree)

            
    
if __name__ == "__main__":
    if len(argv)>1:
        main(argv[1:])
    else:
        main(['1','5'])
