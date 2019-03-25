from sqlalchemy import Table, Column, Integer, String, MetaData, ForeignKey, Float, Sequence, create_engine
import sqlalchemy as sa
from sqlalchemy.sql.expression import func
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.schema import PrimaryKeyConstraint,UniqueConstraint

import subprocess, sqlite3, argparse, sys, re, shutil
from operator import mul
from math import ceil
from contextlib import contextmanager, closing, ExitStack
from functools import partial,reduce
from io import BytesIO, StringIO
from itertools import chain,product,islice
from glob import glob
from tempfile import mkdtemp
import utils
from os import path
from multiprocess import Pool, Manager
from time import time
from Bio import Phylo, AlignIO
import numpy as np

VMR = lambda x: np.nanvar(x) / np.nanmean(x) # aka index of dispersion
total_length = lambda x: np.sum(x)

def summarize(t):
    covs = tree_config.get_cov(t)
    return {'id':utils.nwstr(t),
            **covs,
            'topology':utils.nwstr(t,format=9),
            'vmr':VMR(list(covs.values())),
            'length':total_length(covs[tip] for tip in utils.tips(t.get_leaf_names()))
    }

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Process some data.')
    parser.add_argument('--outgroup',type=int,
                        default=4,
                        help='output database filename')
    parser.add_argument('--sim',type=str,
                        help='sim model')
    parser.add_argument('--infer',type=str,
                        help='sim model')
    parser.add_argument('--seqtype',type=str,
                        help='dna or amino acid')
    parser.add_argument('--seqlength',type=int,
                        help='simulator (seqgen/indelible')
    parser.add_argument('--theta',type=float,
                        help='scaled mutation rate used in simulation')
    parser.add_argument('--simengine',type=str,
                        help='simulator (seqgen/indelible')
    parser.add_argument('--infengine',type=str,
                        help='inference engine (fasttree/raxml)')
    parser.add_argument('--verbose', action='store_true',
                        help="debug (verbose) mode.")
    parser.add_argument('--overwrite', action='store_true',
                        help="debug (verbose) mode.")
    parser.add_argument('--out','-o',type=str,
                        help='output database filename',required=True)
    parser.add_argument('--indir','-i',type=str,
                        help='input database filename',required=True)

    args = parser.parse_args()

    print(args)

    m = re.compile('t_(.*).rooted.trees')

    args.outgroup = 4
    leafnames = range(1,5)
    tree_config = utils.TreeConfig(leafnames=leafnames,
                                   outgroup=args.outgroup,
                                   subtree_sizes=[4])
    
    dbfile = ''
    engine = create_engine('postgresql://bkrosenz@localhost/sim4') #sqlite:///%s'%args.out)
    metadata = MetaData(bind=engine)
    metadata.reflect(views=True)
    
    cov_labels = next(tree_config.cov_iter(range(1,tree_config.subtree_sizes[0]+1)))
    field_names = cov_labels + ['vmr','length']
    param_names = ['sim_model','sim_engine','infer_model','infer_engine','seq_type','seq_length','theta']
    itree_constraints = ['true','inf']+param_names
    get_fields = lambda: [Column(fn,Float) for fn in field_names] + [Column('topology',String)]

    
    #### initialize tables
    
    if args.overwrite:

        # print('tables',metadata.sorted_tables)
        # for t in metadata.tables.values():
        #     t.drop()#metadata.tables[t]) #(
        # for t in metadata.tables.keys():#reversed(metadata.sorted_tables):
        #     print( t )
        
        try: metadata.drop_all(tables=reversed(metadata.sorted_tables))
        except: print('no tables to drop:',reversed(metadata.sorted_tables))
        
        stree_table = Table('species_trees', metadata,
                            Column('id', String, primary_key=True),
                            *get_fields()
        )
        
        gtree_table = Table('gene_trees', metadata,
                            Column('id', String, primary_key=True),
                            *get_fields(),
        )

        generated_table = Table('generated', metadata,
                                Column('id', Integer, primary_key=True), 
                                Column('sid', None, ForeignKey('species_trees.id')),
                                Column('gid', None, ForeignKey('gene_trees.id')),
                                UniqueConstraint('sid','gid',name='gt_uix') #TODO this assumes Pr(gt1=gt2|st) = 0; only true w/ high precision branch lengths.

        )
        
        inferred_table = Table('inferred', metadata,
                            Column('true', None, ForeignKey('gene_trees.id')),
                            Column('inf', None, ForeignKey('gene_trees.id')),
                            Column('sim_model',String),
                            Column('sim_engine',String),
                            Column('infer_model',String),
                            Column('infer_engine',String),
                            Column('seq_type',String),
                            Column('theta',Float),
                            Column('seq_length',Integer),
                            PrimaryKeyConstraint(*itree_constraints, name='it_uix')
        )
        
        metadata.create_all()
    
    else:
        stree_table = metadata.tables['species_trees']
        gtree_table = metadata.tables['gene_trees']
        inferred_table = metadata.tables['inferred']
        generated_table = metadata.tables['generated']
    
   
    gtree_files = glob(path.join(args.indir,'trees/*.rooted.trees'))
                       
    join_times = [
        [float(f) for f in m.findall(s)[0].split('_')]
        for s in gtree_files
    ]

    trees = (
        tree_config.make_tree( '( ( ( 1:{t1}, 2:{t1} ):{t2_1}, 3:{t2} ):{to_2}, 4:{to});'.format(
            t1 = s[0],
            t2 = s[1],
            to = s[2],
            t2_1 = s[1] - s[0],
            to_2 = s[2] - s[1],
        ) )
        for s in join_times
    )
    st_covs = [ summarize(t) for t in trees ]
    itree_path = path.join(
            args.indir,'inferred_trees/*{sim}_{inf}*.rooted.trees'.format(
                sim = args.sim,
                inf = args.infer
            )
        )

    itree_files = glob( itree_path )

    conn = engine.connect()        
    Session = sessionmaker(bind=engine)
    session = Session()

    #print(st_covs)
    #print ('path',itree_path)
    
    params = dict(
        zip( param_names, (args.sim,
                           args.simengine,
                           args.infer,
                           args.infengine,
                           args.seqtype,
                           args.seqlength,
                           args.theta) )
    ) # TODO: just use namespace -> dict conversion
    
    conn.execute(
        insert(stree_table).values(st_covs).on_conflict_do_nothing(
            constraint = stree_table.primary_key
        )
    )

    #### main loop
    for stree,s,i in zip(st_covs, gtree_files, itree_files):
        with open(i) as f:
            itree_values = [ summarize(tree_config.make_tree(tree)) for tree in f.readlines() if '(' in tree ]
        with open(s) as f:
            gtree_values = [ summarize(tree_config.make_tree(tree)) for tree in f.readlines() if '(' in tree ]
        if not itree_values or not gtree_values: # empty files
            continue
        for statement in [
            insert(gtree_table).values(
                gtree_values
            ).on_conflict_do_nothing(
                constraint = gtree_table.primary_key
            ),
            insert(gtree_table).values(
                itree_values
            ).on_conflict_do_nothing(
                constraint = gtree_table.primary_key
            ),
            insert(inferred_table).values(
                [ { 'true':gtree['id'],
                    'inf':itree['id'],
                    **params}
                  for gtree,itree in zip(gtree_values,itree_values) ]
            ).on_conflict_do_nothing(
                constraint = 'it_uix'
            ),
            insert(generated_table).values(
                [ { 'sid':stree['id'],
                    'gid':gtree['id'] }
                  for gtree in gtree_values ]
            ).on_conflict_do_nothing(
                constraint = 'gt_uix'
            ) ]:
            conn.execute(statement)

    session.close()
