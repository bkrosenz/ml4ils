from copy import deepcopy
from sqlalchemy import Table, Column, Integer, String, MetaData, ForeignKey, Float, Sequence, create_engine
from sqlalchemy import exc
import sqlalchemy as sa
from sqlalchemy.sql.expression import func
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.schema import PrimaryKeyConstraint,UniqueConstraint
import subprocess, sqlite3, argparse, sys, re, shutil
from operator import mul
from math import ceil
#from contextlib import contextmanager, closing, ExitStack
from functools import partial,reduce
from io import BytesIO, StringIO
from itertools import chain,product,islice
from glob import glob
from tempfile import mkdtemp
import utils
from os import path
from multiprocess import Pool, Manager
from itertools import *
from time import time
from multiprocessing import Queue, Manager, Pool
from time import time
#from Bio import Phylo, AlignIO
import numpy as np


def initialize(q, seqtype, model=None, engine=None, phylipdir=None):
    global d,quartets,rx,pdir
    pdir = phylipdir
    if model is None or engine is None:
        rx = re.compile('(.*?)-(.*?)-(.*?)')
        d = {'seq_type':seq_type}
    else:
        rx = None
        d = {'infer_model':model, 'seq_type':seq_type, 'infer_engine':engine}
    quartets = q
    

def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)
                
def tree2dict(tree):
    return {}

def summarize(fn):
    records = []
    if rx is not None:
        prefix, model, engine = rx.findall(fn)
        d.update({'infer_model':model, 'infer_engine':engine})
    if pdir is not None:
        path.join(pdir,prefix,'.phylip')
        # TODO get seq len & gap content
	with open(fn) as f:
        for q in quartets.itertuples():
            tree_config = utils.TreeConfig(leafnames=q[1:],
                                           outgroup=q.Outgroup,
                                           subtree_sizes=[4])
            for t in map(tree_config.make_tree,f):
                length = #TODO
                covs = tree_config.get_cov(t)
                record = deepcopy(d).update( {
                    'vmr':utils.VMR(),
                    'tree':utils.nwstr(t),
                    'seq_length':length,
                    **covs,
                    'topology':utils.nwstr(t,format=9),
                    'vmr':utils.VMR(list(covs.values())),
                    'length':utils.total_length(
                        covs[tip] for tip in utils.tips(t.get_leaf_names())
                    )
                } )
                records.append(record)
    return records

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Process some data.')
    #TODO parallelize
    parser.add_argument('--quartets',type=str,
                        help="""path to file containing quartet csv. 
                        Must have column header w/ column 'outgroup'""")
    parser.add_argument('--buffsize',type=int,
                        help="""size of shared memory buffer.""")
    parser.add_argument('--procs',type=int,
                        help="""number of subprocesses.""")
    parser.add_argument('--seqtype',type=int,
                        help="""alignment type (DNA or PROT).""")
    parser.add_argument('--model',type=str,
                        help='e.g. WAG, JTT, nucleotide')
    parser.add_argument('--engine',type=str,
                        help='inference engine (fasttree/raxml)')
    parser.add_argument('--verbose', action='store_true',
                        help="debug (verbose) mode.")
    parser.add_argument('--overwrite', action='store_true',
                        help="overwrite existing tables.  WARNING: doesn't work yet.")
    parser.add_argument('--out','-o',type=str,
                        help='output database filename',required=True)
    parser.add_argument('--indir','-i',type=str,
                        help='input directory of treefiles',required=True)

    args = parser.parse_args()

    print(args)

    m = re.compile('(.*).rooted.trees')
    quartets = utils.pd.read_csv(args.quartets,header=0,index_col=False )
    leafnames = quartets.columns
    
    dbfile = ''
    engine = create_engine('postgresql://bkrosenz@localhost/metazoa') #sqlite:///%s'%args.out)
    metadata = MetaData(bind=engine)
    metadata.reflect(views=True)

    gtree_table = metadata.tables['gene_trees']
    
    conn = engine.connect()        
    Session = sessionmaker(bind=engine)
    session = Session()

    filenames = os.listdir(args.indir)
    
    if not args.model

    # with Manager() as manager:
    #     sl = manager.list()
    #     for q in quartets:
    #         p.apply_async(summarize,(q,))
            
    with Pool( args.procs,
               initializer = initialize(quartets, args.model, args.seqtype, args.engine)) as p:
        for chunk in grouper(
            chain(p.imap_unordered(summarize, filenames, chunksize=csize)),
                args.buffsize):
            conn.execute(
                insert(gtree_table).values(
                    chunk
                )
            )
            
