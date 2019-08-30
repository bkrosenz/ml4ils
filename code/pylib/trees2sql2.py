import pandas as pd
from sqlalchemy import Table, Column, Integer, String, MetaData, ForeignKey, Float, Sequence, create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import insert
import argparse, sys, re
import utils
from os import path,listdir
from multiprocess import Pool
from itertools import *
from Bio import AlignIO

def initialize(q, seq_type, model=None, engine=None):
    """input: a dataframe, str, str, str"""
    global d,quartets,rx
    if model is None or engine is None:
        rx = re.compile('(.*?)-(.*?)-(.*?)')
        d = {'seq_type':seq_type}
    else:
        rx = None
        d = {'infer_model':model, 'seq_type':seq_type, 'infer_engine':engine}
    quartets = [qtup._asdict() for qtup in quartets.itertuples(index=False)]
    

def grouper(iterable, n, fillvalue=None):
    """Collect data into fixed-length chunks or blocks.
    grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"""
    args = [iter(iterable)] * n
    z = zip_longest(*args, fillvalue=fillvalue)
    s = next(z)
    while None not in s:
        yield s
        s = next(z)
    yield (*filter(None,s),)
                
def tree2dict(tree):
    return {}

def summarize(args):
    treefile,seqfile = args
    prefix, model, engine = treefile.split('-')
    d.update({'infer_model':model, 'infer_engine':engine})
    # TODO get seq len & gap content
    records = []
    with open(treefile) as f:
        for line in f:
            tree = utils.Tree(line)
            # turn named tuples into dict minus index
            for q in quartets:
                
                tree_config = utils.TreeConfig(leafnames=q.keys(),
                                               outgroup='o',
                                               subtree_sizes=[4])
                t = tree_config.make_tree(
                    utils.subtree(tree, q.values(), newick=False, rename=utils.invert(q))
                )
                align = AlignIO.read(seqfile, "phylip-relaxed")
                length = align.get_alignment_length() 
                #TODO gaps = #
                sys.stdout.flush()
                covs = tree_config.get_cov(t)
                record =  {
                    **q,
                    'tree':utils.nwstr(t),
                    'seq_length':length,
                    **covs,
                    'topology':utils.nwstr(t,format=9),
                }

                records.append(record)
    df = pd.DataFrame(records)

    df['vmr'] = utils.VMR(df[[*covs.keys(),]])
    df['length'] = utils.total_length(df[utils.tips(q.keys())])
    for k,v in d.items():
        df[k] = v
    # TODO: find out most efficient data structure
    return df.to_dict('records')

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Process some data.')
    #TODO parallelize
    parser.add_argument('--quartets',type=str,
                        help="""path to file containing quartet csv. 
                        Must have column header w/ column 'outgroup'""")
    parser.add_argument('--buffsize',type=int, default = 500,
                        help="""size of shared memory buffer.""")
    parser.add_argument('--csize',type=int, default = 50,
                        help="""size of chunks to pass to each proc.""")
    parser.add_argument('--procs',type=int,
                        help="""number of subprocesses.""")
    parser.add_argument('--seqtype',type=str,
                        help="""alignment type (DNA or PROT).""")
    parser.add_argument('--model',type=str,
                        help='e.g. WAG, JTT, nucleotide')
    parser.add_argument('--engine',type=str,
                        help='inference engine (fasttree/raxml)')
    parser.add_argument('--verbose', action='store_true',
                        help="debug (verbose) mode.")
    parser.add_argument('--out','-o',type=str,
                        help='output database name',required=True)
    parser.add_argument('--treedir','-i',type=str,
                        help='''input directory of treefiles.  files must be named like:
                        Metazoa_Choano_LG_Subset31.1-fastTree-lg''',required=True)
    parser.add_argument('--seqdir',type=str,
                        help='input phylipfile dir',required=True)
    parser.add_argument('--overwrite', action='store_true',
                        help="overwrite existing tables.")

    args = parser.parse_args()

    print(args)

    m = re.compile('(.*).rooted.trees')
    quartets = utils.pd.read_csv(args.quartets,header=0,index_col=False )
    quartets.rename(columns={'Outgroup':'o','Porifera':'p','Ctenophora':'c','Bilateria':'b'},
                    inplace=True)
    
    dbfile = ''
    engine = create_engine('postgresql://bkrosenz@localhost/'+args.out) #sqlite:///%s'%args.out)
    metadata = MetaData(bind=engine)
    metadata.reflect(views=True)

    gtree_table = metadata.tables['gene_trees']
    
    conn = engine.connect()        
    Session = sessionmaker(bind=engine)
    session = Session()

    if args.overwrite:
        for t in reversed(metadata.sorted_tables):
            session.execute('''TRUNCATE TABLE {}'''.format(t.name))
        session.commit()

    prefs = listdir(args.treedir)
    seqfiles = (path.join(args.seqdir,s.split('-')[0]+'.phylip') for s in prefs)
    treefiles = (path.join(args.treedir,s) for s in prefs)

    with Pool( args.procs,
               initializer = initialize(quartets, args.seqtype, args.model, args.engine)
    ) as p:
        for chunk in grouper(
                chain.from_iterable( p.imap_unordered(summarize,
                                        zip(treefiles, seqfiles),
                                        chunksize=args.csize) ),
                args.buffsize):
            try:
                conn.execute(
                    insert(gtree_table).values(
                        chunk
                    )
                )
            except Exception as e:
                print(len(chunk),chunk[0],e)
            
