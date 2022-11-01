import argparse
import gc
import pickle
import re
import time
from functools import partial, reduce
from glob import glob
from itertools import *
from operator import add, iadd
from os import path, remove, stat, walk

import numpy as np
from joblib import Parallel, delayed, dump, load
# from multiprocess import Pool
from pathos.multiprocessing import ProcessingPool as Pool

import parallel_utils3 as par
import utils as u
from sql_utils import (make_session_kw, prepare_heterotachy, prepare_one_rate, psycopg2,
                       select)

schema = 'sim4'
outgroup_name = '4'


def main(args):
    """example arguments:
            leafnames=['1', '2', '3', '4'],
            length=250, ndsets=100
            outdir='/N/dc2/projects/bkrosenz/deep_ils/results/train_data/',
            outfile='test.hdf5', overwrite=True,
            procs=4, ngenes=5000)"""

    outpath = path.join(args.outdir, args.outfile)
    if path.exists(outpath):
        if args.overwrite:
            remove(outpath)
        else:
            raise IOError(
                "file {} exists.  use --overwrite to overwrite".format(outpath)
            )

    tree_config = u.TreeConfig(
        leafnames=args.leafnames,
        outgroup=4,
        include_outgroup=True,
        subtree_sizes=[4]
    )

    labels = u.leaves2labels(args.leafnames, diag=False)

    label_mapper = {
        "pdist": partial(u.triu, labels=labels, diag=False)
    }

    with open('/N/u/bkrosenz/BigRed3/.ssh/db.pwd') as f:
        password = f.read().strip()
    if args.debug:
        print('connecting...')
    session, conn = make_session_kw(
        username='bkrosenz_root',
        password=password,
        database='bkrosenz',
        schema=schema,
        statement_prepare=prepare_one_rate if args.table == 'one_rate' else prepare_heterotachy,
        port=5444,
        host='10.79.161.8',
        with_metadata=False  # sasrdspp02.uits.iu.edu'
    )

    if args.debug:
        print('getting species trees...')
    ngenes = args.ngenes
    n_samps = ngenes * args.ndsets

    species_trees = u.pd.read_sql_query(
        f'execute sids({args.length},{ngenes})',
        con=conn,
        index_col='sid'
    )

    if args.ebl:
        species_trees = species_trees[species_trees.ebl == args.ebl]

    if species_trees.empty:
        print('no species trees found for', args)
        exit(1)

    mapper = {
        "pdist": partial(
            u.triu, labels=labels,
            diag=False
        )
    }

    def summarize(df):
        df2 = par.apply_mapping(df, mapper=mapper)
        return par.summarize_chunk(df2, group_cols=['tid'])

    results = []
    ix = []

    if args.debug:
        print(species_trees, 'making pool...')

    with Pool(args.procs) as pool:
        n = 0
        for stree in species_trees.itertuples():
            if args.debug:
                print('stree', stree)
            try:
                query = f'execute sample_sid({stree.Index},{args.length},{n_samps})'
                x = u.pd.read_sql_query(
                    query, con=conn) if n_samps > 0 else u.pd.DataFrame()
            except psycopg2.Error as e:
                print(e, 'reconnecting...')
                try:
                    session, conn = make_session_kw(
                        username='bkrosenz_root',
                        password=password,
                        database='bkrosenz',
                        schema=schema,
                        statement_prepare=prepare_heterotachy,
                        port=5444,
                        host='10.79.161.8',
                        with_metadata=False  # sasrdspp02.uits.iu.edu'
                    )
                    x = u.pd.read_sql_query(query,
                                            con=conn)
                except:
                    print("couldn't reconnect")
                    continue
            x.fillna(value=np.nan, inplace=True)
            if args.debug:
                print("xquery:", query, 'x', x, '\nnum strees:', n)
            res = pool.map(summarize, u.chunker(x, args.ngenes))
            ix.extend((stree.ebl, stree.ibl, i)
                      for i in range(len(res)))
            results.extend(res)
            if args.debug:
                if not x.empty:
                    n += 1
                if n > 2:
                    break

        if not results:
            print('no records found for', args)
            exit(1)
        if args.debug:
            print(results)

    u.write_results(results, ix, outpath)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="""get summary stats for
    classification/regression"""
    )
    parser.add_argument(
        "--procs",
        "-p",
        type=int,
        help="num procs",
        default=4
    )
    parser.add_argument(
        "--table",
        type=str,
        help="table to query",
        default="heterotachy"
    )
    parser.add_argument(
        "--ndsets",
        "-d",
        type=int,
        help="num dataset",
        default=100
    )
    parser.add_argument(
        "--ngenes",
        "-g",
        type=int,
        help="""num genes in each dataset.  
            All genes must share the same heterotachy multipliers.""",
        default=250
    )
    parser.add_argument(
        "--length",
        "-l",
        type=int,
        help="""length of each gene. 
            If not specified will sample from all lengths in the DB.""",
        default=0
    )
    parser.add_argument(
        "--ebl",
        type=float,
        help="set ebl"
    )
    parser.add_argument(
        "--rfrac",
        type=float,
        help="""fraction of recombinant
                sequences in each dset. 
                If rfrac is not set and nblocks==1, all seqs will be recombinant"""
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite"
    )

    # TODO: make these 2 into a mutually exclusive required group
    parser.add_argument(
        "--outdir",
        help="directory to store results files",
        default="/N/dc2/projects/bkrosenz/deep_ils/results",
    )
    parser.add_argument(
        "--outfile",
        help="output hdf5 name",
        default="covs_trio.hdf5"
    )
    parser.add_argument(
        "--leafnames",
        nargs="+",
        default=(*map(str, range(1, 5)),),
        help="taxa names - list or filename",
    )
    parser.add_argument(
        "--outgroup",
        help="taxa names - list or filename")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="debug")

    args = parser.parse_args()
    print("\n----\nArguments: {}\n".format(args))
    main(args)
    print("finished\n")
