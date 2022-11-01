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
from unittest import result

import numpy as np
from joblib import Parallel, delayed, dump, load
# from multiprocess import Pool
from pathos.multiprocessing import ProcessingPool as Pool

import parallel_utils3 as par
import utils as u
from sql_utils import (make_session_kw, prepare, prepare_all_lengths, psycopg2,
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

    outpath = args.outdir/args.outfile
    if outpath.exists():
        if args.overwrite:
            outpath.unlink(missing_ok=True)
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

    with open('/N/u/bkrosenz/BigRed3/.ssh/db.pwd') as f:
        password = f.read().strip()
    if args.debug:
        print('connecting...')
    session, conn = make_session_kw(
        username='bkrosenz_root',
        password=password,
        database='bkrosenz',
        schema=schema,
        statement_prepare=prepare if args.length else prepare_all_lengths,
        port=5444,
        host='10.79.161.8',
        with_metadata=False  # sasrdspp02.uits.iu.edu'
    )

    if args.debug:
        print('getting species trees...')

    # nblocks requires rfrac to be set
    if args.nblocks > 1 and not args.rfrac:
        args.rfrac = 1

    if args.rfrac:
        ngenes_r = int(args.ngenes * args.rfrac)
        ngenes = args.ngenes - ngenes_r
        n_samps_r = ngenes_r*args.ndsets
    else:
        ngenes = args.ngenes
    n_samps = ngenes * args.ndsets

    species_trees = u.pd.read_sql_query(
        f'execute sids_nonrec({args.length},{ngenes})',
        con=conn,
        index_col='sid'
    )

    if args.rfrac:
        species_trees_rec = u.pd.read_sql_query(
            f'execute sids_rec({args.length},{ngenes_r})',
            con=conn,
            index_col='sid'
        ).drop(species_trees.columns, axis=1)
        species_trees = species_trees.join(species_trees_rec, how='inner')

    if args.ebl:
        species_trees = species_trees[species_trees.ebl == args.ebl]

    if species_trees.empty:
        print('no species trees found for', args)
        exit(1)

    # xcols = ['pdist', 'tid', 'top_1', 'top_2', 'top_3', 'nsites']

    mapper = u.pdist_mapper(args.leafnames)

    def summarize(df):
        df = par.apply_mapping(df, mapper=mapper)
        return par.summarize_chunk(df, group_cols=['tid'])

    results = []
    ix = []

    if args.debug:
        print(species_trees, 'making pool...')

    with Pool(args.procs) as pool:
        if args.train:
            while len(results) < args.ndsets:
                query = 'execute sample_nonrec({})'.format(0.075)
                x = (u.pd.read_sql_query(
                    query, con=conn)
                    .set_index(['ebl', 'ibl'])
                    .sample(frac=1)
                    .fillna(value=np.nan))
                # TODO make sure chunker respects boundaries between strees
                res = pool.map(
                    summarize, (g for n, g in x.groupby(['ebl', 'ibl'])))
                ix.extend(x.index.to_list())
                results.extend(res)

        else:
            n = 0
            for stree in species_trees.itertuples():
                if args.debug:
                    print('stree', stree)
                try:
                    query = 'execute sample_sid_nonrec({},{},{})'.format(
                        stree.Index,
                        args.length,
                        n_samps)
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
                            statement_prepare=prepare if args.length else prepare_all_lengths,
                            port=5444,
                            host='10.79.161.8',
                            with_metadata=False  # sasrdspp02.uits.iu.edu'
                        )

                        x = u.pd.read_sql_query(query,
                                                con=conn)
                    except:
                        print("couldn't reconnect")
                        continue
                # TODO: make this sample the same frac for each ngenes-size chunk
                if args.rfrac:
                    if args.nblocks > 1:
                        query_recomb = 'execute sample_sid_blocks_rec({},{},{},{})'.format(
                            stree.Index,
                            args.length,
                            args.nblocks,
                            n_samps_r,
                        )
                    else:
                        query_recomb = 'execute sample_sid_rec({},{},{})'.format(
                            stree.Index,
                            args.length,
                            n_samps_r
                        )
                    x = (x
                         .append(
                             u.pd.read_sql_query(query_recomb, con=conn),
                             ignore_index=True)
                         )
            # use sample to randomly reorder
                x = x.sample(frac=1).fillna(value=np.nan)
                if args.debug:
                    print("xquery:", query, 'x', x, '\nnum strees:', n)
                # TODO make sure chunker respects boundaries between strees
                res = pool.map(summarize, u.chunker(x, args.ngenes))
                ix.extend((stree.ebl, stree.ibl, i)
                          for i in range(len(res)))
                results.extend(res)
                if args.debug:
                    if not x.empty:
                        n += 1
                    if n > 2:  # only read a few datasets
                        break

        if not results:
            print('no records found for', args, query,
                  (args.rfrac and query_recomb or ''))
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
        "--ndsets",
        "-d",
        type=int,
        help="num dataset",
        default=100
    )
    parser.add_argument(
        "--train",
        action='store_true',
        help="""ignores most other arguments (besides ndsets)
        and samples ndsets genomes with >=25 genes each from a random species trees"""
    )

    parser.add_argument(
        "--ngenes",
        "-g",
        type=int,
        help="num genes in each dataset",
        default=250
    )
    parser.add_argument(
        "--nblocks", "-b",
        type=int,
        help="""number of recomb blocks.
            nblocks>1 will set rfrac==1.
            If nblocks==1 and rfrac>0, 
            recomb seqs of all lengths will be sampled.""",
        default=1
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
                If rfrac is not set and nblocks==1, 
                all seqs will be recombinant"""
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite"
    )

    # TODO: make these 2 into a mutually exclusive required group
    parser.add_argument(
        "--outdir",
        type=u.Path,
        help="directory to store results files",
        default="/N/dc2/projects/bkrosenz/deep_ils/results",
    )
    parser.add_argument(
        "--outfile",
        type=u.Path,
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
