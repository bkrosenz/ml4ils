import argparse
import enum
import gc
import gzip
from pathlib import Path
import random
import re
from functools import partial
from glob import glob
from os import path, stat
import time
from sys import argv

import pandas as pd
from joblib import Parallel, delayed
from multiprocess import Pool
from psycopg2.errors import UniqueViolation

import utils
from sql_utils import make_session_kw, write_rows_to_sql

schema = 'sim4'
outgroup_name = '4'


def make_tree(tstr):
    t = utils.Tree(tstr)
    t.set_outgroup(outgroup_name)
    return t


filename_regex = re.compile(
    't_([\d.]+)_([\d.]+)_([\d.]+)(\.r\d+)?\.trees(\.gz)?')


def fn2nw(s):
    ta, tb, tc, *_ = filename_regex.findall(s)[0]
    # _, ta, tb, tc = path.splitext(path.basename(s))[0].split('_')
    ibl = float(tb)-float(ta)
    outgroup_bl = float(tc)-float(tb)
    tstr = f'(4:{tc},(3:{tb},(1:{ta},2:{ta}):{ibl}):{outgroup_bl});'
    return utils.nwstr(make_tree(tstr))


def summarize(tstr, stree=False):
    t = make_tree(tstr)
    d = {
        "newick": utils.nwstr(t),
        "pdist": utils.get_pdist(t).tolist(),
        "tid": top2tid[utils.nwstr(t, format=9)]
    }
    if stree:
        d['ebl'] = utils.get_ebl(t)
        d['ibl'] = utils.get_ibl(t)
    return d


def file_to_df(fn, sid):
    trees = file_to_tuples((fn, sid))
    return pd.DataFrame(trees)


def file_to_tuples(args: tuple):
    fn, sid = args
    with utils.TreeFile(fn) as f:
        trees = [process_line(x, sid) for x in enumerate(f)]
    print(f'finished {len(trees)} trees')
    return trees


def process_line(args: tuple, sid):
    i, line = args
    tree_stats = summarize(line)
    tree_stats['sid'] = sid
    tree_stats['tree_no'] = i
    return tree_stats


with open('/N/u/bkrosenz/BigRed3/.ssh/db.pwd') as f:
    password = f.read().strip()

if __name__ == '__main__':
    treedir = Path(argv[1])
    nprocs = int(argv[2])
    table_name = argv[3] if len(argv) > 3 else 'gene_trees'

    session, metadata, conn = make_session_kw(username='bkrosenz_root',
                                              password=password,
                                              database='bkrosenz',
                                              schema=schema,
                                              port=5444,
                                              host='10.79.161.8'  # sasrdspp02.uits.iu.edu'
                                              )

    top2tid = pd.read_sql_table('topologies',
                                con=conn,
                                schema=schema,
                                columns=['topology', 'tid'],
                                index_col='topology'
                                )['tid'].to_dict()

    filenames = list(treedir.glob('*trees.gz'))
    random.shuffle(filenames)
    strees = [fn2nw(fn.stem) for fn in filenames]

    d = pd.DataFrame(
        summarize(nw, stree=True) for nw in strees
    )
    # d = (pd
    #      .read_sql_table('species_trees', conn, schema=schema)
    #      .merge(d, on=['newick', 'tid', 'ebl', 'ibl'], how='outer')
    #      )
    # d['pdist'] = d.pdist_x.fillna(d.pdist_y)
    # d.drop(columns=['pdist_x', 'pdist_y'], inplace=True)
    written = write_rows_to_sql(d, conn, schema, n=1, table='species_trees')
    # try:
    #     d.to_sql('species_trees',
    #              conn,
    #              schema=schema,
    #              method='multi',
    #              if_exists='append',
    #              chunksize=25,
    #              index=False)
    print(
        f'wrote {written} species trees to sql'
    )
    # except Exception as e:
    #     print('error writing species trees', e)

    nw2id = pd.read_sql_table('species_trees',
                              con=conn,
                              schema=schema,
                              columns=['sid', 'newick'],
                              index_col='newick')['sid']
    nw2id.to_csv(treedir/'sid.csv.gz',
                 index=True, header=True)
    nw2id = nw2id.to_dict()

    sids = (nw2id[nw] for nw in strees if nw in nw2id)

    csize = int(5000/nprocs)
    MONTHS = 4
    min_date = time.time()-3600*24*30*MONTHS

    print('processing gts')
    with Pool(nprocs) as p:
        for fn, sid in zip(filenames, sids):
            if fn.stat().st_mtime < min_date:
                continue
            print(fn, sid)
            processor = partial(process_line, sid=sid)

            if fn.suffix == '.gz':
                open = gzip.open
            try:
                with open(fn, 'rt') as f:
                    c = p.map(processor,
                              enumerate(filter(utils.is_newick, f)),
                              chunksize=csize)
                    d = pd.DataFrame(c)
            except:
                print('error reading file:', fn)
            print(d.columns)
            try:
                written = write_rows_to_sql(d, conn, schema, table=table_name)
                # d.to_sql('gene_trees', conn,
                #          schema=schema,
                #          method='multi',
                #          if_exists='append',
                #          index=False)
                print(
                    f'wrote {written} gene tree datasets to sql table {table_name}')
            except Exception as e:
                print('error writing gene trees from file:', fn)
                print(e)
            gc.collect()
    print('finished updating gene trees')

    # \copy (select * from sim4.gene_tree_counts) to '/N/project/phyloML/deep_ils/results/train_data/gene_tree_counts.csv' csv header
