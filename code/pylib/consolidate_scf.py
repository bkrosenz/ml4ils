from sqlalchemy.exc import IntegrityError
from psycopg2.errors import UniqueViolation
from functools import partial
from multiprocess import Pool
from sql_utils import make_session_kw, select
from sys import argv
import gc
import utils
import random
from os import path
from glob import glob
import pandas as pd
from joblib import Parallel, delayed
import argparse
import re
from collections import namedtuple

schema = 'sim4'
outgroup_name = '4'


def make_tree(tstr):
    t = utils.Tree(tstr)
    t.set_outgroup(outgroup_name)
    return t


stree_rx = re.compile(r't_([\d\.]+)_([\d\.]+)_([\d\.]+)')
filename_rx = re.compile(r'(t_\d.*)_(WAG|LG)(_ds\d+)?')
dirname_rx = re.compile('/?(\d+)bp_b(\d+)_g(\d+)')


def fn2nw(s):
    ta, tb, tc = stree_rx.search(s).groups()
    ibl = float(tb)-float(ta)
    outgroup_bl = float(tc)-float(tb)
    tstr = f'(4:{tc},(3:{tb},(1:{ta},2:{ta}):{ibl}):{outgroup_bl});'
    return utils.nwstr(make_tree(tstr))


def value_mapper(d):
    def f(X):
        try:
            return [d[x] for x in X]
        except:
            return d[X]
    return f


Params = namedtuple(
    'Params',
    ('filename',
     'stree',
     'stree_str',
     'smodel',
     'ds')
)


def parse_filename(fn):
    s = path.basename(fn)
    m = filename_rx.search(s)
    try:
        stree_str, smodel, ds = m.groups()
        if ds:
            ds = int(ds[3:]) - 1
    except Exception as e:
        print(fn)
        print(m.groups())
        raise e
    return Params(filename=fn,
                  stree=fn2nw(stree_str),
                  stree_str=stree_str,
                  smodel=smodel,
                  ds=ds)


if __name__ == '__main__':
    nprocs = int(argv[2])
    with open('/N/u/bkrosenz/BigRed3/.ssh/db.pwd') as f:
        password = f.read().strip()
    print(argv)
    session, conn = make_session_kw(username='bkrosenz_root',
                                    password=password,
                                    database='bkrosenz',
                                    schema=schema,
                                    port=5444,
                                    host='10.79.161.8',
                                    with_metadata=False  # sasrdspp02.uits.iu.edu'
                                    )

    top2tid = pd.read_sql_table('topologies',
                                con=conn,
                                schema=schema,
                                columns=['topology', 'tid'],
                                index_col='topology'
                                )['tid'].to_dict()

    RECOMB = False
    dirname = argv[1]
    if 'bp_b' in dirname:
        try:
            match = dirname_rx.search(dirname).groups()
            slength, nblocks, ngenes = map(
                int, match
            )
        except AttributeError as e:
            print(dirname)
            raise(e)
        RECOMB = True
        scf_table = 'rec_scf'
    elif 'hetero' in dirname:
        slength, ngenes = 500, 250
        scf_table = 'heterotachy_scf'
    elif '1_rate' in dirname:
        slength, ngenes = 500, 250
        scf_table = 'one_rate_scf'
    else:
        slength = int(re.search('(\d+)bp', dirname).group(1))
        scf_table = 'nonrec_scf'
    print(slength, scf_table, 'getting filenames')
    dirname = utils.Path(dirname)
    filenames = list(
        filter(utils.is_nonempty,
               (dirname / 'scf').glob('*_tree1.gz'))
    )

    random.shuffle(filenames)
    print('reading species tree')
    nw2sid = pd.read_sql_table('species_trees',
                               con=conn,
                               schema=schema,
                               columns=['sid', 'newick'],
                               index_col='newick'
                               )['sid'].to_dict()

    csize = int(5000/nprocs)
    sim_models = ('LG', 'WAG')

    # TODO check file size, keep recmap in memory
    stree = ds = smodel = recfile = None

    with Pool(nprocs) as p:
        for param in p.map(parse_filename, filenames):
            written = 0
            if param.stree != stree:
                stree = param.stree
                try:
                    sid = nw2sid[stree]
                except KeyError:
                    print('{} not found in species_trees, skipping'.format(stree))
                    continue

            try:
                d = pd.read_csv(param.filename,
                                header=None,
                                sep='\t',
                                usecols=range(1, 5),
                                names=('top_1', 'top_2', 'top_3', 'nsites')
                                )
                d['sid'] = sid

                ds = param.ds
                if RECOMB:
                    rfilename = dirname/'seqs' / \
                        f'{param.stree_str}_{param.smodel}.permutations.npy'
                    try:
                        recmap = utils.np.load(rfilename)
                        _, _, n = recmap.shape

                        d['tree_no'] = recmap[ds, :, :].T.tolist()
                        if (n != len(d)):
                            raise AttributeError
                    except FileNotFoundError:
                        print('no recfile found at ', rfilename)
                    except ValueError:
                        print(param)
                    d['ds_no'] = ds
                else:
                    d['tree_no'] = d.index
                d['sid'] = sid
                d['sim_model'] = param.smodel
                d['theta'] = 0.01
                d['seq_length'] = slength

                try:
                    d.to_sql(scf_table, conn,
                             schema=schema,
                             method='multi',
                             if_exists='append',
                             index=False)
                    written = len(d)
                except (UniqueViolation, IntegrityError) as e:
                    print('Error: most likely from duplicate gene trees.',
                          param,
                          d.columns, sep='\t')
                except Exception as e:
                    print('error writing', e, d)
                    print(param)
                finally:
                    print('wrote {} scf datasets to sql'.format(
                        written))
                    gc.collect()

            except OSError as e:
                print(e, 'param:', param, sep='\n')
    print('finished updating scf')
