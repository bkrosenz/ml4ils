import parallel_utils3 as par
import argparse
import gc
import random
import re
from collections import namedtuple
from functools import partial
from pathlib import Path
from posixpath import join
from sys import argv
from time import time

import dask.dataframe as dd
import pandas as pd
from joblib import Parallel, delayed
from multiprocess import Pool

import utils as u

filename_rx = re.compile(
    r"(.*)_(\d+)\.([WAGL+FTR]+)\.treefile$")

#   qCF: Fraction of concordant sites supporting quartet Seq1,Seq2|Seq3,Seq4 (=qCF_N/qN)
#   qCF_N: Number of concordant sites supporting quartet Seq1,Seq2|Seq3,Seq4
#   qDF1: Fraction of discordant sites supporting quartet Seq1,Seq3|Seq2,Seq4  (=qDF1_N/qN)
#   qDF1_N: Number of discordant sites supporting quartet Seq1,Seq3|Seq2,Seq4
#   qDF2: Fraction of discordant sites supporting quartet Seq1,Seq4|Seq2,Seq3 (=qDF2_N/qN)
#   qDF2_N: Number of discordant sites supporting quartet Seq1,Seq4|Seq2,Seq3
#   qN: Number of decisive sites with four taxa Seq1,Seq2,Seq3,Seq4 (=qCF_N+qDF1_N+qDF2_N)


def value_mapper(d):
    def f(X):
        try:
            return [d[x] for x in X]
        except:
            return d[X]

    return f


Params = namedtuple(
    "Params", ("filename", "prefix", "gene",  "imodel")
)


def parse_filename(fn: Path) -> Params:
    s = fn.name
    m = filename_rx.search(s)
    try:
        prefix, gene, imodel, = m.groups()
    except AttributeError as e:
        print(fn, filename_rx)
        raise e
    return Params(
        filename=fn,
        prefix=prefix,
        gene=gene,
        imodel=imodel
    )


outgroup = set(['Choanoflagellida', 'Choanoflagellatea', 'Ichthyosporea', 'Filasterea',
                'Fungi'])
ingroup = set(['Bilateria', 'Cnidaria', 'Placozoa', ])
outgroup_rx = re.compile('^('+')|('.join(outgroup)+')')
ingroup_rx = re.compile('^('+')|('.join(ingroup)+')')


def clade_name_mapper(s):
    return outgroup_rx.sub('Outgroup', ingroup_rx.sub('ParaHoxozoa', s))
    # return outgroup_rx.sub('Outgroup', s)


# top2tid = {'(4,(1,(2,3)));': 3, '(4,(2,(1,3)));': 2, '(4,(3,(1,2)));': 1}

clade_mapper = {'ParaHoxozoa': '1', 'Ctenophora': '2',
                'Porifera': '3', 'Outgroup': '4'}
# clade_mapper.update({o: '1' for o in ingroup})

# clade_mapper = {'Ctenophora': 'cte', 'Porifera': 'por'}
# clade_mapper.update({o: 'out' for o in outgroup})
# clade_mapper.update({o: 'bil_cni' for o in ingroup})


def tree2pdist(tree: u.Tree, q) -> pd.DataFrame:
    tree = tree.copy('deepcopy')
    tree.prune(q)
    for l in tree.get_leaves():
        l.name = clade_mapper[l.name.split('_')[0]]
    tree.set_outgroup('4')
    d = u.summarize(tree)
    d['ix'] = q

    # for taxon in q:
    #     d[clade_mapper[taxon.split('_')[0]]] = taxon
    # del d['newick']
    return d


def filter_quartet_list(quartet: list) -> bool:
    """Assumes that Outgroups have not been merged, and the list may contain more than 4 taxa"""
    found = 0
    for clades in (outgroup, ingroup, ['Porifera'], ['Ctenophora']):
        found += any(s.find(c) > -1 for s in quartet for c in clades)
    return found >= 4


# clades = {
#     k: str(v) for v, k in enumerate(
#         ('ParaHoxozoa', 'Cten', 'Por', 'Out'))
# }

splits = {
    '2143': {'qDF1': 'qDF1', 'qCF': 'qCF', 'qDF2': 'qDF2', 'qN': 'qN'},
    '1234': {'qDF1': 'qDF1', 'qCF': 'qCF', 'qDF2': 'qDF2', 'qN': 'qN'},
    '2413': {'qCF': 'qDF1', 'qDF1': 'qCF', 'qDF2': 'qDF2', 'qN': 'qN'},
    '4123': {'qCF': 'qDF2', 'qDF1': 'qCF', 'qDF2': 'qDF1', 'qN': 'qN'},
    '2314': {'qCF': 'qDF2', 'qDF1': 'qCF', 'qDF2': 'qDF1', 'qN': 'qN'}
}


class SCF:
    def __init__(self, filepath):
        # TODO : make sure scf has expected labeling
        self.scf = pd.read_csv(
            filepath,
            delim_whitespace=True,
            usecols=['qCF', 'qDF1', 'qDF2', 'qN',
                     'Seq1', 'Seq2', 'Seq3', 'Seq4'],
            index_col=['Seq1', 'Seq2', 'Seq3', 'Seq4'],
        ).dropna()

    def reorder_scf(self):
        rows = []
        for idx, row in self.scf.iterrows():
            topology = ''.join(clade_mapper[t.split('_')[0]] for t in idx)
            row.index = row.index.map(splits[topology])
            rows.append(row)
        self.scf = pd.DataFrame(rows)
        return self

    @staticmethod
    def filter_quartet(quartet: list) -> bool:
        """filters for those quartets that span the ('ParaHoxozoa', 'Out', 'Por', 'Cten') branch"""
        found = 0
        for clade in clade_mapper:
            found += sum(s.startswith(clade) for s in quartet) == 1
        return found == 4

    @staticmethod
    def clade_order(s: str):
        '''TODO: handle NA clade names'''
        for c in clade_mapper:
            if s.startswith(c):
                return clade_mapper[c]
        return '0'

    def map_index(self, ix2name):
        def name_mapper(tup):
            """sort must respect  s1,s2|s3,s4 so we know which topos qCF, DF, and DF2 map to"""
            tup = tuple(ix2name[i] for i in tup)

            def s(t):
                return sorted(t, key=self.clade_order)
            left, right = sorted((s(tup[:2]), s(tup[2:])))
            return (*left, *right)
        self.scf.index = self.scf.index.map(name_mapper)
        return self

    def sort_index(self):
        """This should be called after reorder scf"""
        def s(t):
            return tuple(sorted(t, key=self.clade_order))
        self.scf.index = self.scf.index.map(s)
        self.scf.index.names = [s.split('_')[0] for s in self.scf.index[0]]

    def filter(self):
        scf = self.scf
        scf = scf[scf.index.map(self.filter_quartet)]
        if not scf.empty:
            scf.index.names = [s.split('_')[0] for s in scf.index[0]]
        self.scf = scf
        return self

    def percent(self):
        # convert frac to percent for consistency
        if u.np.allclose(self.scf.drop(columns='qN').sum(1), 1):
            self.scf.loc[:, ['qCF', 'qDF1', 'qDF2']
                         ] = self.scf[['qCF', 'qDF1', 'qDF2']]*100

    @property
    def empty(self):
        return self.scf.empty

    @property
    def index(self):
        return self.scf.index

    def to_dataframe(self):
        return self.scf


def write_quartets(filename: Path, threads: int = 4):
    import numpy as np
    from Bio import AlignIO

    # TODO: must first translate species name -> bil_cni/out/por to reorder index columns of scf.
    # Then select using array_agg.
    _, prefix, gene, imodel = parse_filename(filename)
    dirname = filename.parent
    tree = u.read_trees(filename)[0]
    for l in tree:
        l.name = clade_name_mapper(l.name)
    a = AlignIO.read(dirname/f'{prefix}_{gene}.nex', format='nexus')

    ix2name = {
        ix: clade_name_mapper(seq.name.replace(
            '@', '_')) for ix, seq in enumerate(a, 1)
    }

    scf = (
        SCF(filepath=dirname/f'{prefix}_{gene}.scf')
        .map_index(ix2name)
        .filter()
    )

    if scf.empty:
        # print(filename, 'no quartets found')
        return False
    scf.reorder_scf()
    scf.sort_index()
    try:
        scf.percent()
    except:
        print(scf, filename)
        raise
    # quartet_records = dict.fromkeys(quartet_ids.values())
    summarize_tree = partial(tree2pdist, tree)
    quartet_records = Parallel(threads)(
        delayed(summarize_tree)(q) for q in scf.index)
    d = (pd
         .DataFrame
         .from_records(quartet_records, index='ix')
         )
    d.index = pd.MultiIndex.from_tuples(d.index, names=scf.index.names)
    d = scf.to_dataframe().join(d)

    # mapper = u.pdist_mapper(tree.get_leaf_names())
    # d = par.apply_mapping(d, mapper=mapper)
    # d = par.summarize_chunk(d, group_cols=['tid'])

    d["infer_model"] = imodel
    d["seq_length"] = a.get_alignment_length()
    d["infer_engine"] = "iqtree"
    d["seq_type"] = "AA"
    d["matrix_name"] = filename.parent.name
    d.to_pickle(filename.with_suffix('.quartets.pd.gz'))
    return True


def write_hdf(s: Path, procs=4):
    from summarize_meta import summarize_dataset
    # try:
    #     df = pd.concat(map(pd.read_pickle, s.glob('*pd.gz')))
    # except ValueError:
    #     print(list(s.glob('*pd.gz')))
    #     return
    # df = df[['qCF', 'qDF1', 'qDF2', 'qN', 'pdist', 'tid', 'seq_length']]
    # mapper = u.pdist_mapper(list('1234'))
    # df = par.apply_mapping(df, mapper=mapper)

    # groups = df.groupby(df.index.names)
    # summary_stats = pd.concat(
    #     Parallel(n_jobs=procs, prefer='threads')(
    #         delayed(par.summarize_chunk)(g, ['tid']) for n, g in groups
    #     )
    # )

    # summary_stats.index = groups.groups.keys()
    summary_stats = summarize_dataset(s, by='taxa')
    summary_stats.to_hdf(s.parent/'summary_stats.hdf5', key=s.stem)
    return summary_stats


def main(args):

    csize = int(5000 / args.procs)
    # TODO check file size, keep recmap in memory
    print(write_quartets(Path(
        '/N/project/phyloML/deep_ils/data/metazoa/li2021/genes/Whelan2017_full_only_choanozoa.genes/Subset18_02_126.LG+F.treefile'),
        args.threads))
    directories = list(args.seqdir.glob("*.genes"))
    random.shuffle(directories)
    if args.procs == 1:
        for dirname in directories:
            start_time = time()
            done = 0
            for filename in dirname.glob('*.treefile'):
                # done += filename.with_suffix('.quartets.pd.gz').exists(
                done += write_quartets(filename, threads=args.threads)
            print(
                f'dir: {dirname}\ttime: {time()-start_time}\twrote: {done}')
            write_hdf(dirname)
    else:
        with Pool(args.procs) as p:
            for dirname in directories:
                start_time = time()
                res = p.imap_unordered(
                    partial(write_quartets, threads=args.threads),
                    dirname.glob('*.treefile'))
                print(
                    f'dir: {dirname}\ttime: {time()-start_time}\twrote: {sum(res)}')
            p.imap_unordered(write_hdf, directories)
    print("finished updating inferred gene trees")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process some data.")
    parser.add_argument(
        "--buffsize",
        type=int,
        default=100,
        help="""size of buffer for sql writes."""
    )
    parser.add_argument(
        "--csize",
        type=int,
        default=50,
        help="""size of chunks to pass to each proc."""
    )
    parser.add_argument(
        "--procs",
        type=int,
        default=4,
        help="""course-grained parallelism.""")
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="""fine-grained parallelism.""")
    parser.add_argument(
        "--seqtype",
        type=str,
        default='PROT',
        help="""alignment type (DNA or PROT).""")
    parser.add_argument(
        "--engine",
        type=str,
        default="iqtree",
        help="inference engine (fasttree/raxml). Not implemented.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="debug (verbose) mode.")
    parser.add_argument(
        "--ignore_errors",
        action="store_true",
        help="debug (verbose) mode."
    )
    parser.add_argument(
        "--seqdir",
        type=Path,
        help="input folder containing directories containing trees and scf files",
        required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing tables."
    )

    args = parser.parse_args()

    print(args)
    main(args)