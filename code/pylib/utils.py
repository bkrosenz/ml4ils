from scipy.special import comb as nchoosek
from ete3 import Tree
import pandas as pd
from numpy.lib.function_base import iterable
import numpy as np
import numba
from typing import Callable, List, Union
from pathlib import Path
from itertools import combinations_with_replacement as combr
from itertools import combinations as combs
from itertools import *  # cycle, islice, permutations, product, repeat, zip_longest
from functools import lru_cache, partial, reduce
from collections import Iterable, defaultdict
import time
import re
from os import PathLike
import inspect
import gzip
import glob
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from joblib import load, dump, Parallel, delayed
from operator import add
__package__ = 'pylib'

top2tid = {'(4,(1,(2,3)));': 3, '(4,(2,(1,3)));': 2, '(4,(3,(1,2)));': 1}

SEED = 123454
MONTHS = 6
SECONDS = 3600*24*30*MONTHS


class Permuter:
    # TODO: rewrite this to store a permutation and take df's as input
    def __init__(self, taxa: Iterable = range(1, 5), outgroup: Union[str, int] = 4):
        # np.random.choice(self.leafnames, (n, 2))
        taxa = list(map(str, taxa))

        pdists = ["{}_{}".format(*l) for l in combs(taxa, 2)]

        outgroup = str(outgroup)
        leafnames = [t for t in taxa if t != outgroup]

        self.topologies = top2tid
        feature_strings = pdists + list(top2tid.keys())
        self.pairs = tuple(combr(leafnames, 2))
        self.mappers = [{}]
        for old, new in combs(leafnames, 2):
            m = {s: rename(old, new, s) for s in feature_strings}
            for k in top2tid:
                tid_old = top2tid[k]
                tid_new = top2tid[m[k]]
                m[tid_old] = tid_new
                m[f"top_{tid_old}"] = f"top_{tid_new}"
                del m[k]
            self.mappers.append(m)

    @property
    def indices(self):
        return [self.mappers[i] for i in self.ix]

    @indices.setter
    def indices(self, x: Iterable):
        """sets random permutation inds of same length as x"""
        self.ix = np.random.randint(0, len(self.mappers), len(x))

    @staticmethod
    def argsort(ix, *args):
        """returns the sorting index obtained from ix by swapping leaves a & b """
        if args and args[0] != args[1]:
            regexes = [
                partial(map_rx, re.compile(f), t)
                for f, t in zip((*args, "#"), ("#", *args))
            ]
            s = []
            for k in ["feature", "top"]:
                v = ix.get_level_values(k)
                for r in regexes:
                    # df = df.rename(columns=r)  # .rename(index=r,level='top')
                    v = v.map(r)
                s.append(v)
            return np.argsort(list(zip(*s)))  # .sort_index('index')
        else:
            return np.argsort(ix)

    def permute(self, x: pd.DataFrame) -> pd.DataFrame:
        """permutes cols of df according to self.indices matrix.
        returns a n-by-p matrix"""
        if not hasattr(self, "indices"):
            self.indices = x
        if len(x) != len(self.ix):
            raise ValueError(
                f"ix and input df are of different dims: {x}, {self.ix}")
        return (
            pd.concat(
                [
                    x.iloc[self.ix == i, :].rename(columns=self.mappers[i])
                    for i in np.unique(self.ix)
                ],
                join="inner",
            )
            .sort_index(axis="index")
            .sort_index(axis="columns")
        )


def load_and_filter(fn: PathLike,
                    topologies: bool = False,
                    permuter: Permuter = None):
    """Load a single hdf file, permutes topologies, and calculates y_prob value from IBL.
since the permuter maps are [0:top1->top1, 1:top1->top1, 2:top1->top3, 3:top1->top2],
    the species_topology index must be adjusted accordingly."""
    df = (load_y(fn, return_X=True)
          .drop('randomcolumn', axis=1, errors='ignore')
          .dropna(subset=[('y_prob', "", "")])
          .drop(columns=('y_true', '',  '')))
    if permuter is None:
        permuter = Permuter()
    df = permuter.permute(df)
    if topologies:
        df['species_topology'] = permuter.ix
        df['species_topology'] = df['species_topology'].map(
            {0: 0, 1: 0, 2: 2, 3: 1})
    return df


def load_hdf_files(filenames: 'list[Path]',
                   njobs: int,
                   dropna: int = True,
                   topologies: bool = False,
                   permuter: Permuter = None) -> pd.DataFrame:
    """Loads hdf files, permutes topologies, and calculates y_prob value from IBL"""

    X = Parallel(n_jobs=njobs)(
        delayed(load_and_filter)(fn, topologies, permuter) for fn in filenames
    )
    if dropna:
        return pd.concat(X).dropna()
    else:
        return pd.concat(X)


def load_data(data_dir: Path = None,
              data_files:  'list[Path]' = None,
              njobs: int = 12,
              test_size: float = .1,
              from_scratch: bool = True,
              return_index: bool = False,
              convert_to_pytorch: bool = True,
              train_test_file: Path = None,
              log_proba: str = None,
              dropna: bool = True,
              random_seed=123454,
              topology: bool = False,
              classify: bool = False,
              preprocessor: Path = None,
              conditions: str = None,
              tol: float = .1,
              rlim: float = None,
              llim: float = None,
              ):
    """if data_files is not specified, will look in data_dir. 
    Otherwise, data_dir is where train_test_file npz file is stored.
    Impute missing values from train split, return train-test splits with y_prob target"""
    from sklearn.impute import KNNImputer, MissingIndicator, SimpleImputer
    from sklearn.pipeline import make_pipeline, make_union

    if isinstance(data_dir, str):
        data_dir = Path(data_dir)

    if data_dir is not None:
        if data_files is None:
            data_files = data_dir.glob('*.hdf5')
        # elif data_files[0].parent != data_dir:
        #     raise ValueError(
        #         "Exactly one of data_files or data_dir must be specified")
    else:
        # if data_files is None:
        #     raise ValueError(
        #         "Exactly one of data_files or data_dir must be specified")
        data_dir = data_files[0].parent

    if train_test_file is None:
        train_test_file = data_dir / "train_test_split.npz"
    if not from_scratch and train_test_file.exists():
        with np.load(train_test_file) as X:
            x_train, x_test, y_train, y_test = X["x_train"], X["x_test"], X["y_train"], X["y_test"]
        print('loaded X from file...')
    else:
        try:
            X = load_hdf_files(data_files,
                               min(len(data_files), njobs),
                               topologies=topology,
                               dropna=dropna)
            if conditions is not None:
                X = X.query(conditions)
            # X.to_pickle(data_dir/'X_full.pd.gz', compression="gzip")
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {data_dir}")

        if topology:
            y = X.species_topology.values.astype(int)  # sigmoid(y) > .999

        elif classify:
            y = sigmoid(X['y_prob'].values)
            if rlim is not None and llim is not None:
                ix = np.logical_or(y < llim, y > rlim)
                X = X[ix]
                y = y[ix]
            elif tol:
                y = (y > 1-tol).astype(int)
        else:
            y = X['y_prob'].values
        ix = X.index
        X = X.drop(columns=['species_topology', 'y_prob'],
                   errors="ignore").values

        if 0 < test_size < 1:
            from sklearn.model_selection import train_test_split
            x_train, x_test, y_train, y_test, ix_train, ix_test = train_test_split(
                X, y, ix,
                test_size=test_size,
                random_state=random_seed,
                shuffle=True,
            )

        else:
            x_train = x_test = X
            ix_train = ix_test = ix
            y_train = y_test = y

        # TODO: if test_size==1, need to load a saved preprocessor from somewhere
        if preprocessor is None:
            preprocessor = StandardScaler()
            if not dropna:
                imputer = SimpleImputer(missing_values=np.nan,
                                        strategy='median',
                                        fill_value=0)
                preprocessor = make_pipeline(
                    make_union(imputer, MissingIndicator(
                        features='all'), n_jobs=4),
                    preprocessor)
            preprocessor.fit(x_train)
            dump(preprocessor, data_dir/'preprocessor.joblib')
        else:
            preprocessor = load(preprocessor)

        train_shape, test_shape = x_train.shape, x_test.shape
        x_train = preprocessor.transform(x_train)  # .reshape(train_shape)
        x_test = preprocessor.transform(x_test)  # .reshape(test_shape)
        np.savez_compressed(
            train_test_file,
            x_train=x_train, x_test=x_test,
            y_train=y_train, y_test=y_test)
    if not (topology or classify):
        y_train = sigmoid(y_train)
        y_test = sigmoid(y_test)
        if log_proba == 'both' or log_proba == 'train':
            y_train = np.log(y_train)
        if log_proba == 'both' or log_proba == 'test':
            y_test = np.log(y_test)
    if convert_to_pytorch:
        import torch.utils.data as data_utils
        from torch import Tensor
        x_train, x_test, y_train, y_test = map(
            Tensor, (x_train, x_test, y_train, y_test))
        if topology or classify:
            y_train = y_train.long()
            y_test = y_test.long()
        trainset = data_utils.TensorDataset(x_train, y_train)
        testset = data_utils.TensorDataset(x_test, y_test)
        if return_index:
            return trainset, testset, ix_train, ix_test
        return trainset, testset

    if return_index:
        return x_train, x_test, y_train, y_test, ix_train, ix_test
    return x_train, x_test, y_train, y_test


def summarize(t: Union[str, Tree], stree: bool = False) -> dict:
    """top2tid can be overwritten by version read from db in caller's namespace"""
    if isinstance(t, str):
        t = make_tree(t)
    d = {
        "newick": nwstr(t),
        "pdist": get_pdist(t).tolist(),
        "tid": top2tid[nwstr(t, format=9)],
    }
    if stree:
        d["ebl"] = get_ebl(t)
        d["ibl"] = get_ibl(t)
    return d


def is_recent(fn: Path, min_date=None) -> bool:
    '''Filters out files older than MONTHS and smaller than 2KiB'''
    if min_date is None:
        min_date = time.time()-SECONDS
    stats = fn.stat()
    return stats.st_mtime > min_date and stats.st_size > 2048


class TreeFile(object):
    def __init__(self, filename):
        self.filename = filename
        try:
            self.file = gzip.open(self.filename)
            next(self.file)
            self.file.seek(0)
        except gzip.BadGzipFile:
            self.file = open(filename, 'r')

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.file.close()

    def __next__(self):
        line = self.file.__next__()
        try:
            line = line.decode()
        except AttributeError:
            pass

        while not is_newick(line):
            line = self.file.__next__()
            try:
                line = line.decode()
            except AttributeError:
                pass
        return line

    def __iter__(self):
        return self

    def read_trees(self):
        return list(self)

# def read_gzip(filename: str) -> 'list[str]':
#     """read newick strings from file."""
#     try:
#         with gzip.open(filename) as f:
#             return list(filter(is_newick, f))
#     except gzip.BadGzipFile:
#         with open(filename, 'r') as f:
#             return list(filter(is_newick, f))


def make_tree(tstr: str,
              outgroup_name='4') -> Tree:
    t = Tree(tstr)
    try:
        t.set_outgroup(outgroup_name)
    except:
        pass
    return t


def ibl_est(s: Union[str, Path], max_height=100) -> pd.DataFrame:
    d = pd.read_hdf(s, "x")
    ibl = (d["1_4"] + d["1_3"] - d["1_2"] - d["3_4"]
           ).groupby(["ebl", "ibl"]).mean() / 2
    return (
        ibl[ibl.index.get_level_values("ebl") < max_height].reset_index().corr()[
            "ibl"]
    )


def ibl_corr() -> pd.DataFrame:
    '''g must be in caller's namespace'''
    for l in [100, 200, 500, 1000]:
        for f in np.arange(0.1, 1, 0.1):
            s = Path(f"g100_l{l}_f{f:.1f}_d500.hdf5")
            if s.exists():
                g[(l, f)] = ibl_est(s)
    return pd.DataFrame(g)


def make_if_not_exists(dirname: Union[str, Path]):
    """call mkdir recursively on entire path of subdirectories"""
    from pathlib import Path
    Path(dirname).mkdir(parents=True, exist_ok=True)
    return dirname


def write_results(results: 'list[pd.DataFrame]', ix,
                  outpath: Union[Path, str]):
    """write simulation summaries to file.  
    ebl,ibl,dset_no must be provided as separate index param."""
    rownames = ("ebl", "ibl", "dset_no")
    x = pd.concat(results, keys=ix, names=rownames).droplevel(-1)

    (
        x.sort_index("index")
        .sort_index("columns")
        .astype(np.float16)
        .to_hdf(outpath, key="x", format="fixed")
    )


def load_y(X: Union[pd.DataFrame, str, Path],
           key: str = "x",
           return_X: bool = False) -> Union[pd.DataFrame, 'tuple[pd.Series]']:
    """NOTE: if there are inferred trees in X which do not have corresponding true trees
    in the csv file, NaN y values will be returned"""
    # TODO: make sure this does what we want
    if not isinstance(X, pd.DataFrame):
        X = pd.read_hdf(X, key)
    g = pd.read_csv(
        "/N/project/phyloML/deep_ils/results/train_data/gene_tree_counts.csv",
        index_col=["ebl", "ibl", "tid"],
    )
    y_frac = (g / g.groupby(level=("ebl", "ibl")).sum()).xs(1, 0, "tid")
    y_true = log_odds(y_frac).reindex(X.index)

    ibl = X.index.get_level_values("ibl").to_frame(False).set_index(X.index)
    y_prob = log_odds(1 - 2 * np.exp(-2 * ibl) / 3, 1e-11)
    if return_X:
        X["y_true"] = y_true
        X["y_prob"] = y_prob
        return X
    else:
        y_true.columns = ["f"]
        y_prob.columns = ["p"]
        return y_true, y_prob


def log_odds(y: Union[pd.DataFrame, np.ndarray],
             eps: float = 1e-10) -> Union[pd.DataFrame, np.ndarray]:
    y = np.clip(y, eps, 1 - eps)
    return np.log(y / (1 - y))


def sigmoid(y, eps=1e-10):
    return 1 / (1 + np.exp(-y))


def get_ebl(t: Tree):
    return t.get_distance("1", t.get_common_ancestor("1", "2"))


def get_ibl(t: Tree):
    return t.get_distance(
        t.get_common_ancestor("1", "2"), t.get_common_ancestor("1", "3")
    )


def get_pdist(t: Tree):
    leaves = sorted(t.get_leaves(), key=lambda s: s.name)
    n = len(leaves)
    m = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            m[i, j] = m[j, i] = t.get_distance(leaves[i], leaves[j])
    return m


def df2counts(df: pd.DataFrame, sp_tree="(4,(3,(1,2)));"):
    """get frac of concordant trees"""
    ycounts = df["count"].xs("mean", "index", "stat").unstack()
    return ycounts[sp_tree] / ycounts.sum(1)


def compose(g: Callable, f: Callable):
    def h(*args, **kwargs):
        print("compose args", args, kwargs)
        return g(f(*args, **kwargs))

    return h


def drop_redundant_count_cols(df: pd.DataFrame, inplace: bool = False):
    """must have 1-level column index.
    just keep 1 of the 5 redundant count statistics in unstacked df"""
    cols = [cc for cc in df.columns.tolist(
    ) if "count" in cc and "mean" not in cc]
    res = df.drop(cols, axis=1, inplace=inplace)
    if inplace:
        res = df
    res.columns.names = ["feature", "stat", "top"]
    return res


def group_by_dataset(df: pd.DataFrame):
    d2 = df.rename_axis(index=["top", "stat"], columns="feature")
    d2 = d2.reset_index()
    d2["dataset"] = d2.index // 15

    return d2.set_index(["dataset", "top", "stat"])


def swap(a: str, b: str, s: str) -> str:
    return s.replace(a, "##").replace(b, a).replace("##", b)


def rename(a, b, s):
    s = swap(a, b, s)
    if "_" in s:
        return "_".join(sorted(s.split("_")))
    elif is_newick(s):
        return nwstr(s, format=9)


class OldPermuter:
    # TODO: rewrite this to store a permutation and take df's as input
    def __init__(self, leafnames: Iterable, outgroup: Union[str, int]):
        # np.random.choice(self.leafnames, (n, 2))
        leafnames = list(map(str, leafnames))

        pdists = ["{}_{}".format(*l) for l in combs(leafnames, 2)]

        outgroup = str(outgroup)
        if outgroup in leafnames:
            leafnames.remove(outgroup)

        topologies = set(
            nwstr("({},({},({},{})));".format(outgroup, *l), 9)
            for l in permutations(leafnames)
        )
        self.topologies = topologies
        top2id = {k: v + 1 for v, k in enumerate(topologies)}
        feature_strings = pdists + list(topologies)
        self.pairs = tuple(combr(leafnames, 2))
        self.mappers = [{}]
        for old, new in combs(leafnames, 2):
            m = {s: rename(old, new, s) for s in feature_strings}
            for k in topologies:
                m[top2id[k]] = top2id[m[k]]
                m["top_{}".format(top2id[k])] = "top_{}".format(top2id[m[k]])
                del m[k]
            self.mappers.append(m)

    @property
    def indices(self):
        return [self.mappers[i] for i in self.ix]

    @indices.setter
    def indices(self, x: Iterable):
        """sets random permutation inds of same length as x"""
        self.ix = np.random.randint(0, len(self.mappers), len(x))

    @staticmethod
    def argsort(ix, *args):
        """returns the sorting index obtained from ix by swapping leaves a & b """
        if args and args[0] != args[1]:
            regexes = [
                partial(map_rx, re.compile(f), t)
                for f, t in zip((*args, "#"), ("#", *args))
            ]
            s = []
            for k in ["feature", "top"]:
                v = ix.get_level_values(k)
                for r in regexes:
                    # df = df.rename(columns=r)  # .rename(index=r,level='top')
                    v = v.map(r)
                s.append(v)
            return np.argsort(list(zip(*s)))  # .sort_index('index')
        else:
            return np.argsort(ix)

    def permute(self, x: pd.DataFrame) -> pd.DataFrame:
        """permutes cols of df according to self.indices matrix.
        returns a n-by-p matrix"""
        if not hasattr(self, "indices"):
            self.indices = x
        if len(x) != len(self.ix):
            raise ValueError(
                f"ix and input df are of different dims: {x}, {self.ix}")
        return (
            pd.concat(
                [
                    x.iloc[self.ix == i, :].rename(columns=self.mappers[i])
                    for i in np.unique(self.ix)
                ],
                join="inner",
            )
            .sort_index(axis="index")
            .sort_index(axis="columns")
        )


trailing_str = re.compile(r"_[\d]+_[\d]+")
m = re.compile(r"([\d]+)_")
nwchars = re.compile("[();]")


def nw2leaves(x: str):
    return tuple(sorted(nwchars.sub("", x).split(",")))


def nw2leafstr(x: str):
    return ",".join(sorted(nwchars.sub("", x).split(",")))


nC4 = partial(combs, r=4)
nC3 = partial(combs, r=3)
nC2 = partial(combs, r=2)


def leaves2cov(leaf1: str, leaf2: str):
    return "_".join((leaf1, leaf2))


def pair2cov(pair: Iterable):
    return "_".join(map(str, pair))


def triplet2nw(a, b, c):
    return "(%s,(%s,%s);" % (*sorted((a, b)), c)


def duplicate(iterable: Iterable, n):
    """generator that repeats each element in x n times"""
    for x in iterable:
        for y in (x,) * n:
            yield y


def triu(a: Iterable, labels=None, diag: bool = True, exclude: Iterable = None):
    """takes iterable and gets lower triangular
    entries in *column-major order*.
    Optionally removes certain cols."""
    if labels is not None:
        if diag:
            d = dict(
                zip(
                    labels,
                    (z for i, x in enumerate(a)
                     for j, z in enumerate(x) if j <= i),
                )
            )
        else:
            d = dict(
                zip(
                    labels,
                    (z for i, x in enumerate(a)
                     for j, z in enumerate(x) if j < i),
                )
            )
        if exclude is not None:
            for e in exclude:
                if e in d:
                    d.pop(e)
    else:
        if diag:
            d = [z for i, x in enumerate(a) for j, z in enumerate(x) if j <= i]
        else:
            d = [z for i, x in enumerate(a) for j, z in enumerate(x) if j < i]
        if exclude is not None:
            for e in exclude:
                if e in d:
                    d.remove(e)

    return d


def flatten(obj: Iterable):
    """use recursive version for greater than 1 level"""
    from itertools import chain

    return list(chain.from_iterable(obj))
    # for i in obj:
    #     if isinstance(i, Iterable) and not isinstance(i, string_types):
    #         yield from flatten(i)
    #     else:
    #         yield i


# def array_index(arr, vocab=None, to_values=None):
#     """converts arr of arbitrary type to integer-indexed array"""
#     if vocab is None:
#         vocab = np.unique(arr)
#     vocab_size = len(vocab)
#     if to_values is None:
#         to_values = np.arange(1, vocab_size+1)
#     sort_idx = np.argsort(vocab)
#     idx = np.searchsorted(vocab, alignment, sorter=sort_idx)

#     return to_values[sort_idx][idx]


def get_topo_freqs(session, table, col):
    """get topology freqs by species tree.  (Can't subsample)"""
    from sqlalchemy.sql.expression import func

    q = session.query(func.count(col).label("count"), col, table.c.sid).group_by(
        col, table.c.sid
    )  # redundant
    print("query", q)
    gtops = pd.read_sql(q.statement, q.session.bind).pivot(
        index="sid", columns=col)
    gtops.columns = gtops.columns.droplevel()
    return gtops


def default_counter(keys: Iterable, counts: dict):
    return [counts[k] if k in counts else 0 for k in keys]


def times2tree(join_times: Iterable):
    """takes join times of ladderized newick and returns rooted tree"""
    join_times = sorted(join_times)
    names = map(str, range(2, len(join_times) + 1))
    t1 = 0
    nwstr = "1"
    for n, t in zip(names, join_times):
        nwstr = "({b1}_{t1},{b2}_{t2})".format(b1=nwstr, b2=n, t1=t - t1, t2=t)
        t1 = t
    return Tree(nwstr + ";")


def invert(f):
    """inverts a 1-to-1 dict/ordered dict or other"""
    return f.__class__(map(reversed, f.items()))


def is_nonempty(fn: Path) -> bool:
    return fn.stat().st_size > 2048


def is_newick(tree_str: str) -> bool:
    """Necessary but not sufficient test for newickness.
        Accepts strings of the format <metadata> <newick_tree>"""
    fields = tree_str.split()
    if not len(fields):
        return False
    # try:
    #     t = fields[-1].decode()
    # except AttributeError:
    t = fields[-1]
    return t.endswith(";") and t.startswith("(")


def tips(leaves: Iterable) -> List:
    """['a','b','c'] -> ['a:a','b:b','c:c']"""
    if type(leaves) == str:
        return [pair2cov((c, c)) for c in leaves.split(",")]
    else:
        return [pair2cov((c, c)) for c in leaves]


# assumes this pair is already sorted, maps '1,2,3' -> ['1:1','1:2',...]
def leafset2pairs(leaves: Iterable) -> List:
    if type(leaves) == str:
        return [pair2cov(c) for c in combr(leaves.split(","), 2)]
    else:
        return [pair2cov(c) for c in combr(leaves, 2)]


# operates along rows of df
def VMR(x):
    return (
        isinstance(x, np.ndarray)
        and np.nanvar(x, 1) / np.nanmean(x, 1)
        or np.nanvar(x) / np.nanmean(x)
    )


# aka index of dispersion


@numba.njit(parallel=True)
def tot_len(y):
    """compute total length from symmetric cov mat"""
    m, n = y.shape
    r = 0
    for i in range(m):
        for j in range(n):
            r += i == j and 2 * y[i, j] or -y[i, j]
    return r / 2


def total_length(x, version=None):
    """return the total length of a tree from the tips (old) or cov mat(new)"""
    if version == "v2":
        return tot_len(x)
    if isinstance(x, np.ndarray):
        return np.sum(x, 1)
    elif inspect.isgenerator(x):
        return sum(x)
    else:
        return np.sum(x)


# TODO: make this a class
summaries = ("mean", "std", "min", "max", "median")
summary_dict = dict(zip(("nan" + s for s in summaries), summaries))

# dask_median = partial(da.percentile, q=50)
# TODO: doesnt handle nans


@numba.guvectorize(
    [
        (numba.float32[:, :], numba.float32[:, :]),
        (numba.float64[:, :], numba.float64[:, :]),
    ],
    "(n,k),(m,k)",
    nopython=True,
)
def summarize_into(x, res):
    """maps mxn array/df into nx5 summary statistics.
    turned off parallel=True, since numba blocks forever when used with a
    multiprocessing queue"""
    for i in range(x.shape[1]):
        res[0, i] = np.nanmean(x[:, i])
        res[1, i] = np.nanstd(x[:, i])
        res[2, i] = np.nanmin(x[:, i])
        res[3, i] = np.nanmax(x[:, i])
        res[4, i] = np.nanmedian(x[:, i])


def cov_summaries(x):
    return (
        np.nanmean(x, 0),
        np.nanstd(x, 0),
        np.nanmin(x, 0),
        np.nanmax(x, 0),
        np.nanmedian(x, 0),
    )


# regex to extract taxa names
# ? to handle polytomies
triotaxa = re.compile(r"\((\d+),\(?(\d+),(\d+)\)?\);")


def leaves2labels(leafnames: Iterable, diag=True) -> 'list[str]':
    if diag:
        comb_fun = combr
    else:
        comb_fun = combs
    return [
        "_".join(s) for s in sorted(comb_fun(leafnames, 2), key=lambda c: (c[1], c[0]))
    ]


def pdist_mapper(leafnames):
    labels = leaves2labels(leafnames, diag=False)

    mapper = {
        "pdist": partial(
            triu, labels=[l for l in labels],
            diag=False
        )
    }
    return mapper


def num_rooted_trees(n):
    return int(np.math.factorial(2 * n - 3) / (2 ** (n - 2) * np.math.factorial(n - 2)))


def standardize(t: Tree) -> Tree:
    """sort the leaves in a standard format *in place*"""
    t.sort_descendants()
    t.ladderize()
    return t


def nwstr(t: Union[Tree, str], format=1) -> str:
    """FORMAT DESCRIPTION SAMPLE
    0 flexible with support values ((D:0.723274,F:0.567784)1.000000:0.067192,(B:0.279326,H:0.756049)1.000000:0.807788);
    1 flexible with internal node names ((D:0.723274,F:0.567784)E:0.067192,(B:0.279326,H:0.756049)B:0.807788);
    2 all branches + leaf names + internal supports ((D:0.723274,F:0.567784)1.000000:0.067192,(B:0.279326,H:0.756049)1.000000:0.807788);
    3 all branches + all names ((D:0.723274,F:0.567784)E:0.067192,(B:0.279326,H:0.756049)B:0.807788);
    4 leaf branches + leaf names ((D:0.723274,F:0.567784),(B:0.279326,H:0.756049));
    5 internal and leaf branches + leaf names ((D:0.723274,F:0.567784):0.067192,(B:0.279326,H:0.756049):0.807788);
    6 internal branches + leaf names ((D,F):0.067192,(B,H):0.807788);
    7 leaf branches + all names ((D:0.723274,F:0.567784)E,(B:0.279326,H:0.756049)B);
    8 all names ((D,F)E,(B,H)B);
    9 leaf names ((D,F),(B,H));
    100 topology only ((,),(,));"""
    if isinstance(t, str):
        t = Tree(t)
    return standardize(t).write(format=format)


def subtree(parent_tree: Tree, taxa, newick: bool = True, rename=None) -> Tree:
    """default: returns n-taxon topologically invariant subtree in newick fmt.
    sorted in lexicographic->ladderized order: ((a,b),c);"""
    if not all(x in parent_tree for x in taxa):
        return None  # not in this tree
    t = parent_tree.copy("newick")
    t.prune(taxa)
    if rename is not None:
        for n in t:
            if n.name in rename:
                n.name = rename[n.name]
    if newick:
        return standardize(t).write(format=9)
    return t


def get_cov(tree, leafnames: Iterable = None, joined: bool = False, format: str = ""):
    """return phylogenetic covariance dict; cov over all tips, including duplicate genes.
        Ordered by leaf names.
    This is an unbound version of the method in TreeConfig, useful for raw newick trees
    """
    # TODO clean up
    if isinstance(tree, str):
        t = Tree(nwstr)
    elif isinstance(tree, Tree):
        t = tree
    if leafnames is None:
        leafnames = sorted(t.get_leaf_names())
    leaves = sorted((l for l in t if l.name in leafnames),
                    key=lambda x: x.name)

    if format == "array":  # WARNING: assumes all taxa are represented
        ntaxa = len(leaves)
        m = np.empty((ntaxa, ntaxa))
        for i, leaf1 in enumerate(leaves):
            for j in range(i, ntaxa):
                leaf2 = leaves[j]
                m[i, j] = m[j, i] = t.get_distance(
                    leaf1.get_common_ancestor(leaf2))
        return m
    elif format == "joined":
        d = {leaves2cov(*k): np.nan for k in combr(sorted(leafnames), 2)}
    else:
        d = {k: np.nan for k in combr(sorted(leafnames), 2)}

    for leaf1, leaf2 in combr(sorted(leaves, key=lambda x: x.name), 2):
        key = joined and leaves2cov(leaf1.name, leaf2.name) or (leaf1, leaf2)
        d[key] = t.get_distance(leaf1.get_common_ancestor(leaf2))
    if np.any(np.isnan(list(d.values()))):
        print(d, nwstr(t))
        raise ValueError
    return d


class TreeConfig:
    def __init__(
        self,
        leafnames=None,
        ntaxa=None,
        outgroup=None,
        subtree_sizes=None,
        include_outgroup=False,
    ):
        self.include_outgroup = include_outgroup  # return outgroup w/ every nw topo
        if ntaxa is None and leafnames is None:
            raise ("must specify one of: ntaxa,leafnames")
        elif ntaxa:
            self.ntaxa = ntaxa
            self.leafnames = sorted(str(i) for i in range(ntaxa))
        elif leafnames:
            self.leafnames = sorted(str(i) for i in leafnames)
            self.ntaxa = len(leafnames)
        if subtree_sizes is None:
            self.subtree_sizes = [self.ntaxa]
        else:
            self.subtree_sizes = [int(s) for s in subtree_sizes]
        self.outgroup_name = str(outgroup)
        self.ntops = sum(
            num_rooted_trees(k) for k in self.subtree_sizes
        )  # TODO: make this a list
        self.npairs = sum(k * (k + 1) // 2 for k in self.subtree_sizes)
        self.top_schema = [
            ("topo", np.str),
            ("count", np.int16),
        ]  # must be list so we know which is which
        self.cov_schema = {
            leaves2cov(leaf1, leaf2): np.float32
            for leaf1, leaf2 in combr(self.leafnames, 2)
        }  # have to sort if its a set

    @property
    @lru_cache(maxsize=16)
    def trio_pair_inds(self):
        inds = np.empty(
            shape=(nchoosek(self.ntaxa, 3, exact=True), 3), dtype=np.int8)
        get_trio_inds(inds, self.ntaxa)
        return inds

    def make_tree(self, nwstr, single_copy=True):
        """makes ete3 tree from newick string, sets outgroup if it is present.
        If single_copy=False, keep the highest-index gene from each family,
        otherwise ignore families with duplicates"""
        if isinstance(nwstr, Tree):
            t = nwstr
        else:
            try:
                t = Tree(nwstr)  # trailing_str.sub('',nwstr))
            except Exception as e:
                print("couldn not create tree from", nwstr)
                raise e

        if not single_copy:
            gene_copies = {m.search(k.name).group(
                1): k for k in t.get_leaves()}
            # keep 1st copy.  TODO: make this customizable
            t.prune(gene_copies.values())
        elif len(set(trailing_str.sub("", leaf.name) for leaf in t)) < len(t):
            # ignore multi-copy genes
            return None

        for leaf in t:  # now remove trailing id
            leaf.name = trailing_str.sub("", leaf.name)

        try:
            t.prune(self.leafnames)
            t.set_outgroup(self.outgroup_name)
        # usually its ValueError("Node names not found: "...
        except Exception as e:
            # TODO: should we ignore these trees?
            pass

        return t

    def get_labels(self):
        return leaves2labels(self.leafnames, diag=False)

    def get_cov(self, t, format=""):
        """return phylogenetic covariance dict; cov over all tips, including duplicate genes"""
        leaves = sorted(
            (l for l in t if l.name in self.leafnames), key=lambda x: x.name
        )
        if format == "array":  # WARNING: assumes all taxa are represented
            m = np.empty((self.ntaxa, self.ntaxa))
            for i, leaf1 in enumerate(leaves):
                for j in range(i, self.ntaxa):
                    leaf2 = leaves[j]
                    m[i, j] = m[j, i] = t.get_distance(
                        leaf1.get_common_ancestor(leaf2))
        else:
            m = {k: np.nan for k in self.cov_schema}
            for leaf1, leaf2 in combr(leaves, 2):
                key = leaves2cov(leaf1.name, leaf2.name)
                m[key] = t.get_distance(leaf1.get_common_ancestor(leaf2))
            if np.any(np.isnan(list(m.values()))):
                print(m, nwstr(t))
                raise ValueError
        return m

    def get_topologies(self, tree):
        leaf_names = sorted(
            l.name for l in tree if l.name != self.outgroup_name)
        taxa_subsets = ((*combs(leaf_names, r),) for r in self.subtree_sizes)
        return [subtree(tree, taxa) for taxa in taxa_subsets]

    def nw(self, leaf_names=None):
        return list(self.nw_iter(leaf_names))

    def nw_iter(self, leaf_names=None):
        """generates all nw trees for a,b in leafnames in sort-ladderize order"""
        try:
            from .enumerate_trees import enum_unordered
        except:
            from enumerate_trees import enum_unordered
        if leaf_names is None:
            leaf_names = self.leafnames.copy()
            # [ln for ln in self.leafnames if self.outgrou
            leaf_names.remove(self.outgroup_name)
        for tup in enum_unordered(leaf_names):
            if self.include_outgroup:
                tup = (self.outgroup_name, tup)
            yield standardize(Tree(re.sub("[\"' ]", "", str(tup)) + ";")).write(
                format=9
            )

    def nw_cov_iter(self, leaf_names=None):
        """generates a:b strings for a,b in leafnames, exclude variances a:a and outgroup"""
        return zip(self.nw_iter(leaf_names), self.cov_iter(leaf_names))

    def cov_top_iter(self, leaf_names=None):
        """generates a:b strings for a,b in leafnames, exclude variances a:a and outgroup"""
        if leaf_names is None:
            leaf_names = self.leafnames
            # [ln for ln in self.leafnames if self.outgroup_name!=ln]
            leaf_names.remove(self.outgroup_name)
        taxa_subsets = ((*combs(leaf_names, r),) for r in self.subtree_sizes)
        for c in taxa_subsets:
            yield (
                [pair2cov(leaves) for leaves in combs(c, 2)],
                [pair2cov(leaves) for leaves in combr(c, 2)],
            )

    def top_iter(self, leaf_names=None):
        """generates a:b strings for a,b in leafnames, exclude variances a:a and outgroup"""
        if leaf_names is None:
            # [ln for ln in self.leafnames if self.outgroup_name!=ln]
            leaf_names = self.leafnames
        taxa_subsets = ((*combs(leaf_names, r),) for r in self.subtree_sizes)
        for c in taxa_subsets:
            yield [pair2cov(leaves) for leaves in combs(c, 2)]

    def cov_iter(self, leaf_names=None):
        """generates a:b strings for a,b in leafnames, include variances a:a and outgroup"""
        if leaf_names is None:
            # [ln for ln in self.leafnames if self.outgroup_name!=ln]
            leaf_names = self.leafnames
        taxa_subsets = ((*combs(leaf_names, r),) for r in self.subtree_sizes)
        for c in taxa_subsets:
            yield [pair2cov(leaves) for leaves in combr(c, 2)]

    def __repr__(self):
        return "{}\n{}\noutgroup: {}\n".format(
            type(self), self.leafnames, self.outgroup_name
        )


# covariance -> top funcs
# @numba.njit(parallel=True)


def covs2top(arr):
    """assume input is in form ab,ac,bc or aa,ab,ac,bb,bc,cc
    for 3 taxa, returns the label of the cherry;  0=(ab)c, 1=(ac)b, 2=(bc)a, None=polytomy"""
    if len(arr) == 3:
        s = np.sort(arr)
        if s[1] != s[0]:  # polytomy
            return np.nan
        return arr.idxmax()


@numba.njit
def get_trio_inds(x, n):
    """x is (n-choose-3)-by-3"""
    r, l = x.shape
    row = 0
    for i in range(n - 2):
        x[row, 0] = i
        for j in range(i + 1, n - 1):
            x[row, 1] = i * n + j
            for k in range(j + 2, n):
                x[row, 2] = k
                row += 1


###### data join and reduction funcs ########

# dd.Aggregation('custom_sum',)

# @numba.njit(parallel=True)
def sort_tops(row):
    if row.count_t.isna():
        cols = [leaves2cov(*leaves)
                for leaves in combr(row.index.split(","), 2)]


class Reductions:
    def __init__(self, covs):
        self.covs = covs
        self.stat_names = ("mean", "std", "min", "max")
        self.reducers = lambda x: (x.mean(0), x.std(0), x.min(0), x.max(0))
        self.col_names = covs.columns

    #    def reduce(self, col_names=None):
    #        if col_names is None:
    #            col_names = self.col_names
    #        x = self.covs[col_names].dropna().to_dask_array()
    #        stats = da.concatenate(
    #            self.reducers(x),
    #            axis=0
    #        )  # .reshape((1,-1))
    #        return stats  # .to_dask_dataframe(columns=meta.keys())

    def get_metadata(self, col_names):
        return {"_".join(k): np.float32 for k in product(col_names, self.stat_names)}


# HDF


class HDF5Store(object):
    """
    Simple class to append value to a hdf5 file on disc (usefull for building keras datasets)

    Params:
        datapath: filepath of h5 file
        dataset: dataset name within the file
        shape: dataset shape (not counting main/batch axis)
        dtype: numpy dtype
    attributes: ('attrib_name',(dset1_attribs,dset2_attribs,...))

    Usage:
        hdf5_store = HDF5Store('/tmp/hdf5_store.h5',['X'], shape=[(20,20,3)])
        x = np.random.random(hdf5_store.shape)
        hdf5_store.append(x)
        hdf5_store.append(x)

    From https://gist.github.com/wassname/a0a75f133831eed1113d052c67cf8633
    """

    def __init__(
        self,
        datapath,
        datasets,
        shapes,
        dtypes,
        attributes=("", ""),
        compression="gzip",
        chunk_len=1,
        nan_to_num=False,
    ):
        import h5py
        from os import path

        self.datapath = datapath
        self.datasets = datasets
        self.shapes = dict(zip(datasets, shapes))
        self.chunk_len = chunk_len
        self.dtypes = dict(zip(datasets, dtypes))
        self.noNans = nan_to_num
        attribute_list = (np.array(a, dtype="S100")
                          for a in attributes[1])  # utf8

        if not path.isfile(datapath):
            with h5py.File(self.datapath, mode="w", libver="latest") as h5f:
                for dtype, (dataset, shape), attr in zip(
                    self.dtypes.values(), self.shapes.items(), attribute_list
                ):

                    dset = h5f.create_dataset(
                        dataset,
                        shape=(0,) + shape,
                        maxshape=(None,) + shape,
                        dtype=dtype,
                        compression=compression,
                        chunks=True,
                    )  # since chunks depend on length of avi

                    dset.attrs[attributes[0]] = attr
                print("made dset")
                for ds in h5f:
                    print(ds)

        else:
            print("file exists: %s\nappending..." % datapath)

    def append(self, values, dataset):
        import h5py

        shape = self.shapes[dataset]
        with h5py.File(self.datapath, mode="a", libver="latest") as h5f:
            for ds in h5f:
                print(ds)
            dset = h5f[dataset]
            last = dset.shape[0]
            dset.resize((last + 1,) + shape)
            dset[last] = [np.reshape(values, shape)]
            h5f.flush()

    def extend(self, value_list, dataset):
        """takes a LIST OR TUPLE and appends it to hdf file"""
        import h5py

        shape = self.shapes[dataset]
        n = len(value_list)
        try:
            vals = np.reshape(value_list, (-1, *shape)
                              ).astype(self.dtypes[dataset])
            if self.noNans:
                vals = np.nan_to_num(vals, copy=False)
            with h5py.File(self.datapath, mode="a", libver="latest") as h5f:
                dset = h5f[dataset]
                last = dset.shape[0]
                dset.resize((last + n,) + shape)
                dset[last: (last + n)] = vals
                h5f.flush()
        except Exception as e:
            print(
                "Error writing to hdf",
                self.shapes,
                self.datasets,
                "shape",
                shape,
                dataset,
                "value len",
                n,
                (-1, *shape),
            )
            raise (e)

    def __repr__(self):
        return "\n".join(
            map(str, (self.datapath, self.datasets, self.dtypes, self.shapes))
        )

    def remove(self):
        from os import remove
        remove(self.datapath)


def get_feature_str(learner):
    return str(get_features(learner))


def get_features(learner):
    if hasattr(learner, "feature_importances_"):
        return learner.feature_importances_
    elif hasattr(learner, "coef_"):
        return learner.coef_.ravel()
    return None


def multidict_to_df(user_dict, names=("ix1", "ix2")):
    """convert dict-of-dicts or dict-of-lists to multilevel dataframe.
    2nd arg is the 2 names of the index variables."""
    keys = list(user_dict.keys())
    try:
        if type(user_dict[keys[0]]) == list:  # dict of lists
            df = pd.DataFrame.from_dict(
                {
                    (i, j): user_dict[i][j]
                    for i in keys
                    for j in range(len(user_dict[i]))
                },
                orient="index",
            )
        else:
            df = pd.DataFrame.from_dict(
                {(i, j): user_dict[i][j]
                 for i in keys for j in user_dict[i].keys()},
                orient="index",
            )
    except Exception as e:
        print(user_dict[keys[0]])
        raise (e)

    df.index = pd.MultiIndex.from_tuples(df.index)
    df.index.names = names
    return df


# general
def grouper(iterable, n: int, fillvalue=None):
    """Collect data into fixed-length chunks or blocks.
    grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"""
    args = [iter(iterable)] * n
    z = zip_longest(*args, fillvalue=fillvalue)
    s = next(z)
    while None not in s:
        yield s
        s = next(z)
    yield (*filter(None, s),)


def chunker(iterable, n: int, fillvalue="none"):
    """like grouper, but slightly faster.
    with fillvalue='none', does not fill last slice"""
    if isinstance(iterable, int):
        iterable = range(iterable)
    try:
        i1, i2 = 0, n
        r = iterable[i1:i2]
        while not r.empty:
            yield r
            i1 = i2
            i2 += n
            r = iterable[i1:i2]
    except:
        it = iter(iterable)
        r = (*islice(it, n),)
        if fillvalue == "none":
            while r:
                yield r
                r = (*islice(it, n),)
        else:
            args = [iter(iterable)] * n
            return zip_longest(*args, fillvalue=fillvalue)


lmap = lambda func, *iterables: list(map(func, *iterables))


def idem(f, x, *args, **kwargs):
    """makes conversions idempotent.  TODO: make this a higher-order function"""
    try:
        return f(x, *args, **kwargs)
    except:
        return x


# descriptive stats


def read_trees(tree_file, outgroup=None):
    with open(tree_file, "r") as f:
        trees = [Tree(s) for s in f if s.endswith(";\n")]
        if outgroup is not None:
            for t in trees:
                t.set_outgroup(outgroup)
    return trees


def get_tree_dist(gtrees, itrees):
    comps = (t1.compare(t2) for t1, t2 in zip(gtrees, itrees))
    return np.mean(np.array([s["rf"] for s in comps]) != 0)


def coal2div(dirname: Path,
             model='LG_LG'):
    # TODO: make sure topologies match for each tree
    from .sequtils import seqs2pwdists
    rx = re.compile('^t_([\d\.]+)_')
    rx_stem = re.compile('^(t_.*?)_(LG|WAG).*')
    lengths = {}
    # for fn in dirname.glob('inferred_trees/*gz'):
    for fn in dirname.glob('seq_backups/*gz'):
        if model in fn.stem:
            try:
                with TreeFile(fn) as tf:
                    div_length = np.mean([make_tree(tree).get_distance(
                        '1', '2')/2 for tree in tf])
                stem = rx_stem.match(fn.stem).group(1)
                parentfile = dirname / 'trees' / (stem + '.trees.gz')
                with TreeFile(parentfile) as parent:
                    coal_length = np.mean(
                        [make_tree(tree).get_distance('1', '2')/2 for tree in parent])
                seq_length = seqs2pwdists(
                    dirname/'seq_backups'/(stem+'_LG.seqs.tar.gz'))['1,2']/2
                # (div_length, seq_length)
                lengths[coal_length] = (seq_length,)
            except StopIteration as e:
                continue
                # raise e
    lengths = (pd
               .DataFrame.from_dict(lengths,
                                    orient='index',
                                    # ['raxml_units', 'seq_div'],
                                    columns=['seq_div'],
                                    )
               .sort_index()
               .dropna()
               )
    lengths.index.set_names('coal_units', inplace=True)
    return lengths


def process_coal2div(dirname: Path, fn: Path):
    from .sequtils import seqs2pwdists
    rx_stem = re.compile('^(t_.*?)_(LG|WAG).*')

    # with TreeFile(fn) as tf:
    #     div_length = [make_tree(tree).get_distance(
    #         '1', '2')/2 for tree in tf]
    stem = rx_stem.match(fn.stem).group(1)
    parentfile = dirname / 'trees' / (stem + '.trees.gz')
    with TreeFile(parentfile) as parent:
        coal_length = [make_tree(tree).get_distance(
            '1', '2')/2 for tree in parent]
    seq_length = seqs2pwdists(
        dirname/'seq_backups'/(stem+'_LG.seqs.tar.gz'), reduce=None)['1,2']/2
    # (div_length, seq_length)
    return list(zip(coal_length, seq_length))


def coal2div2(dirname: Path,
              model='LG'):
    # TODO: make sure topologies match for each tree
    with Parallel(n_jobs=2) as parallel:
        lengths = parallel(
            delayed(process_coal2div)(dirname, fn)
            for fn in dirname.glob('seq_backups/*gz') if model in fn.stem)

    lengths = reduce(add, lengths)
    lengths = (pd
               .DataFrame(
                   lengths,
                   # columns=['raxml_units', 'seq_div'],
                   columns=['coal_units', 'seq_div'],
               )
               .set_index('coal_units')
               .sort_index()
               .dropna()
               )
    # lengths.index.set_names('coal_units', inplace=True)
    return lengths
