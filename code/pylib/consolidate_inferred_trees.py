from contextlib import contextmanager
import argparse
import gc
import random
import re
import time
from functools import partial
from pathlib import Path
from sys import argv
import gzip
import pandas as pd
from joblib import Parallel, delayed
from multiprocess import Pool
from psycopg2.errors import UniqueViolation
from collections import namedtuple
from sqlalchemy.exc import IntegrityError

import utils
from sql_utils import make_connection, select, write_rows_to_sql

schema = "sim4"
outgroup_name = "4"
MONTHS = 6


stree_rx = re.compile(r"t_([\d\.]+)_([\d\.]+)_([\d\.]+)")
filename_rx = re.compile(
    r"(t_\d.*)_(WAG|LG)_[PROTCA]*(LG|WAG)[+G]*[\._](ds\d+\.)?")


def fn2nw(s: str) -> str:
    ta, tb, tc = stree_rx.search(s).groups()
    ibl = float(tb) - float(ta)
    outgroup_bl = float(tc) - float(tb)
    tstr = "(4:{tc},(3:{tb},(1:{ta},2:{ta}):{ibl}):{obl});".format(
        ta=ta, tb=tb, tc=tc, ibl=ibl, obl=outgroup_bl
    )
    return utils.nwstr(utils.make_tree(tstr))


def value_mapper(d):
    def f(X):
        try:
            return [d[x] for x in X]
        except:
            return d[X]

    return f


Params = namedtuple(
    "Params", ("filename", "stree", "stree_str", "smodel", "imodel", "ds")
)


def parse_filename(fn: Path) -> Params:
    s = fn.name
    m = filename_rx.search(s)
    try:
        stree_str, smodel, imodel, ds = m.groups()
    except AttributeError as e:
        print(fn, filename_rx)
        raise e
    if ds:
        ds = int(ds[2:-1]) - 1
    return Params(
        filename=fn,
        stree=fn2nw(stree_str),
        stree_str=stree_str,
        smodel=smodel,
        imodel=imodel,
        ds=ds,
    )


def process_line(args: 'tuple[int,str]',
                 sid: int) -> dict:
    """process line.  
    If the line string contains a number, this will override the first field of args."""
    i, line = args
    line = line.split()
    if len(line) == 2:
        i = int(line[0])
    line = line[-1]
    tree_stats = utils.summarize(line)
    tree_stats["sid"] = sid
    tree_stats["tree_no"] = i
    return tree_stats


if __name__ == "__main__":
    print("Usage: <dirpath> <nprocs> <schema.table>")
    dirpath = Path(argv[1])
    dirname = dirpath.name
    nprocs = int(argv[2])

    print('arguments:', argv)

    conn = make_connection(schema)

    top2tid = pd.read_sql_table(
        "topologies",
        con=conn,
        schema=schema,
        columns=["topology", "tid"],
        index_col="topology",
    )["tid"].to_dict()
    nblocks = 1
    heterotachy = False
    if 'heterotachy' in dirname:
        inferred_table = 'heterotachy_inferred'
        slength = 500
        heterotachy = True
    elif '1_rate' in dirname:
        inferred_table = 'one_rate_inferred'
        slength = 500
        heterotachy = True
    elif "bp_b" in dirname:
        slength, nblocks, ngenes = map(
            int, re.search("/?(\d+)bp_b(\d+)_g(\d+)", dirname).groups()
        )
        inferred_table = "rec_inferred"
    else:
        slength = int(re.search("(\d+)bp", dirname).group(1))
        inferred_table = "nonrec_inferred"
    print("seq length:", slength, inferred_table, "getting filenames")

    filenames = list(
        filter(
            utils.is_recent,
            (dirpath / "inferred_trees").glob("*.raxml.trees*")),
    )

    random.shuffle(filenames)
    print("reading species tree")
    nw2sid = pd.read_sql_table(
        "species_trees",
        con=conn,
        schema=schema,
        columns=["sid", "newick"],
        index_col="newick",
    )["sid"].to_dict()

    csize = int(5000 / nprocs)
    sim_models = ("LG", "WAG")
    infer_models = ("PROTCATLG", "PROTCATWAG")

    # TODO check file size, keep recmap in memory
    stree = ds = smodel = imodel = None
    # print(filenames)
    with Pool(nprocs) as p:
        for param in p.map(parse_filename, filenames):
            written = 0
            if param.stree != stree:
                stree = param.stree
                sid = nw2sid[stree]

                processor = partial(process_line, sid=sid)

            newick_trees = list(utils.TreeFile(param.filename))
            c = p.map(
                processor,
                enumerate(newick_trees),
                chunksize=csize
            )
            d = pd.DataFrame(c)
            if param.smodel != smodel:
                smodel = param.smodel
                # TODO get new scf
                if nblocks > 1 and param.ds != ds:
                    ds = param.ds
                    rfilename = dirpath / "seqs" / \
                        f"{param.stree_str}_{smodel}.permutations.npy"

                    try:
                        recmap = utils.np.load(rfilename)
                        recfile = True
                    except FileNotFoundError:
                        recfile = False
            if nblocks > 1:
                d.set_index("tree_no", inplace=True)
                d["ds_no"] = ds
                if recfile:
                    # since tree_no index is 1-based
                    try:
                        d["tree_no"] = recmap[ds, :, d.index - 1].tolist()
                    except IndexError as e:
                        print(e, "skipping tree_no")
            # elif heterotachy:
            #     d["ds_no"] = d["tree_no"] // 250
            d["sid"] = sid
            d["sim_model"] = param.smodel
            d["infer_model"] = param.imodel
            d["theta"] = 0.01
            d["seq_length"] = slength
            d["infer_engine"] = "raxml"
            d["sim_engine"] = "seq-gen"
            d["seq_type"] = "AA"

            try:
                written = write_rows_to_sql(
                    d,
                    conn,
                    schema=schema,
                    table=inferred_table,
                )
            except (UniqueViolation, IntegrityError) as e:
                print('Error: most likely from duplicate gene trees.',
                      param,
                      d.columns, sep='\n')
            except Exception as e:
                print('error writing', e, d.columns)
                print(param)
            finally:
                if written > 0:
                    print("wrote {} inferred gene tree datasets to sql".format(written))
                gc.collect()
    print("finished updating inferred gene trees")
