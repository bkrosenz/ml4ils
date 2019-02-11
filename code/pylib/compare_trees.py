from glob import glob

# plotting
from sqlalchemy.orm import load_only,sessionmaker
from sqlalchemy import Table, Column, Integer, String, MetaData, ForeignKey, Float, Sequence, create_engine
import sqlalchemy as sa
from sqlalchemy.sql.expression import func
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.schema import PrimaryKeyConstraint,UniqueConstraint

import matplotlib
matplotlib.use('Agg')
from itertools import chain,islice
import matplotlib.pyplot as plt
from collections import defaultdict,Counter
from io import StringIO
from functools import partial,reduce
from operator import add
import argparse,re,gc,pickle
from ete3 import Tree
from scipy.sparse import csr_matrix, triu
from os import path,walk,remove
from scipy import stats
from numba import njit,vectorize
from numba.types import *
import pandas as pd
import numpy as np
import utils
import h5py
from joblib import dump, load, Parallel, delayed
import dask
from dask.diagnostics import ProgressBar
import dask.bag as db
from dask.distributed import Client, LocalCluster, fire_and_forget, progress
# use with distributed to take care of this error https://github.com/dask/distributed/issues/1467
import pandas as pd
pd.options.mode.chained_assignment = None
import dask.array as da
import dask.dataframe as dd
from sqlalchemy.orm import load_only

def main():
tree_config=utils.TreeConfig(leafnames=range(1,5),
                                 outgroup=4,
                                 include_outgroup=True,
                                 subtree_sizes=[4])

#### start processing
algnames =  [a1+'_'+a2 for a1, a2 in utils.combr(('wag','lg'),2)]

# col names for sql db
cov_cols = next(tree_config.cov_iter(range(1,tree_config.subtree_sizes[0]+1)))

metadata = MetaData(bind=engine)
metadata.reflect()

Session = sessionmaker(bind=engine)

session=Session()
    q=session.query(
        'i_'+c-.label('count'), col,table.c.sid
    ).group_by(col, table.c.sid) # redundant
