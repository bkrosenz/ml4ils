# Alternative to_sql() *method* for DBs that support COPY FROM
import csv
from functools import partial, reduce
from getpass import getpass
from io import StringIO
from operator import add, iadd
from typing import Callable

import psycopg2
from sqlalchemy import MetaData, Table, bindparam, create_engine, func, text
from sqlalchemy.engine.url import URL
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext import baked
from sqlalchemy.orm import load_only, sessionmaker
from sqlalchemy.sql import select

from utils import compose, pd


def write_rows_to_sql(df: pd.DataFrame,
                      conn,
                      schema: str,
                      table: str = 'gene_trees',
                      n=10):
    """very slow! writes batches of n from a df to the table, ignoring duplicates"""
    wrote = 0
    # for _, row in df.iterrows():
    for i in range(0, len(df), 2):
        try:
            # row.to_frame().T
            df[i:i+n].to_sql(
                table,
                conn,
                schema=schema,
                method='multi',
                if_exists='append',
                index=False)
            wrote += n
        except IntegrityError as e:
            # print(e)
            continue
    return wrote


def psql_insert_copy(table, conn, keys, data_iter):
    """
    Execute SQL statement inserting data

    Parameters
    ----------
    table : pandas.io.sql.SQLTable
    conn : sqlalchemy.engine.Engine or sqlalchemy.engine.Connection
    keys : list of str
        Column names
    data_iter : Iterable that iterates the values to be inserted
    """
    # gets a DBAPI connection that can provide a cursor
    dbapi_conn = conn.connection
    with dbapi_conn.cursor() as cur:
        s_buf = StringIO()
        writer = csv.writer(s_buf)
        writer.writerows(data_iter)
        s_buf.seek(0)

        columns = ', '.join('"{}"'.format(k) for k in keys)
        if table.schema:
            table_name = '{}.{}'.format(table.schema, table.name)
        else:
            table_name = table.name

        sql = 'COPY {} ({}) FROM STDIN WITH CSV'.format(
            table_name, columns)
        cur.copy_expert(sql=sql, file=s_buf)


def make_session(args):
    url = URL("postgresql", username=args.user, host="localhost",
              database=args.db)
    engine = create_engine(url,
                           max_overflow=0,
                           connect_args={"timeout": 3000,
                                         'pool_timeout': 3000})
    conn = engine.connect()
    Session = sessionmaker(bind=engine)
    session = Session()
    if hasattr(args, "schema"):
        metadata = MetaData(bind=engine, schema=args.schema)
    else:
        metadata = MetaData(bind=engine)
    metadata.reflect(views=True)
    return (session, metadata, conn)


def filter_table_by(table, **kwargs):
    """returns a func that adds filters for all kwargs to a query"""

    def f(q):
        for k, v in kwargs.items():
            q = q.filter(table.c[k] == v)
        return q

    return f


class Baker(object):
    """wrapper for bakery which can be serialized"""

    def __init__(self, initial_func, yieldper=None):
        """initial func must take a session object and return a query"""
        self.initial = initial_func
        self.yield_per = (
            lambda x: x if yieldper is None else lambda q: q.yield_per(
                yieldper)
        )
        self.funcs = []

    def add_func(self, func):
        """add a func which takes a query and returns another query"""
        self.funcs.append(func)
        return self

    def __iadd__(self, other):
        """add a func in-place"""
        self.add_func(other)
        return self

    def __add__(self, other):
        """make a copy and add a func"""
        new = self.copy()
        new.add_func(other)
        return new

    def copy(self):
        new = Baker(self.initial)
        new.funcs = self.funcs.copy()
        return new

    def to_statement(self, session):
        """returns Baked.Result if params, else SQLAlechemy statement.  
        must call statement"""
        # TODO: clean up
        bakery = baked.bakery()
        return (
            reduce(iadd, self.funcs, bakery(self.initial)
                   ).to_query(session).statement
        )

    def compile(self, session, **params):
        """return result obj if params specified"""
        bakery = baked.bakery()
        q = reduce(iadd, self.funcs +
                   [self.yield_per], bakery(self.initial))(session)
        if params:
            return q.params(**params)
        else:
            print("q:", type(q))
            return q

    def __repr__(self):
        return str(self.initial) + " " + str(self.funcs)


class SlimBaker(Baker):
    """functional syntax of baked objects, but returns
    a sqlalchemy.sql.expression.Selectable object"""

    def __init__(self, initial_func, yieldper=None):
        """initial func must take a session object and return a query"""
        self.func = initial_func
        self.yield_per = (
            lambda x: x if yieldper is None else lambda q: q.yield_per(
                yieldper)
        )

    def add_func(self, func):
        """add a func which takes a query and returns another query"""
        self.func = compose(func, self.func)
        return self

    def __iadd__(self, other):
        """add a func in-place"""
        self.add_func(other)
        return self

    def __add__(self, other):
        """make a copy and add a func"""
        new = self.copy()
        new.add_func(other)
        return new

    def copy(self):
        return Baker(self.func)

    def to_statement(self, session, **params):
        """returns Baked.Result if params, else SQLAlechemy statement.  
        must call statement"""
        # TODO: clean up
        return self.yield_per(self.func(session)).statement

    def compile(self, session, **params):
        """return result obj if params specified"""
        q = self.func(session)
        if params:
            q = q.params(**params)
        return self.yield_per(q)


def make_connection(schema):
    with open("/N/u/bkrosenz/BigRed3/.ssh/db.pwd") as f:
        password = f.read().strip()

    session, conn = make_session_kw(
        username="bkrosenz_root",
        password=password,
        database="bkrosenz",
        schema=schema,
        port=5444,
        host="10.79.161.8",
        with_metadata=False,  # sasrdspp02.uits.iu.edu'
    )
    return conn


def prepare_bootstrap(conn, schema, table, columns):
    s = '''prepare sample(int) as
        select {cols} from {table} 
            order by random()
            limit $1;'''.format(
        cols=','.join(columns),
        table=table
    )
    conn.execute('set search_path to {};'.format(schema))
    conn.execute(s)


def prepare(conn):
    """TABLESAMPLE is not completely random.
    Sample > target number, then randomly shuffle and limit.
    for testing on rec/nonrec - only pick gene trees inferred from correct substitution model.
    Disallows model misspecification (LG->LG and WAG->WAG) except in sample_model_nonrec statement.
    """
    # args: length,ngenes

    nonrec_prepared = '''
    
    prepare sids_nonrec(int,int) as
    select s.* from
        (select sid from nonrec_inferred 
              where seq_length=$1 
         group by sid having count(1)>=$2) i
    natural join 
        (select ebl,ibl,sid from species_trees) s;

    prepare sample_sid_nonrec(int,int,int) as
    select i.pdist,
        i.tid,
        c.top_1,
        c.top_2,
        c.top_3,
        c.nsites,
        i.seq_length,
        i.randomcolumn
    from
                    (select r.pdist,
                            r.tid,
                            r.tree_no,
                            r.sim_model,
                            r.seq_length,
                            r.randomcolumn
                        from nonrec_inferred r
                        -- tablesample system (90)
                        where           r.seq_length=$2
                                        and position(r.sim_model in r.infer_model)>0
                                        and r.sid=$1) i 
    natural left join
                    (select s.top_1,
                            s.top_2,
                            s.top_3,
                            s.nsites,
                            s.tree_no,
                            s.sim_model
                        from nonrec_scf s
                        where           s.seq_length=$2
                                        and s.sid=$1) c
    order by randomcolumn
    limit $3;

    prepare sample_sid_model_nonrec(int,int,text,text,int) as
    select i.pdist,
        i.tid,
        c.top_1,
        c.top_2,
        c.top_3,
        c.nsites,
        i.seq_length,
        i.randomcolumn
    from
                    (select r.pdist,
                            r.tid,
                            r.tree_no,
                            r.sim_model,
                            r.seq_length,
                            r.randomcolumn
                        from nonrec_inferred r
                        -- tablesample system (80)
                        where           r.seq_length=$2
                                        and r.sim_model=$3
                                        and position($4 in r.infer_model)>0
                                        and r.sid=$1) i 
    natural left join
                    (select s.top_1,
                            s.top_2,
                            s.top_3,
                            s.nsites,
                            s.tree_no
                        from nonrec_scf s
                        where           s.seq_length=$2
                                        and s.sim_model=$3
                                        and s.sid=$1) c
    order by randomcolumn
    limit $5;'''

    rec_prepared = '''prepare sids_rec(int,int) as
    select s.* from
        (select sid from rec_inferred 
              where seq_length=$1 
         group by sid having count(1)>=$2) i
    natural join 
        (select ebl,ibl,sid from species_trees) s;

    prepare sample_sid_blocks_rec(int,int,int,int) as
    select i.pdist,
        i.tid,
        c.top_1,
        c.top_2,
        c.top_3,
        c.nsites,
        i.seq_length,
        i.randomcolumn
    from
                    (select r.pdist,
                            r.tid,
                            r.tree_no,
                            r.sim_model,
                            r.ds_no,
                            r.seq_length,
                            r.randomcolumn
                        from rec_inferred r
                        -- tablesample system (70)
                        where           r.seq_length=$2
                                        and  position(r.sim_model in r.infer_model)>0
                                        and r.sid=$1
                                        and array_length(r.tree_no,1)=$3) i 
    natural left join
                    (select s.top_1,
                            s.top_2,
                            s.top_3,
                            s.nsites,
                            s.tree_no,
                            s.sim_model,
                            s.ds_no
                        from rec_scf s
                        where           s.seq_length=$2
                                        and s.sid=$1) c
    order by randomcolumn
    limit $4;
    
    prepare sample_sid_rec(int,int,int) as
    select i.pdist,
        i.tid,
        c.top_1,
        c.top_2,
        c.top_3,
        c.nsites,
        i.seq_length,
        i.randomcolumn
    from
                    (select r.pdist,
                            r.tid,
                            r.tree_no,
                            r.sim_model,
                            r.ds_no,
                            r.seq_length,
                            r.randomcolumn
                        from rec_inferred r
                        -- tablesample system (70)
                        where           r.seq_length=$2
                                        and  position(r.sim_model in r.infer_model)>0
                                        and r.sid=$1) i 
    natural left join
                    (select s.top_1,
                            s.top_2,
                            s.top_3,
                            s.nsites,
                            s.tree_no,
                            s.sim_model,
                            s.ds_no
                        from rec_scf s
                        where           s.seq_length=$2
                                        and s.sid=$1) c
    order by randomcolumn
    limit $3;'''
    conn.execute('set search_path to sim4;')
    conn.execute(nonrec_prepared)
    conn.execute(rec_prepared)


def prepare_heterotachy(conn):
    '''Disallows model misspecification.  Only one seq length condition.'''

    # args: length,ngenes

    prepared = '''prepare sids(int) as
    select s.* from
        (select sid from gene_trees_heterotachy250 
         group by sid having count(1)>=$2) i
    natural join 
        (select ebl,ibl,sid from species_trees) s;

    prepare sample_sid_model(int,text,text,int) as
    select i.pdist,
        i.tid,
        c.top_1,
        c.top_2,
        c.top_3,
        c.nsites,
        i.seq_length
    from
                    (select r.pdist,
                                r.tid,
                                r.tree_no,
                                r.seq_length,
                                r.sim_model,
                                r.randomcolumn
                             from heterotachy_inferred r 
                             --tablesample system (90) 
                             where r.sim_model=$2
                                        and position($3 in r.infer_model)>0
                                        and r.sid=$1
                            order by r.randomcolumn) i 
    natural left join
                    (select s.top_1,
                            s.top_2,
                            s.top_3,
                            s.nsites,
                            s.tree_no
                        from heterotachy_scf s
                        where s.sid=$1
                            and s.sim_model=$2) c
    limit $4;

    prepare sample_sid(int,int,int) as
    select i.pdist,
        i.tid,
        c.top_1,
        c.top_2,
        c.top_3,
        c.nsites,
        i.seq_length
    from
                    (select r.pdist,
                                r.tid,
                                r.tree_no,
                                r.seq_length,
                                r.sim_model,
                                r.randomcolumn
                             from heterotachy_inferred r 
                             --tablesample system (90) 
                             where r.sid=$1
                             and  position(r.sim_model in r.infer_model)>0
                             order by r.randomcolumn) i 
    natural left join
                    (select s.top_1,
                            s.top_2,
                            s.top_3,
                            s.nsites,
                            s.tree_no,
                            s.sim_model
                        from heterotachy_scf s
                        where         s.sid=$1) c
    limit $3;'''
    conn.execute('set search_path to sim4;')
    conn.execute(prepared)


def prepare_one_rate(conn):
    '''Disallows model misspecification.  Only one seq length condition.'''

    # args: length,ngenes

    prepared = '''prepare sids(int) as
    select s.* from
        (select sid from gene_trees_one_rate 
         group by sid having count(1)>=$2) i
    natural join 
        (select ebl,ibl,sid from species_trees) s;

    prepare sample_sid_model(int,text,text,int) as
    select i.pdist,
        i.tid,
        c.top_1,
        c.top_2,
        c.top_3,
        c.nsites,
        i.seq_length
    from
                    (select r.pdist,
                                r.tid,
                                r.tree_no,
                                r.seq_length,
                                r.sim_model,
                                r.randomcolumn
                             from one_rate_inferred r 
                             --tablesample system (90) 
                             where r.sim_model=$2
                                        and position($3 in r.infer_model)>0
                                        and r.sid=$1
                            order by r.randomcolumn) i 
    natural left join
                    (select s.top_1,
                            s.top_2,
                            s.top_3,
                            s.nsites,
                            s.tree_no
                        from one_rate_scf s
                        where s.sid=$1
                            and s.sim_model=$2) c
    limit $4;

    prepare sample_sid(int,int,int) as
    select i.pdist,
        i.tid,
        c.top_1,
        c.top_2,
        c.top_3,
        c.nsites,
        i.seq_length
    from
                    (select r.pdist,
                                r.tid,
                                r.tree_no,
                                r.seq_length,
                                r.sim_model,
                                r.randomcolumn
                             from one_rate_inferred r 
                             --tablesample system (90) 
                             where r.sid=$1
                             and  position(r.sim_model in r.infer_model)>0
                             order by r.randomcolumn) i 
    natural left join
                    (select s.top_1,
                            s.top_2,
                            s.top_3,
                            s.nsites,
                            s.tree_no,
                            s.sim_model
                        from one_rate_scf s
                        where         s.sid=$1) c
    limit $3;'''
    conn.execute('set search_path to sim4;')
    conn.execute(prepared)


def prepare_all_lengths(conn):
    '''Allows model misspecification.'''
    # args: length,ngenes

    nonrec_prepared = '''

    prepare sample_nonrec(numeric) as
    select itrees.pdist,
        itrees.tid,
        scf.top_1,
        scf.top_2,
        scf.top_3,
        scf.nsites,
        itrees.seq_length,
        itrees.randomcolumn,
        strees.ebl,
        strees.ibl
    from
                    (select r.sid,
                            r.pdist,
                            r.tid,
                            r.tree_no,
                            r.sim_model,
                            r.seq_length,
                            r.randomcolumn
                        from nonrec_inferred r
                        tablesample system ($1)
                        where          position(r.sim_model in r.infer_model)>0
                                      ) itrees
    natural left join
                    (select s.sid,
                            s.top_1,
                            s.top_2,
                            s.top_3,
                            s.seq_length,
                            s.nsites,
                            s.tree_no,
                            s.sim_model
                        from nonrec_scf s) scf
    natural inner join 
        (select ebl,ibl,sid from species_trees) strees
    where ebl<300
--    order by randomcolumn limit $1
    ;
    
    prepare sids_nonrec(int,int) as
    select s.* from
        (select sid from nonrec_inferred 
         group by sid having count(1)>=$2) i
    natural join 
        (select ebl,ibl,sid from species_trees) s;

    prepare sample_sid_nonrec(int,int,int) as
    select i.pdist,
        i.tid,
        c.top_1,
        c.top_2,
        c.top_3,
        c.nsites,
        i.seq_length,
        i.randomcolumn
    from
                    (select r.pdist,
                                r.tid,
                                r.tree_no,
                                r.seq_length,
                                r.sim_model,
                                r.randomcolumn
                             from nonrec_inferred r 
                             --tablesample system (90) 
                             where r.sid=$1) i 
    natural left join
                    (select s.top_1,
                            s.top_2,
                            s.top_3,
                            s.nsites,
                            s.tree_no,
                            s.sim_model
                        from nonrec_scf s
                        where         s.sid=$1) c
    order by randomcolumn
    limit $3;

    prepare sample_sid_model_nonrec(int,text,text,int) as
    select i.pdist,
        i.tid,
        c.top_1,
        i.seq_length,
        c.top_2,
        c.top_3,
        c.nsites,
        i.randomcolumn
    from
                    (select r.pdist,
                            r.tid,
                            r.seq_length,
                            r.tree_no,
                            r.sim_model,
                            r.randomcolumn
                        from nonrec_inferred r
                        --tablesample system (90)
                        where          r.sim_model=$2
                                        and position($3 in r.infer_model)>0
                                        and r.sid=$1) i 
    natural left join
                    (select s.top_1,
                            s.top_2,
                            s.top_3,
                            s.nsites,
                            s.tree_no
                        from nonrec_scf s
                        where s.sid=$1
                            and s.sim_model=$2) c
    order by randomcolumn
    limit $4;'''

    rec_prepared = '''prepare sids_rec(int,int) as
    select s.* from
        (select sid from rec_inferred 
            group by sid having count(1)>=$2) i
    natural join 
        (select ebl,ibl,sid from species_trees) s;

    prepare sample_sid_blocks_rec(int,int,int,int) as
    select i.pdist,
        i.tid,
        c.top_1,
        c.top_2,
        c.top_3,
        c.nsites,
        i.seq_length,
        i.randomcolumn
    from
                    (select r.pdist,
                            r.tid,
                            r.tree_no,
                            r.seq_length,
                            r.sim_model,
                            r.ds_no,
                            r.randomcolumn
                        from rec_inferred r
                        -- tablesample system (90)
                        where r.sid=$1
                            and array_length(r.tree_no,1)=$3) i 
    natural left join
                    (select s.top_1,
                            s.top_2,
                            s.top_3,
                            s.nsites,
                            s.tree_no,
                            s.sim_model,
                            s.ds_no
                        from rec_scf s
                        where         s.sid=$1) c
    order by randomcolumn
    limit $4;
    
    prepare sample_sid_rec(int,int,int) as
    select i.pdist,
        i.tid,
        c.top_1,
        c.top_2,
        c.top_3,
        i.seq_length,
        c.nsites,
        i.randomcolumn
    from
                    (select r.pdist,
                            r.tid,
                            r.tree_no,
                            r.sim_model,
                            r.ds_no,
                            r.seq_length,
                            r.randomcolumn
                        from rec_inferred r
                        where r.sid=$1) i 
    natural left join
                    (select s.top_1,
                            s.top_2,
                            s.top_3,
                            s.nsites,
                            s.tree_no,
                            s.sim_model,
                            s.ds_no
                        from rec_scf s
                        where         s.sid=$1) c
    order by randomcolumn
    limit $3;'''
    conn.execute('set search_path to sim4;')
    conn.execute(nonrec_prepared)
    conn.execute(rec_prepared)


def make_session_kw(username: str,
                    database: str,
                    port: int,
                    schema='public',
                    host: str = "localhost",
                    password: str = None,
                    statement_prepare: Callable = prepare,
                    with_metadata: bool = True):
    """Uses the IU-hosted DB.  prepares query statements"""
    try:
        if password is None:
            password = getpass()
        url = URL("postgresql",
                  username=username,
                  host=host,
                  password=password,
                  port=port,
                  database=database)
        engine = create_engine(
            url,
            connect_args={'connect_timeout': 5000},
            max_overflow=5)
        conn = engine.connect()
        Session = sessionmaker(bind=engine)
        session = Session()
        statement_prepare(conn)
    except Exception as e:
        print(url)
        raise e
    if with_metadata:
        metadata = MetaData(bind=engine, schema=schema)
        metadata.reflect(views=True)
        return (session, metadata, conn)
    else:
        return (session, conn)
