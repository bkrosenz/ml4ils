import subprocess, sqlite3, argparse, sys, re, shutil
from operator import mul
from math import ceil
from contextlib import contextmanager, closing, ExitStack
from functools import partial,reduce
from itertools import chain,product,islice
from utils import utils as u
from os import path
from time import time

class SQLio:
    PROG="/N/dc2/projects/bio_hahn_lab/soft/bin" # accessible from any IU system
    
    def __init__(self,dbfile,table_name,overwrite=False):
        """ create table, set table name attribute"""
        self.db=sqlite3.connect(dbfile,check_same_thread=False,
                        isolation_level = None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.table = table_name
        self.overwrite = overwrite
    
        return self

    def get_key_str(self):
        return ','.join(self.fields)
    
    def get_schema(self):
        if not self.field_types:
            raise TypeError("must define keys before setting a schema")
        return ', '.join( ' '.join(kv) for kv in self.field_types.items() )

    def create_table(self,foreign_key=None,parent=None):
        if self.overwrite:
            self.db.execute('DROP TABLE IF EXISTS {tn}'.format(tn=self.table))
        if foreign_key is not None:
            self.db.execute('''CREATE TABLE IF NOT EXISTS 
            {tn} (id integer primary key, {fields},
            FOREIGN KEY ({fk}) REFERENCES {p}({fk});'''.format( tn = self.table,
                                                                fields = self.get_schema(),
                                                                fk = foreign_key,
                                                                p = parent )
            )
        else:
            self.db.execute('''CREATE TABLE IF NOT EXISTS 
            {tn} (id integer primary key, {fields});'''.format( tn=self.table,
                                                                fields=self.get_schema())
            )


    def insert_many(self,values,commit=True):
        self.insert_into(self.table,self.fields,values)
        if commit:
            self.db.commit()
    
    def insert_into(self,table,fields,vals,cur=None):
        """if no cursor is passed, open a new cursor and AUTOMATICALLY COMMIT AFTER POPULATING THE TABLE.
        If a cursor is passed, just let the caller clean it up."""

        field_str = ','.join(fields)
        q_str = ','.join(['?']*len(fields))

        self.db.execute('''CREATE TABLE IF NOT EXISTS 
        {tn} (id integer primary key, {fields});'''.format( tn=table,
                                                            fields=field_str)
        )
        self.db.executemany(
            'INSERT INTO {tb} ({fs}) VALUES ({ks});'.format(tb=table,
                                                            fs=field_str,
                                                            ks=q_str),
            vals
        )

    def set_intersect(self,value_dict):
        '''find param combs which haven't been simulated yet.
        return '''

        fields=value_dict.keys()
        field_str = ','.join(fields)
        in_str = ' AND '.join(field+' IN (' + ','.join(value_dict.values()) + ')' for field in fields)
        try:
            cur = self.db.cursor()
                
            cur.execute('''SELECT {fs} FROM 
            (SELECT {fs}, count(*) AS c FROM {tab} 
            GROUP BY {fs}) 
            WHERE c > {nloci}
            AND WHERE {instr};'''.format(tab = self.table,
                                fs = field_str,
                                nloci = self.n,
                                instr = in_str)
            )
            param_intersect =  cur.fetchall()
            print ('found intersect:',param_intersect[:10])
        except:
            raise
        finally:
            # clean up
            #self.db.rollback() # only with a tmp 
            cur.close()
        return fields,param_intersect
    
    def set_minus(self,value_dict,return_fields=None):
        '''find param combs which haven't been simulated yet.
        return '''

        fields = value_dict.keys()
        field_str = ','.join(fields)
        try:
            cur = self.db.cursor()
            cur.execute('''DROP TABLE IF EXISTS tmp;''')
            self.insert_into('tmp',
                             fields = fields,
                             vals = product(*value_dict.values()),
                             cur = cur)

                # get relative complement
            cur.execute('''SELECT {fs} FROM tmp EXCEPT 
            SELECT {fs} FROM 
            (SELECT {fs}, count(*) AS c FROM {tab} 
            GROUP BY {fs}) 
            WHERE c >= {nloci};'''.format(tab = self.table,
                                            fs = field_str,
                                            nloci = self.n)
            )
            new_params =  cur.fetchall()
        except:
            eprint('valdict',value_dict)
            cur.execute('select * from tmp;')
            eprint(cur.fetchmany(100))
            raise
        finally:
            cur.execute('''DROP TABLE IF EXISTS tmp;''')
            cur.close()
            
        return fields,new_params #+ counts

    def batch_iter(self, selected_fields, batchsize = 10000, query_len = 500):
        """get all sims in db with these params.
        batchsize = # records to return
        query_len = # AND expressions (param combs) in query string"""
        
        parameters = product(*self.param_values.values())
        fields_str = ','.join(selected_fields)
        query_str = ' AND '.join(
                '{field}=?'.format(field=f)
                for f in self.params #,v in zip(self.params, vals)
        )
        
        try:
            batches = 0
            cur = self.db.cursor()
            while True:
                params = list(islice(parameters, query_len))
                
                if not params:
                    break

                match_str = ' OR '.join(
                    query_str
                    for _ in range(len(params))
                )

                param_list = [v for vals in params for v in vals]
            
                # use param substitution to handle ;'s in newick strings
                cur.execute(
                    'SELECT {sf} FROM {tn} WHERE {ms}'.format(
                        tn=self.table,
                        sf=fields_str,
                        ms = match_str
                    ),
                    param_list
                )

                while True:
                    batch = cur.fetchmany(batchsize)
                    if not batch:
                        break
                    batches+=1
                    yield batch #chain(*batch)
                
        except: # sqlite3.OperationalError:
            print('params:',params,'sel',selected_fields,'pnames',self.params)
            raise
        finally:
            cur.close()

    def generate(self):
        pass

    def close(self):
        with self.db as cur:
            cur.execute("PRAGMA wal_checkpoint(PASSIVE)")
        self.db.close()

        
class TreeTable(SQLio):
                        # classastributes
    def __init__(self, tree_config, dbfile):
        """takes dict of param values"""
        super().__init__(dbfile = dbfile,
                         table_name = 'geneTrees',
                         overwrite = args['overwrite'])

        self.field_types = {}
        self.field_types.update(tree_config.cov_schema)
        self.field_types.update({'tree':'TEXT', 'tree_num':'INT', 'topology':'INT'})
        self.fields = sorted(self.field_types.keys())

        self.n = args['loci']

        self.create_table()
