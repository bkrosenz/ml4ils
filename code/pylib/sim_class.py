from __future__ import print_function
import cProfile, pstats
import subprocess, sqlite3, argparse, sys, re, shutil
from operator import mul
from math import ceil
from contextlib import contextmanager, closing, ExitStack
from functools import partial,reduce
from io import BytesIO, StringIO
from itertools import chain,product,islice
from glob import glob
from tempfile import mkdtemp
from utils import utils as u
from os import path
from multiprocess import Pool, Manager
from time import time
from Bio import Phylo, AlignIO



outgroup_name = '4'

rx=re.compile('(\.raxml)|(\.counts)|(\.prot)')
clean_filename = lambda s: rx.sub(repl='',string=path.basename(s))

# TODO: add create table entries (id integer primary key, data) to everything so we can save space by using the unique ids as primary keys instead of trees and seqs.  done.
# also have indices to speed up searching:
# CREATE INDEX gt_ind ON geneTrees (ebl,ibl,tout,na,nb,nc,id,tree);
# CREATE INDEX sq_ind ON seqs (sim,scale,cats,length,tree_id,id,align);

try:
    from itertools import imap
    from cStringIO import StringIO
except ImportError:
    # Python 3...
    imap=map
    from io import StringIO

# NOTICE: must pass properly managed db instance to the classes below.
@contextmanager
def dbconnect(dbfile):
    """Per docs.python.org/3/library/sqlite3.html: By default, check_same_thread is True and only the creating thread may use the connection. If set False, the returned connection may be shared across multiple threads. When using multiple threads with the same connection writing operations should be serialized by the user to avoid data corruption. """
    c = sqlite3.connect(dbfile,check_same_thread=False,
                        isolation_level = None)
    c.execute("PRAGMA journal_mode=WAL")
    yield c
    c.close()

def eprint(*args, **kwargs):
    """prints to stderr. useful if we ignore stdout"""
    print(*args, file=sys.stderr, **kwargs)
            
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

    def create_table(self):
        if self.overwrite:
            self.db.execute('DROP TABLE IF EXISTS {tn}'.format(tn=self.table))
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
        # with closing(self.db.cursor()) as cur:
        #     cur.execute("SELECT * from {tb} LIMIT 10;".format(tb=table))
        #     print('inserting into', table, "fields", fields)
    

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
            # # get params that have insufficient num sims in db
            # cur.execute('''SELECT *, COUNT(1) as cnt 
            # FROM {tab} AS tab
            # INNER JOIN tmp 
            # ON {join_fields};'''.format(tab=self.table,
            #                             join_fields=join_str)
            # )
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
            # cur.execute('''SELECT {fs} FROM tmp EXCEPT 
            # SELECT {fs} FROM 
            # (SELECT {fs}, count(*) AS c FROM {tab} 
            # GROUP BY {fs}) 
            # WHERE c > {nloci}
            # INTERSECT 
            # SELECT {fs} FROM tmp;'''.format(tab = self.table,
            #                                 fs = field_str,
            #                                 nloci = self.n)
            # )
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
            

                # match_str = ' OR '.join(
                #     ' AND '.join(
                #         '{field}="{val}"'.format(field=f,val=v)
                #         if self.field_types[f]=='TEXT'
                #         else'{field}={val}'.format(field=f,val=v)
                #         for f,v in zip(self.params, vals)
                # )
                #     for vals in params
                # )

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
                    #                for _ in range(ceil(float(query_len)/batchsize)): # we know how many batches
                    batch = cur.fetchmany(batchsize)
                    if not batch:
                        break
                    batches+=1
                    yield batch #chain(*batch)
                
        except: # sqlite3.OperationalError:
            
            # print('fields',selected_fields,'str','SELECT {sf} FROM {tn} WHERE {ms}'.format(
            #                     tn=self.table,
            #                 sf=fields_str,
            #                     ms = match_str
            #                 ))
  #          print('finished',batches,'batches')
        # except: # all other exceptions
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
        
class TreeSim(SQLio):
                        # classastributes
    params = ('ebl', 'ibl', 'na', 'nb', 'nc', 'tout')
    
    def __init__(self, args, dbfile):
        """takes dict of param values"""
        super().__init__(dbfile=dbfile,
                         table_name = 'geneTrees',
                         overwrite = args['overwrite'])


        self.param_values = {f:args[f] for f in self.params}
        
        self.field_types = {k:'FLOAT' for k in (*self.params, *u.taxa_combs)}
        self.field_types.update({'tree':'TEXT', 'tree_num':'INT', 'topology':'INT'})
        self.fields = sorted(self.field_types.keys())

        self.n = args['loci']

        self.create_table()
        
    def simulate(self,pool=None):

        param_names, params_to_simulate = self.set_minus(self.param_values)

        # print ('names',param_names, 'params',params_to_simulate)
        # exit()

        jobs=[]
        for config in params_to_simulate:
            ns = argparse.Namespace(**dict(zip(param_names,config)))
            ns.schema = self.fields
            ns.nreps = self.n
            ns.prog = self.PROG
            jobs.append( pool.apply_async(self.worker,
                                          (ns,),
                                          callback = self.insert_many) ) # apply needs an iterable[+mapping]
        try:
            output = ''
            for job in jobs:
                output  = job.get()
        except:
            eprint( output )
            raise
#        self.insert_many(sims)

    @staticmethod
    def worker(args):
        """"takes namespace obj with fields: ebl,ibl,tout,tmpdir,nreps"""
        nsamps=1
        total_samps=1+3*nsamps
        ta=args.ebl # convert to absolute coal times
        tb=args.ibl+ta
        tc=args.tout+tb

        # write out all the files for seq-gen to use
        cmd_str = """{prog}/ms {total_samps} {nreps} -T \
        -I {total_samps} {nsamps} {nsamps} {nsamps} 1 \
        -ej {ta} 2 1 -ej {tb} 3 1 -ej {tc} 4 1 \
        | sed -n 5~3p \
        ;""".format(
            prog=args.prog,
            nreps=args.nreps,
            nsamps=nsamps,
            total_samps=total_samps,
            ta=ta, tb=tb, tc=tc)
    
        with subprocess.Popen(cmd_str,
                              shell=True,
                              stdout=subprocess.PIPE,
                              stderr=None, #subprocess.STDOUT,
                              bufsize=-1) as ps:
            mstrees, stderr = ps.communicate()

        # output
        d = dict.fromkeys(args.schema)
        d.update({k:v for k,v in vars(args).items() if k in d})

        def value_generator(trees):
            tree_num = 0
            for t in Phylo.parse(StringIO(mstrees.decode()),format='newick'):
                d['tree'] = t.format('newick') #tree_str
 #               s=StringIO(tree_str)
                # t = Phylo.read(s, 'newick')
                t.root_with_outgroup({'name':outgroup_name})
                d['tree_num'] = tree_num
                d['topology'] = u.get_topo(t)
                d.update( u.get_dist_mat(t) )
                tree_num+=1
                yield [d[k] for k in args.schema]
        results = value_generator(mstrees)
        return list(results)
        

class SeqSim(SQLio):
    params = ('sim','scale','cats','length','tree_id') # tree param will be added later
    
    def __init__(self, args, tree_simulator, dbfile):
        """takes dict of param values"""
        super().__init__(dbfile=dbfile,
                         table_name = 'seqs',
                         overwrite = args['overwrite'])

        
        self.param_values = {f:args[f] for f in self.params if f in args}
        
        self.field_types = {'scale':'FLOAT',
                            'sim':'TEXT','tree_id':'INT',
                            'type':'TEXT',
                            'length':'INT','cats':'INT',
                            'align':'TEXT'}
        
        self.fields = sorted(self.field_types.keys())
        self.n = 1 # 1 simulate seq per gene tree
        self.tree_simulator = tree_simulator
        self.create_table()
        
    def simulate(self, pool=None):
        jobs=[]
#        print('params',self.tree_simulator.param_values)
        # use an iterator to limit calls to seq-gen at 10000 seqs
        for tree_batch in self.tree_simulator.batch_iter(['id','tree']):
            ids2trees = dict(tree_batch)
            # if not ids2trees:
            #     break

            new_params = self.param_values
            new_params['tree_id'] = ids2trees.keys()
            #OR: new_params['tree_id'] = list(zip(*trees_and_ids)) # warning: this will consume lotsa memory
            param_names, params_to_simulate =  self.set_minus(
                new_params
            )
#            print('new params',params_to_simulate)
            # todo: replace namespace w/ dict throughout for greater efficiency
            for config in params_to_simulate:
                ns = argparse.Namespace(**dict(zip(param_names,config)))
#                print (ns)
                ns.tree = ids2trees[ns.tree_id]
                ns.schema = self.fields
                ns.nreps = self.n
                ns.prog = self.PROG
                jobs.append( pool.apply_async(self.worker,
                                              (ns,),
                                              callback = self.insert_many) ) # apply needs an iterable[+mapping]
        try:
            output = ''
            for job in jobs:
                output  = job.get()
        except:
            eprint( output )
            raise
#        self.insert_many(sims)

    @staticmethod
    def worker(args):
        """"takes namespace obj with fields: sim, scale, length, cats, tree
        simulates a SINGLE alignment"""
        try:
            t = args.tree # unused.  
        except:
            eprint('in worker',args)
            raise
            
        # output in phylip format (BioPython only supports mult aligns in phylip)
        # num of trees is limited by OS line limit.
        # TODO: need to mod seq-gen again so it doesn't wait for user input, but blocks on pipe
        seqgencmd="""echo '{tree}' | {prog}/seq-gen-stdin -op  -q \
        -m{model} -g{cats} -s{scale} -l{seqlen} \
        ;""".format(
            prog=args.prog,
            model = args.sim,
            scale = args.scale,
            seqlen = args.length,
            cats = args.cats,
            tree = args.tree
            
        )

        with subprocess.Popen(seqgencmd,
                              shell=True,
                              stdout=subprocess.PIPE, #stderr=subprocess.STDOUT,
                              stderr=None, #subprocess.STDOUT,
                              bufsize=-1) as ps:
            phylip_align, stderr = ps.communicate(args.tree)

        d = dict.fromkeys(args.schema)
        d.update({k:v for k,v in vars(args).items() if k in d})
        d['type'] = 'dna' if d['sim'] in ('HKY', 'F84', 'GTR') else 'protein'
            
        def value_generator(align):
            # just using AlignIO to split the align, there's prob a faster way
            for align in AlignIO.parse(StringIO(align.decode()),format='phylip'): 
                d['align'] = align.format('phylip')
                yield [d[k] for k in args.schema]

        results = value_generator(phylip_align)
        return list(results)

class TreeInf(SQLio):
    params = ('model','align_id')
    
    def __init__(self, args, seq_simulator, dbfile):
        """takes dict of param values"""
        super().__init__(dbfile=dbfile,
                         table_name = 'raxml',
                         overwrite = args['overwrite'])
        
        self.param_values = {f:args[f] for f in self.params if f in args}

        self.field_types = {k:'FLOAT' for k in u.taxa_combs}
        self.field_types.update({'model':'TEXT',
                                 'align_id':'INT',
                                 'topology':'INT',
                                 'tree':'TEXT'})
        
        self.fields = sorted(self.field_types.keys())

        self.seq_simulator = seq_simulator
        self.raxml = args['raxml']
        self.n = 1 # assume we only ever make 1 tree per seq; could also store top k ML trees
        self.create_table()
        
    def simulate(self, pool=None):
        jobs=[]
        new_params = self.param_values
        # batchize must be one, or else we have to unzip -> align,align id

        print('inferring...')
        for batch in self.seq_simulator.batch_iter(['id','align'],batchsize=1):
#            id2align = dict(batch)
            # more efficient: have a single call to batch_iter, get all aligns, then submit 1-by-1
            align_id,align = batch[0]
            new_params['align_id'] = [align_id] #id2align.keys()[0]
            param_names, params_to_simulate =  self.set_minus(
                new_params
            )
#            print('align',align_id,'params',param_names, params_to_simulate[:10])
            for config in params_to_simulate:
                ns = argparse.Namespace(**dict(zip(param_names,config)))
                ns.schema = self.fields
                ns.align = align # id2align[ns.align_id]
                ns.nreps = self.n
                ns.prog = self.PROG
                ns.raxml = self.raxml
                jobs.append( pool.apply_async(self.worker,
                                              (ns,),
                                              callback = self.insert_many) ) # apply needs an iterable[+mapping]
        try:
            output = ''
            for job in jobs:
                output  = job.get()
        except:
            eprint( output )
            raise
#        self.insert_many(sims)

    @staticmethod
    def worker(args):
        """"takes namespace obj with fields: model, seqs. 
        NOTE: each worker only takes ONE alignment."""
        try:
            dirpath = mkdtemp()
            print('tmpdir:',dirpath)
            seqfile = 'seqs.phy'
            with open(path.join(dirpath,seqfile),'w') as f:
                f.write(args.align)
            raxmlcmd = """{prog}/{raxml_binary} -s {d}/{sf} -w {d} -n raxml \
                -m {model} \
                -p {seed} \
                -o {outgrp} \
            ;""".format(
                prog = args.prog,
                sf = seqfile,
                raxml_binary = args.raxml,
                d = dirpath,
                model = args.model,
                outgrp = outgroup_name,
                seed = int(time())
            )

            with subprocess.Popen(raxmlcmd, 
                                  shell=True,
                                  stdout=subprocess.PIPE,
                                  stderr=None, #subprocess.STDOUT,
                                  bufsize=-1) as ps:
                raxml_output = ps.communicate()[0]
            if not glob(path.join(dirpath,'RAxML_result.raxml')):
                raise ValueError('raxml {rx} did not generate any output'.format(rx=args.raxml))
            with open(path.join(dirpath,'RAxML_result.raxml')) as output:
                t = Phylo.read(output, 'newick')
                t.root_with_outgroup({'name':outgroup_name}) 
                
            d = dict.fromkeys(args.schema)
            d['tree'] = t.format('newick') 
            d['topology'] = u.get_topo(t)
            d.update( u.get_dist_mat(t) )
            d.update({k:v for k,v in vars(args).items() if k in d})
            results = [d[k] for k in args.schema]
            
            return (results,)

        except:
            eprint ('error in raxml:',dirpath,args)
            with open(path.join(dirpath,'RAxML_result.raxml')) as output:
                eprint (output.read())
            raise
        finally:
            shutil.rmtree(dirpath)

        
def main(args):
    arg_dict = vars(args)
    print('param combs:',reduce(mul,(len(s) for s in arg_dict.values() if isinstance(s,list))),
          'nloci:',arg_dict['loci']
    )
              
    try:
        
        pool = Pool(2*args.procs-2) # try this out
        
            # must be called in this order, to ensure we have trees to sim and seqs to analyze...
        pr = cProfile.Profile()
        pr.enable()

        tree_sim = TreeSim(arg_dict, dbfile=args.out)
        tree_sim.simulate(pool)
        print('finished getting trees')
        seq_sim = SeqSim(arg_dict, tree_sim, dbfile=args.out)
        seq_sim.simulate( pool)
        print('finished getting seqs')
        tree_inf = TreeInf(arg_dict, seq_sim, dbfile=args.out)
        tree_inf.simulate(pool)

    except:
        raise
    finally:
        tree_sim.close()
        seq_sim.close()
        tree_inf.close()
        pool.close()
        pool.join()
        pr.disable()
        s = StringIO()
        ps = pstats.Stats(pr, stream=s)
        ps.sort_stats('cumulative').print_stats(20)
        eprint(s.getvalue())


    # wrapper
if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Process some data.')
    parser.add_argument('--procs',type=int,default=4,
                        help='number of processes.  NOTE: seqgen is IO-bound, so this doesnt make a big difference.')
    # TODO: add option to just pass a species tree (arbitrary num taxa). track num taxa.
    # TODO: allow non-ultrametric species tree.  This means seqgen needs to output ancestral node states and we revert mutations IAW the mutation model
    parser.add_argument('--ebl',nargs='+',type=float,
                        help='length of branch subtending MRCA(A,B)')
    parser.add_argument('--ibl',nargs='+',type=float,
                        help='length of branch between MRCA(A,B,C) and MRCA(A,B)')
    parser.add_argument('--tout',nargs='+',type=float,
                        help='length of branch between MRCA(A,B,C,O) and MRCA(A,B,C)')
    parser.add_argument('--loci',type=int,default=10000,
                        help='number of replications (independent loci).  Recommended to simulate a LOT and subsample.')
    parser.add_argument('--na',nargs='+',type=int,default=[1],
                        help='N_e for pop A')
    parser.add_argument('--nb',nargs='+',type=int,default=[1],
                        help='')
    parser.add_argument('--nc',nargs='+',type=int,default=[1],
                        help='')
    
    parser.add_argument('--sim','-s',nargs='+',type=str,
                        choices=['HKY', 'F84', 'GTR', 'JTT', 'WAG', 'PAM', 'BLOSUM', 'MTREV', 'CPREV45', 'MTART', 'GENERAL'],
                            help='substitution model used for simulation (seqgen)'
    )
    parser.add_argument('--length',type=int,nargs='+',default=[1000],
                        help='length of each locus.')
    parser.add_argument('--scale',type=int,nargs='+',default=[1],
                        help='multiple to scale trees by for seq sims.')
    parser.add_argument('--cats',type=int,nargs='+',default=[5],
                        help='number of gamma rate categories.')

    parser.add_argument('--raxml',type=str,
                        help='raxml binary file (depends on os)',required=True)
    parser.add_argument('--model','-m',nargs='+',type=str,
                        choices = ['GTRCAT','GTRGAMMA',
                                   'PROTCATDAYHOFFF', 'PROTCATDCMUTF', 'PROTCATJTTF', 'PROTCATMTREVF', 'PROTCATWAGF', 'PROTCATRTREVF', 'PROTCATCPREVF', 'PROTCATVTF', 'PROTCATBLOSUM62F', 'PROTCATMTMAMF', 'PROTCATLGF', 'PROTCATMTARTF', 'PROTCATMTZOAF', 'PROTCATPMBF', 'PROTCATHIVBF', 'PROTCATHIVWF', 'PROTCATJTTDCMUTF', 'PROTCATFLUF', 'PROTCATSTMTREVF', 'PROTCATDUMMYF', 'PROTCATDUMMY2F', 'PROTCATLG4MF', 'PROTCATLG4XF', 'PROTCATPROT_FILEF', 'PROTCATGTR_UNLINKEDF', 'PROTCATGTRF'],
                        help='substitution model used for inference (raxml)'
                        )

    parser.add_argument('--overwrite','-w', action='store_true',
                        help="overwrite existing tables. Otherwise, append.")
    parser.add_argument('--verbose', action='store_true',
                        help="debug (verbose) mode.")
    parser.add_argument('--out','-o',type=str,
                        help='output database filename',required=True)

    args = parser.parse_args()
    print (args)
    if args.overwrite:
        print('cowardly refusing to overwrite db')
        exit(1)
    main(args)
    
