from __future__ import print_function
from contextlib import contextmanager
from functools import partial
from itertools import chain,product
from glob import glob
import subprocess, sqlite3, argparse, tempfile
from utils import utils as u
import shutil
from os import path
from multiprocess import Pool
#from ete2 import Tree
from Bio import Phylo
import re

rx=re.compile('(\.raxml)|(\.counts)|(\.prot)')
clean_filename = lambda s: rx.sub(repl='',string=path.basename(s))

try:
        from itertools import imap
        from cStringIO import StringIO
except ImportError:
        # Python 3...
        imap=map
        from io import StringIO
        
@contextmanager
def dbconnect(dbfile):
    c = sqlite3.connect(dbfile)
    yield c
    c.close()

def write_to_db(tree_generator,dbfile,table_name):
    with dbconnect(dbfile) as c:
        g=tree_generator.generate()
        c.execute('''CREATE TABLE IF NOT EXISTS 
        {tn} ({fields});'''.format( tn=table_name,
                                    fields=tree_generator.get_field_str())
        )
        with c:
            c.executemany(
                'INSERT INTO {tn}({fields}) VALUES ({values})'.format( tn=table_name,
                                                                       fields=','.join(tree_generator.keys),
                                                                       values=','.join(['?']*len(tree_generator.keys)) ),
                tree_generator.generate()
            )
        
    
class TreeSet:
    def __init__(self,args,files,true_trees=None):
        self.args = vars(args)
        self.files = files
        self.keytypes = {'tree_num':'INT',
                         'tree':'TEXT',
                         'topology':'INT',
                         'align':'VARCHAR(20)'
        }
        self.keytypes.update({k:'FLOAT' for k in u.taxa_combs})
        if true_trees != None:
            for k in ('model','sim'):
                self.keytypes[k] = 'VARCHAR(200)'
            self.keytypes['seqs']='TEXT'
            self.keytypes['true_tree']='TEXT'
            self.true_trees = true_trees
        else:
            for k in ('ibl','ebl','tout','na','nb','nc'):
                self.keytypes[k] = 'FLOAT'
                
        #store sorted keys
        self.keys = sorted(self.keytypes)

        #populate dict
        self.d = dict.fromkeys(self.keytypes)
        for k in self.d:
            if k in self.args:
                self.d[k]=self.args[k]
        self.d['align'] = 'dna' if self.args['sim'] in ('HKY', 'F84', 'GTR') else 'protein'

    def generate(self):
        if type(self.files)==list:
            return chain(*imap(self.value_generator,self.files))
        else: # a single file handle or fpath str
            return self.value_generator(self.files)
        
    def value_generator(self,fn):
        tree_num = 0
        trees = Phylo.parse(fn, 'newick')
        for t in trees:
            t.root_with_outgroup({'name':'4'})
            self.d['tree_num'] = tree_num
            self.d['tree'] = t.format('newick')
            self.d['topology'] = u.get_topo(t)
            if hasattr(self,'true_trees'):
                self.d['true_tree'] = self.true_trees[tree_num]
            self.d.update( u.get_dist_mat(t) )
            tree_num+=1
            yield [self.d[k] for k in self.keys]

    def get_field_str(self):
        return ', '.join( k+' '+self.keytypes[k] for k in self.keys)


def worker(args):
    # generate trees with ms
    # generate seqs with seq-gen
    # discrete gamma rate with 5 categories

    PROG="/N/dc2/projects/bio_hahn_lab/soft/bin"
    CODEDIR="/N/dc2/projects/bkrosenz/deep_ils/code"

    nsamps=1
    seqlen=1000
    scale=1
    rate_categories = 5
    dirpath = tempfile.mkdtemp()
    total_samps=1+3*nsamps
    ta=args.ebl # convert to absolute coal times
    tb=args.ibl+ta
    tc=args.tout+tb
    
    mscmd = """{prog}/ms {total_samps} {nreps} -T \
    -I {total_samps} {nsamps} {nsamps} {nsamps} 1 \
    -ej {ta} 2 1 -ej {tb} 3 1 -ej {tc} 4 1 | tee {d}/trees \
    |;""".format(
        prog=PROG,
        nreps=args.loci,
        d = dirpath,
        nsamps=nsamps,
        total_samps=total_samps,
        ta=ta, tb=tb, tc=tc)

    with subprocess.Popen(mscmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,bufsize=-1) as ps:
        mstrees, stderr = ps.communicate()

    mstrees = mstrees.decode('ascii')
    
    seqgencmd="""{prog}/seq-gen -m{model} \
     -g{cats} \
     -s{scale} -l{seqlen} {d}/trees \
     | split -l{split_lines} - {d}/seqs. \
    ;""".format(
        prog=PROG,
        d = dirpath,
        model = args.sim,
        scale = scale,
        seqlen = seqlen,
        cats = rate_categories,
        split_lines = 1+total_samps,
    )

    with subprocess.Popen(seqgencmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,bufsize=-1) as ps:
        seqs, stderr = ps.communicate(mstrees)

    # infer trees with raxml
    # ls sorts alphabetically, so this should be same order as trees in treefile

    # is it ok to use same seed for each gene? raxml requires one to be specified
    raxmlcmd = """ls {d}/seqs.* \
        | perl -pe "s/.*\/(seqs.*)/\$1/" \
        | xargs -I@ \
                {prog}/{raxml_binary} -s {d}/@ -w {d} -n @.raxml \
                -m {model} \
                -p {seed} \
    ;""".format(
        prog=PROG,
        raxml_binary = args.raxml,
        d = dirpath,
        model = args.model,
        seed = time()
        
    )
    print(raxmlcmd)
    with subprocess.Popen(raxmlcmd, shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,bufsize=-1) as ps:
        raxml_output = ps.communicate()[0]
        if not glob(path.join(dirpath,'RAxML_bestTree.*.raxml')):
            raise ValueError('raxml {rx} did not generate any output'.format(rx=args.raxml))

     # one filehandle
    write_to_db(TreeSet(args=args,
                        files=StringIO(mstrees)),
                dbfile=args.out,
                table_name='true')

    # separate file for each tree
    print ('mstrees',mstrees,len(mstrees))
    write_to_db(TreeSet(args=args,
                        files = glob(path.join(dirpath,'RAxML_bestTree.*.raxml')),
                        true_trees = mstrees),
                dbfile = args.out,
                table_name='inferred')
                    
    shutil.rmtree(dirpath)
    print("finished",args)


def main(args):
    if args.overwrite:
        with dbconnect(args.out) as c:
            for table_name in ('true','inferred'):
                c.execute('''DROP TABLE IF EXISTS {tn};'''.format(tn=table_name))

    p=Pool(args.procs)
    del args.procs # don't need to pass this to each process
    args.overwrite = [args.overwrite]
    v = vars(args)
    params = v.keys()
    res=[]
    for config in product(*v.values()):
        ns = argparse.Namespace(**dict(zip(params,config)))
#        worker(ns)
        res.append( p.apply_async(worker,(ns,)) ) # apply needs an iterable[+mapping]
    for w in res:
        print(w.get())
    p.close()
    p.join()
    
# wrapper
if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Process some data.')
    parser.add_argument('--procs',type=int,default=4,help='N_e for pop A')
    parser.add_argument('--ebl',nargs='+',type=float,help='length of branch subtending MRCA(A,B)')
    parser.add_argument('--ibl',nargs='+',type=float,help='length of branch between MRCA(A,B,C) and MRCA(A,B)')
    parser.add_argument('--tout',nargs='+',type=float,help='length of branch between MRCA(A,B,C,O) and MRCA(A,B,C)')
    parser.add_argument('--loci',nargs=1,type=int,default=[10000],help='number of replications (independent loci).  Recommended to simulate a LOT and subsample.')
    parser.add_argument('--na',nargs='+',type=int,default=[1],help='N_e for pop A')
    parser.add_argument('--nb',nargs='+',type=int,default=[1],help='')
    parser.add_argument('--nc',nargs='+',type=int,default=[1],help='')
    parser.add_argument('--raxml',type=str,nargs=1,help='raxml binary file (depends oppn os)',required=True)
    parser.add_argument('--sim','-s',nargs='+',type=str,help='substitution model used for simulation (seqgen)',
                        choices=['HKY', 'F84', 'GTR', 'JTT', 'WAG', 'PAM', 'BLOSUM', 'MTREV', 'CPREV45', 'MTART', 'GENERAL'] )
    parser.add_argument('--model','-m',nargs='+',type=str,help='substitution model used for inference (raxml)',
                        choices = ['GTRCAT','GTRGAMMA',
                            'PROTCATDAYHOFFF', 'PROTCATDCMUTF', 'PROTCATJTTF', 'PROTCATMTREVF', 'PROTCATWAGF', 'PROTCATRTREVF', 'PROTCATCPREVF', 'PROTCATVTF', 'PROTCATBLOSUM62F', 'PROTCATMTMAMF', 'PROTCATLGF', 'PROTCATMTARTF', 'PROTCATMTZOAF', 'PROTCATPMBF', 'PROTCATHIVBF', 'PROTCATHIVWF', 'PROTCATJTTDCMUTF', 'PROTCATFLUF', 'PROTCATSTMTREVF', 'PROTCATDUMMYF', 'PROTCATDUMMY2F', 'PROTCATLG4MF', 'PROTCATLG4XF', 'PROTCATPROT_FILEF', 'PROTCATGTR_UNLINKEDF', 'PROTCATGTRF'])
    parser.add_argument('--overwrite','-w', action='store_true',help="overwrite existing table. Otherwise, append.  Warning: don't use with multithreaded")
    parser.add_argument('--out','-o',type=str,nargs=1,help='output database filename',required=True)

    args = parser.parse_args()
    print (args)
    main(args)
    
#### arguments: <T_a> <Tb-T_a> <T_c-T_b> <data_dir> <raxml_binary_name> 
### where T_a = T(a,b), T_b = T((a,b),c), T_c = T(((a,b),c),o)


