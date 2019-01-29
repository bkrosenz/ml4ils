from __future__ import print_function
from multiprocess import Pool
from contextlib import contextmanager
from functools import partial
from itertools import chain
from cStringIO import StringIO
from glob import glob
from sys import argv
import subprocess, sqlite3, argparse, tempfile
import shutil

try:
    from itertools import imap
except ImportError:
    imap=map
        
def wrap_with_list(obj):
    if hasattr(obj,'__iter__'):
        return obj
    return [obj]

@contextmanager
def dbconnect(dbfile):
    c = sqlite3.connect(dbfile)
    yield c
    c.close()

class TreeSet:
    def __init__(self,args,files,inferred=True):
        self.args=vars(args)
        self.files=files
        self.keytypes = {'filename':'VARCHAR(200)',
                         'basename':'VARCHAR(50)',
                         'tree_num':'INT',
                         'tree':'TEXT',
                         'topology':'INT'}
        self.d = dict.fromkeys(self.keytypes)
        if inferred:
            for k in ('model','sim'):
                self.d[k]=args[k]
                self.keytypes[k] = 'VARCHAR(200)'
            self.keytypes['seqs']='TEXT'
        else:
            for k in ('ibl','ebl','tout','na','nb','nc'):
                self.d[k] = args[k]
                self.keytypes[k] = 'INT'
        self.keys = sorted(self.d)
        

    def __iter__(self):
        if type(self.files)==list:
            return chain(imap(self.value_generator,self.files))
        else: # a single file handle or fpath str
            return self.value_generator(self.files)
        
    def value_generator(fn):
        tree_num = 0
        self.d['filename'] = fn
        self.d['basename'] = clean_filename(fn)
    
        trees = Phylo.parse(fn, 'newick')
        for t in trees:
            self.d['tree_num'] = tree_num
            self.d['tree'] = t.format('newick')
            self.d['topology'] = get_topo(t)
            self.d.update( get_dist_mat(t) )
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
    
    # mscmd = "{prog}/ms {total_samps} {nreps} -T \
        # -I {total_samps} {nsamps} {nsamps} {nsamps} 1 \
        # -ej {ta} 2 1 -ej {tb} 3 1 -ej {tc} 4 1 \
        # | tee \
        #  >({prog}/seq-gen -m{model} \ 
    #  -g5 \
        #  -s{scale} -l{seqlen} \
        #  | split -l{split_lines} - {d}/seqs.)\
        # ;".format(

    mscmd = """{prog}/ms {total_samps} {nreps} -T 
    -I {total_samps} {nsamps} {nsamps} {nsamps} 1 
    -ej {ta} 2 1 -ej {tb} 3 1 -ej {tc} 4 1 
    ;""".format(
        prog=PROG,
        nreps=args.loci,
        nsamps=nsamps,
        total_samps=total_samps,
        ta=ta, tb=tb, tc=tc)

    with subprocess.Popen(mscmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,bufsize=-1) as ps:
        mstrees, stderr = ps.communicate()

    print (mstrees)
    seqgencmd="""{prog}/seq-gen -m{model} 
     -g{cats} 
     -s{scale} -l{seqlen} 
     | tee >(split -l{split_lines} - {d}/seqs.) 
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
    raxmlcmd = """ls {d}/seqs.* 
        | xargs -I@ 
                {prog}/{raxml_binary} -s {d}/@ -w {d} -n @.raxml 
                -m {model} 
                -p 12345 
    ;""".format(
        prog=PROG,
        raxml_binary = args.raxml,
        d = dirpath,
        model = args.model
    )

    with subprocess.Popen(raxmlcmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,bufsize=-1) as ps:
        raxml_output = ps.communicate()[0]


    resmap = { 'inferred': TreeSet(args=args, inferred=True, fh = glob(path.join(dirpath,'RAxML_bestTree.*.raxml', seqs = seqs))), # separate file for each tree
               'true': TreeSet(args=args, inferred=False, fh=StringIO(mstrees)) # one filehandle
    }

    # Successful, con.commit() is called automatically afterwards

    with dbconnect(args.out) as c:
        for table_name,tree_generator in resmap.iteritems():
            c.execute('''CREATE TABLE IF NOT EXISTS 
            {tn} ({fields});'''.format( tn=table_name,
                                        fields=tree_generator.get_field_str())
            )
            with c:
                c.executemany(
                    'INSERT INTO {tn}({fields}) VALUES ({values})'.format( tn=table_name,
                                                                           fields=','.join(tree_generator.keys),
                                                                           values=','.join(['?']*len(tree_generator.keys)) ),
                    tree_generator
                )

    shutil.rmtree(dirpath)
    print('raxml output:',raxml_output)
    print("finished",args)

def main(args):
    p = Pool(args.procs)
    v = vars(args)
    params = v.keys()
    z=(wrap_with_list(vv) for vv in v.values())
    for config in product(*z):
        p.apply_async(worker, argparse.Namespace(dict(zip(params,config))))
    
    
# wrapper
if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Process some data.')

    parser.add_argument('--loci',nargs=1,type=int,default=10000,help='number of replications (independent loci).  Recommended to simulate a LOT and subsample.')
    parser.add_argument('--na',type=int,nargs=1,default=1,help='N_e for pop A')
    parser.add_argument('--nb',type=int,nargs=1,default=1,help='')
    parser.add_argument('--nc',type=int,nargs=1,default=1,help='')
    parser.add_argument('--raxml',type=str,nargs=1,help='raxml binary file (depends oppn os)',required=True)
    parser.add_argument('--out','-o',nargs=1,type=str,help='output database filename',required=True)
    parser.add_argument('--procs',type=int,nargs=1,default=4,help='number of processes')

    parser.add_argument('--ebl',nargs='+',type=float,help='length of branch subtending MRCA(A,B)',required=True)
    parser.add_argument('--ibl',nargs='+',type=float,help='length of branch between MRCA(A,B,C) and MRCA(A,B)',required=True)
    parser.add_argument('--tout',nargs='+',type=float,help='length of branch between MRCA(A,B,C,O) and MRCA(A,B,C)',required=True)
    parser.add_argument('--sim','-s',nargs='+',help='substitution model used for simulation (seqgen)',
                        choices=['HKY', 'F84', 'GTR', 'JTT', 'WAG', 'PAM', 'BLOSUM', 'MTREV', 'CPREV45', 'MTART', 'GENERAL'] ,required=True)
    parser.add_argument('--model','-m',nargs='+',help='substitution model used for inference (raxml)',
                        choices = ['GTRCAT','GTRGAMMA',
                                   'PROTCATDAYHOFFF', 'PROTCATDCMUTF', 'PROTCATJTTF', 'PROTCATMTREVF', 'PROTCATWAGF', 'PROTCATRTREVF', 'PROTCATCPREVF', 'PROTCATVTF', 'PROTCATBLOSUM62F', 'PROTCATMTMAMF', 'PROTCATLGF', 'PROTCATMTARTF', 'PROTCATMTZOAF', 'PROTCATPMBF', 'PROTCATHIVBF', 'PROTCATHIVWF', 'PROTCATJTTDCMUTF', 'PROTCATFLUF', 'PROTCATSTMTREVF', 'PROTCATDUMMYF', 'PROTCATDUMMY2F', 'PROTCATAUTOF', 'PROTCATLG4MF', 'PROTCATLG4XF', 'PROTCATPROT_FILEF', 'PROTCATGTR_UNLINKEDF', 'PROTCATGTRF'],required=True)

    parser.add_argument('--overwrite','-w',action='store_true',help='overwrite existing table. Otherwise, append')
    

    args = parser.parse_args()#argv[1].split())
    main(args)
    
#### arguments: <T_a> <Tb-T_a> <T_c-T_b> <data_dir> <raxml_binary_name> 
### where T_a = T(a,b), T_b = T((a,b),c), T_c = T(((a,b),c),o)


