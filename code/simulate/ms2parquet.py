from __future__ import print_function
from contextlib import contextmanager
from functools import partial
from itertools import imap,chain
from cStringIO import StringIO
from glob import glob
import subprocess, sqlite3, argparse, tempfile
import shutil


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


def main(args):
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

    mscmd = """{prog}/ms {total_samps} {nreps} -T \
    -I {total_samps} {nsamps} {nsamps} {nsamps} 1 \
    -ej {ta} 2 1 -ej {tb} 3 1 -ej {tc} 4 1 \
    ;""".format(
        prog=PROG,
        nreps=args.loci,
        nsamps=nsamps,
        total_samps=total_samps,
        ta=ta, tb=tb, tc=tc)

    ps = subprocess.Popen(mscmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,bufsize=-1) 
    mstrees, stderr = ps.communicate()
#    ps.terminate()

    print (mstrees.shape)


# wrapper
if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Process some data.')
    parser.add_argument('--ebl',type=float,help='length of branch subtending MRCA(A,B)')
    parser.add_argument('--ibl',type=float,help='length of branch between MRCA(A,B,C) and MRCA(A,B)')
    parser.add_argument('--tout',type=float,help='length of branch between MRCA(A,B,C,O) and MRCA(A,B,C)')
    parser.add_argument('--loci',type=int,default=10000,help='number of replications (independent loci).  Recommended to simulate a LOT and subsample.')
    parser.add_argument('--na',type=int,default=1,help='N_e for pop A')
    parser.add_argument('--nb',type=int,default=1,help='')
    parser.add_argument('--nc',type=int,default=1,help='')
    parser.add_argument('--overwrite','-w', action='store_true',help='overwrite existing table. Otherwise, append')
    parser.add_argument('--out','-o',type=str,help='output database filename',required=True)
    
    args = parser.parse_args()
    main(args)
    
#### arguments: <T_a> <Tb-T_a> <T_c-T_b> <data_dir> <raxml_binary_name> 
### where T_a = T(a,b), T_b = T((a,b),c), T_c = T(((a,b),c),o)


