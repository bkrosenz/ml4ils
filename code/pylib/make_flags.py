"""discoal output format:
command

//
segsites
positions
loc1 ancient1 (most recent)
loc1 ancient2 (most ancient)
loc1 contemp 
loc2 ancient1 (most recent)
loc2 ancient2 (most ancient)
loc2 contemp
...
"""
from itertools import product
import argparse
import utils as u

parser = argparse.ArgumentParser()
parser.add_argument("--ghost", type=int,help="number of ghost locs separating each sampled pop (not implemented)",
                    default=0)
parser.add_argument("--lattice", type=int,help="2-d lattice type: square, triangular, hex (only square implemented)",
                    default=0)
parser.add_argument("--dim", type=int,
                    help="dimension of lattice. must be 1 or 2.",
                    default=2)
parser.add_argument("--replicates", type=int,
                    default=1)
parser.add_argument("--sites", type=int,
                    help="number of sites between which recomb can occur.  For some reason must be > 1, even when recomb rate is zero.",
                    default=2)
parser.add_argument("--ancient", type=int,
                    default=1)
parser.add_argument("--contemp", type=int,
                    default=1)
parser.add_argument("--epochs", type=int,help="number of epochs to sample (including contemporary)",
                    default=4)
parser.add_argument("--locs", type=int,
                    help="number of locations sampled (size of grid)",
                    default=9)
parser.add_argument("--pop", nargs='+',
                    help="'l1,...,ln m'.  Add a population consisting of locs l1,...,ln with mobility (within-pop migration) m",
                    action='append')
parser.add_argument("--move", type=list,
                    help="move pop x1 to location xn passing through x2,...,x_n-1")
parser.add_argument("--step", type=float,
                    default=0.1)
parser.add_argument("--migration", type=float,
                    default=0.1)
parser.add_argument("--outfile", type=str,
                    default='flags.txt')

args = parser.parse_args()

nReplicates = args.replicates
nSites = args.sites
nAncient = args.ancient # samples per ancient epoch
nContemp = args.contemp
nEpochs = args.epochs-1 #number of ancient epochs to sample
step = args.step # time between ancient samps
nLocs = args.locs # perfect square
sampsPerPop =nAncient*nEpochs+nContemp
mig = args.migration

if args.dim==2 and u.module_exists('gmpy2'):
    from gmpy2 import is_square
    if not is_square(nLocs):
        raise ValueError("nLocs must be a perfect square for square lattice")

def mergeEvent(s):
    try:
        s = s.split()
        t = float(s[0])
        x,y = map(int,s[1:])
        return t,x,y
    except:
        TypeError("merge must be 'time sourcePop destPop'")
        
neighbors = u.get_neighbor_fn(args)

flags = {}
flags[' -m '] = [(i,j,mig) for i in range(nLocs) for j in neighbors(i)]
flags[' -A '] = [(nAncient,pop,step*t) for t in range(1,nEpochs+1) for pop in range(nLocs)]
#  '-t':2.0,

flags_full = [' -p ' + ' '.join(map(u.to_str,[nLocs]+[sampsPerPop]*nLocs))]
for k,v in flags.items():
    flags_full.extend(
        ''.join(tup) for tup in product([k], map(u.to_str,v))
    )

totalSamps = (sampsPerPop)*nLocs 
flagstr = '%d %d %d %s' % (totalSamps, nReplicates, nSites, ' '.join(flags_full))
print flagstr

with open(args.outfile,'w') as f:
    f.write(flagstr)
