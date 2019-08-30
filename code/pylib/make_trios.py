from os import path
from itertools import product,chain

# dirname = '/N/dc2/projects/bkrosenz/deep_ils/data/metazoa/whelan2017/species'
# filenames = {
#     'metazoa':'''cnidaria.whelan  bilateria.whelan'''.split(),
#     'ctenophora':'ctenophora.whelan'.split(),
#     'porifera':'porifera.whelan'.split(),
#     'outgroups':'''choanoflagellata.whelan'''.split()
# #    'outgroups':'''ichthyosporea.whelan  choanoflagellata.whelan  fungi.whelan'''.split()
# }

dirname = '/N/dc2/projects/bkrosenz/deep_ils/data/metazoa/Simion2017/species'
filenames = {
    'bilateria':'''bilateria.simion'''.split(),
    'ctenophora':'ctenophora.simion'.split(),
    'porifera':'porifera.simion'.split(),
    'outgroups':'''outgroup.simion'''.split()
#    'outgroups':'''ichthyosporea.simion  choanoflagellata.simion  fungi.simion'''.split()
}

def fsplit(fn):
    with open(path.join(dirname,fn)) as f:
        return filter(None,f.read().split('\n'))

species = {clade:chain.from_iterable(fsplit(fn) for fn in files)
           for clade,files in filenames.items() }

with open(path.join(dirname,'all_trios.bcpo.csv'), 'w') as f:
    f.write(','.join(s.capitalize() for s in filenames)+'\n')
    for tup in product(*species.values()):
        f.write(','.join(tup)+'\n' )

print([len(list(s)) for s in species.values()])

