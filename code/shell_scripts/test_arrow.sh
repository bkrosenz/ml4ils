# source ~/.bash_profile;
# source activate bio3.6;

M=15e9
N=4

cd /N/dc2/projects/bkrosenz/deep_ils/code/pylib
pwd;

python simphy2arrow.py -p $N -m $M --dirlist /N/dc2/projects/bkrosenz/deep_ils/sims/simphy/simdirs.ae.txt --outdir /N/dc2/projects/bkrosenz/deep_ils/results/test_ae

