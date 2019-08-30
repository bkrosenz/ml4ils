python trees2sql2.py \
       --quartets /N/dc2/projects/bkrosenz/deep_ils/data/metazoa/whelan2017/species/all_trios.csv \
       --seqdir /N/dc2/projects/bkrosenz/deep_ils/data/metazoa/whelan2017/metazoa_genes/phylip \
       --treedir /N/dc2/projects/bkrosenz/deep_ils/data/metazoa/whelan2017/metazoa_genes/trees \
       --buffsize 50 \
       --csize 20 \
       -o metazoa \
       --procs $1 \
       --seqtype protein \
       --overwrite
