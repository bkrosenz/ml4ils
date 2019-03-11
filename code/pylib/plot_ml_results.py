import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
import pandas as pd
from sys import argv
from os import path
plt.ion()


# filename to newick:
#rx=re.compile(r"t_(\d+)_([\d\.]+)_([\d\.]+)")

# import matplotlib
# matplotlib.interactive(True)

filepath=len(argv)>1 and argv[1] or '/Users/ben/deep_ils/results/results.classify.csv'

try:
    df = pd.read_csv(filepath, index_col = (0,1)) # TODO: fix 'UnicodeDecodeError: 'utf-8' codec can't decode byte 0x89 in position 0: invalid start byte' error
except:
    print('syntax: python plot_ml_results.py <results_file.csv>')
    raise
finally:
    print('called with',argv)
    
metrics = df.index.get_level_values('metrics').unique()

for metric in metrics:
    data = df.xs(metric, level=1, drop_level=True)
    baseline = sorted(data.index.intersection(['Random','Trivial','Mean','Median']).tolist())
    learners = sorted(data.index.difference(baseline).tolist())+baseline

    print(learners)
    data = data.transpose()
    ax1,box_dict = data.boxplot(column = learners,
                                            rot=45,
                                            fontsize=9,
                                            return_type='both',
                                            patch_artist=True)
    
    
    # Hide these grid behind plot objects
    #ax1.set_axisbelow(True)
    ax1.set_title('Performance')
    ax1.set_xlabel('Algorithm')
    ax1.set_ylabel(metric)

    if len(argv)>2 and argv[2]=='log':
        ax1.set_yscale('log')
        
    # Now fill the boxes with desired colors
    boxColors = ['darkkhaki', 'royalblue']
    
    for patch,name in zip(box_dict['boxes'],learners):
        if name in baseline:
            patch.set(facecolor='pink',color='red')
        else:
            patch.set(facecolor='lightblue')
        

    plt.tight_layout()

    plt.savefig(path.splitext(filepath)[0]+'.%s.png'%metric)
    plt.clf()
