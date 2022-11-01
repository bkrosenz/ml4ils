from multiprocessing.sharedctypes import Value
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve, ConfusionMatrixDisplay, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sys import argv
MAX_EBL = 110
TRUE_CONCORDANCE_THRESHOLD = .9


def calc_rates(true, pred):
    tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
    if tn == fn == 0:
        fnr = 0
    else:
        fnr = fn/(tn+fn)
    fpr = 0 if tp == fp == 0 else fp/(tp+fp)
    return fnr, fpr


s = Path(argv[1]) if len(argv) > 1 else Path(
    '/N/project/phyloML/deep_ils/results/bo_final_small/test/g500_l500_f0.0_20/preds.csv.gz')
dd = pd.read_csv(s)
# dd['ebl'] /= 100
# dd['ibl'] /= 100

d = dd.query(f'ebl<{MAX_EBL}')
z = pd.DataFrame([(t, roc_auc_score(d.y_true > t, d.preds))
                 for t in np.linspace(.4, .999, 50)],
                 columns=['t', 'auc'])
print(f'Best AUC: {z.t[z.auc.idxmax()]}')


target = d.y_true > TRUE_CONCORDANCE_THRESHOLD
prediction = d.preds


fpr, tpr, thresholds = roc_curve(target, prediction, pos_label=1)

# Calculate the G-mean
gmean = np.sqrt(tpr * (1 - fpr))

# Find the optimal threshold
index = np.argmax(gmean)
thresholdOpt = round(thresholds[index], ndigits=4)
gmeanOpt = round(gmean[index], ndigits=4)
fprOpt = round(fpr[index], ndigits=4)
tprOpt = round(tpr[index], ndigits=4)
print('Best Threshold: {} with G-Mean: {}'.format(thresholdOpt, gmeanOpt))
print('FPR: {}, TPR: {}'.format(fprOpt, tprOpt))

rates = []
for ebl in np.arange(20, MAX_EBL+10, 10):
    d_slice = dd.query(f'{ebl}<=ebl<{ebl+10}')
    target = d_slice.y_true > TRUE_CONCORDANCE_THRESHOLD
    bin_pred = d_slice.preds > thresholdOpt
    try:
        rates.append((ebl/100, *calc_rates(target, bin_pred)))
    except ValueError:
        continue
rates = (pd.DataFrame(
    rates,
    columns=['External Branch Length', 'False Negative Rate', 'False Positive Rate'])
    .set_index('External Branch Length'))
rates.rolling(2).mean().plot()
plt.ylim(-.01, 1)
plt.savefig(s.parent/'fnr-tpr.png')
plt.close()
# plt.show()


target = d.y_true > TRUE_CONCORDANCE_THRESHOLD
prediction = d.preds
bin_pred = prediction > gmeanOpt


ConfusionMatrixDisplay.from_predictions(target,
                                        bin_pred,
                                        cmap=plt.cm.Blues,
                                        normalize='true')
plt.savefig(s.parent/'confusion.png')
# plt.show()
plt.close()


roc_auc = roc_auc_score(target, prediction)
roc_label = 'ROC (AUC={:.3f})'.format(roc_auc)

plt.plot(fpr, tpr, label=roc_label)
# plt.text(fpr[index], .9*tpr[index], f'Optimal threshold: {gmeanOpt}')

plt.annotate(f'Optimal threshold: {gmeanOpt}',
             xy=(fpr[index], tpr[index]),
             xytext=(fpr[index], .9*tpr[index]),
             arrowprops=dict(arrowstyle="->",
                             connectionstyle="arc3"),
             )

plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
x = [0.0, 1.0]
plt.plot(x, x, linestyle='dashed', color='red',
         linewidth=2, label='Random Classifier')
plt.legend(fontsize=10, loc='best')
plt.tight_layout()
plt.savefig(s.parent/'roc_auc.png')


exit()
# Create data viz
plotnine.options.figure_size = (8, 4.8)
(
    ggplot(data=df_fpr_tpr) +
    geom_point(aes(x='FPR',
                   y='TPR'),
               size=0.4) +
    # Best threshold
    geom_point(aes(x=fprOpt,
                   y=tprOpt),
               color='#981220',
               size=4) +
    geom_line(aes(x='FPR',
                  y='TPR')) +
    geom_text(aes(x=fprOpt,
                  y=tprOpt),
              label='Optimal threshold \n for class: {}'.format(thresholdOpt),
              nudge_x=0.14,
              nudge_y=-0.10,
              size=10,
              fontstyle='italic') +
    labs(title='ROC Curve') +
    xlab('False Positive Rate (FPR)') +
    ylab('True Positive Rate (TPR)') +
    theme_minimal()
)
