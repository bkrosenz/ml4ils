# sql:
# \copy (select st."1:2"-st."1:3" as ibl, st."1:1"-st."1:2" as ebl, count(1) as c,nblocks as b,ngenes as g from (select * from recomb as rb join gene_trees as gt on gt.id=rb.inf where topology!='(4,(3,(1,2)));' ) as foo join species_trees as st on foo.sid=st.id group by b,g,ebl,ibl order by c) to 'discordant.recomb.csv' csv header
#  \copy (select st."1:2"-st."1:3" as ibl, st."1:1"-st."1:2" as ebl, count(1) as c,nblocks as b,ngenes as g from (select * from recomb as rb join gene_trees as gt on gt.id=rb.inf where topology='(4,(3,(1,2)));' ) as foo join species_trees as st on foo.sid=st.id group by b,g,ebl,ibl order by c) to 'concordant.recomb.csv' csv header

from itertools import combinations
from pathlib import Path
from statistics import mode

import numpy as np
import pandas as pd
import statsmodels
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from statsmodels.formula.api import ols
from sklearn.inspection import PartialDependenceDisplay
import statsmodels.api as sm
from statsmodels.gam.api import BSplines, GLMGam
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor


def calc_anova():
    c = pd.read_csv('concordant.recomb.csv')
    d = pd.read_csv('discordant.recomb.csv')
    m = c.merge(d, on=['ibl', 'ebl', 'b', 'g'], how='outer')
    m['n'] = m.c_x/(m.c_x+m.c_y)
    mod = ols(formula='n ~ g*b*ebl*ibl',
              data=m).fit()
    aov_table = sm.stats.anova_lm(mod, typ=2)
    print(aov_table[aov_table['PR(>F)'] < .2])


def calc_GradHist_PDP(df,
                      s,
                      predictors=['Length', 'Genes', 'Fraction', 'ebl', 'ibl'],
                      features=('Length', 'Genes',),
                      target='abs_err',
                      common_params={
                          "subsample": 50,
                          "n_jobs": 6,
                          "grid_resolution": 20,
                          "centered": True,
                             "random_state": 0,
                      }
                      ):
    X_train, y_train = df[predictors], df[target]

    est = HistGradientBoostingRegressor(random_state=0).fit(X_train, y_train)

    _, ax = plt.subplots(ncols=3, figsize=(9, 4))

    display = PartialDependenceDisplay.from_estimator(
        est,
        X_train,
        features=[*features, tuple(features)],
        kind=['both', 'both', 'average'], ax=ax, **common_params
    )
    display.figure_.subplots_adjust(wspace=0.4, hspace=0.3)
    # plt.tight_layout()
    plt.savefig(s)


def calc_glmgam(df: pd.DataFrame,
                s: Path = Path(
                    '/N/project/phyloML/deep_ils/results/bo_final_2/test')
                ):

    s /= 'glm_fits'
    s.mkdir(parents=True, exist_ok=True)
    # create spline basis for weight and hp
    bs = BSplines(df[['ebl', 'ibl', 'y_true', 'log_true']],
                  df=5*[8], degree=5*[3])

    # penalization weight

    glm = sm.GLM(
        endog=df.ERROR,
        exog=df[['ebl', 'ibl', 'Length', 'Genes', 'Fraction']])

    models = []
    model_names = ['target~(Length+Genes+Fraction)**2',
                   'target~Length*Genes*Fraction',
                   'target~Length*Genes*Fraction-Length',
                   'target~Length+Genes+Fraction', ]
    aic_values = np.empty(len(model_names))
    for target in ('preds', 'np.log(preds)', 'abs_err', 'ERROR'):
        for i, m in enumerate(model_names):
            m = m.replace('target', target)
            gam_bs = GLMGam.from_formula(
                m,
                data=df,
                smoother=bs)
            res_bs = gam_bs.fit()
            models.append(res_bs)
            summary = res_bs.summary()
            aic_values[i] = res_bs.aic
            with open(s/(m+'.tex'), 'w') as f:
                f.write(summary.as_latex())
        best = np.argmin(aic_values)
        print(model_names[best].replace('target', target),
              aic_values[best], '\n-----\n')
        PartialDependenceDisplay.from_estimator(
            models[best],
            df.sample(500),
            ['Length', 'Genes', 'Fraction'],
            kind='both')


if __name__ == '__main__':
    df = pd.read_pickle(
        '/N/project/phyloML/deep_ils/results/bo_final_2/test/rec_fraction.pd.pkl').reset_index()

    df['log_true'] = np.log(df.y_true)
    # df['log_preds'] = np.log(df.preds)

    # normalize
    predictors = ['Length', 'Genes', 'Fraction',
                  'ebl', 'ibl', 'y_true', 'log_true']
    df[predictors] = (df[predictors]-df[predictors].mean()) / \
        df[predictors].std()

for c in combinations(['Genes', 'Length', 'Blocks'], 2):
         print(c)
         rs.calc_GradHist_PDP(z.query('Blocks==2'),
         s/f'blocks2-{"_".join(c)}-abserr.png', features=c, predictors=['Length', 'Genes', 'Blocks', 'ebl', 'ibl'],)


    calc_glmgam(df)
