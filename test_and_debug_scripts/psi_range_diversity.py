#!/usr/bin/env python3
# coding: utf-8
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rpy2.robjects as robjects
import seaborn as sns
from rpy2.robjects import numpy2ri

from rpy2.robjects.packages import importr
vegan = importr('vegan')


def mean_beta_div(dat):
    beta_div = np.array(vegan.vegdist(dat, method='bray'))
    return beta_div.mean()


with open('human_weight_rf_data.pickle', mode='rb') as fh:
    x, y, tf_list = pickle.load(fh)
y_with_idx = y.reset_index()
result = {}
numpy2ri.activate()
for gene, indices in y_with_idx.groupby('Gene')['index']:
    if len(indices) > 2:
        result[gene] = mean_beta_div(x[indices])
numpy2ri.deactivate()
psi_df = pd.read_csv('psi_table.csv', index_col=0)
psi_range = psi_df.max(axis=1) - psi_df.min(axis=1)
feature_div = pd.Series(result)
summary_df = pd.DataFrame(
    {'Range': psi_range, 'Div': feature_div[psi_range.index]})
sns.set(rc={'figure.dpi': 300})
ax = sns.regplot(
    x='Range',
    y='Div',
    data=summary_df,
    marker='.',
    scatter_kws={
        'edgecolors': 'none'})
ax.set_xlabel('PSI Range')
ax.set_ylabel(r'Mean $\beta$ Diversity')
plt.savefig('output.png')
plt.show()
