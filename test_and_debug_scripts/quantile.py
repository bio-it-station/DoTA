#!/usr/bin/env python
# coding: utf-8
import pickle
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from sklearn.preprocessing import MinMaxScaler

with open('/home/chenghung/DoTA/input/rf_data/human_weight_rf_data_filtered.pickle', mode='rb') as fh:
    x_data, y, tf_list = pickle.load(fh)
# Create psi table
psi_df = y.pivot(index='Gene', columns='Tissue')['PSI']
# Calculate stdev
psi_stdev = psi_df.std(axis=1, ddof=0)
# Drop genes with no stdev across different tissues
keep_idx = psi_stdev[psi_stdev != 0].index
psi_df = psi_df.loc[keep_idx]
psi_stdev = psi_stdev[keep_idx]
psi_mean = psi_df.mean(axis=1)
# Sort by PSI mean
psi_df = psi_df.loc[psi_mean.sort_values().index]
idx_df = y.sort_values(by=['Gene', 'Tissue']).reset_index().pivot(
    index='Gene', columns='Tissue')['index']
idx_df = idx_df.loc[psi_mean.sort_values().index]  # sort gene by PSI mean
orig_idx = idx_df.reset_index(drop=True).T.melt()['value']
# Z-score
zpsi_df = psi_df.apply(lambda x: (x - psi_mean[x.name]) / psi_stdev[x.name], axis=1)
# Do data scaling (linear scaling to [0,1])
data = psi_df.values
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data.T).T
# Top 20 % and Last 20%
data_scaled[(data_scaled > 0.2) & (data_scaled < 0.8)] = np.nan
psi_scaled = pd.DataFrame(data_scaled)
# Create table for ploting
psi_data = zpsi_df.reset_index(drop=True).T.melt(var_name='sample', value_name='zpsi')
psi_data['psi'] = psi_df.reset_index(drop=True).T.melt(var_name='sample')['value']
psi_data['psi_scaled'] = psi_scaled.reset_index(drop=True).T.melt(var_name='sample')['value']
psi_data['idx'] = orig_idx
psi_data.dropna(inplace=True)
# Set color code for different groups
psi_data['color'] = 'tab:blue'
psi_data.loc[(psi_data['psi_scaled'] > 0.5), 'color'] = 'tab:red'
psi_data.dropna(inplace=True)
positive_group = psi_data.groupby('color').count()['sample'][0]
negative_group = psi_data.groupby('color').count()['sample'][1]
total_remain = positive_group + negative_group
print('Genes:', len(psi_data['sample'].unique()))
print('Events:', total_remain)
print('Positive counts:', positive_group,
      "{0:.2f}".format(positive_group / len(y) * 100), '%')
print('Negative counts:', negative_group,
      "{0:.2f}".format(negative_group / len(y) * 100), '%')
print('Discarded counts:', len(y) - total_remain,
      "{0:.2f}".format((len(y) - total_remain) / len(y) * 100), '%')
# Shuffle the dataframe for ploting to avoid particular SD group dominant
psi_data = psi_data.sample(frac=1)
# Legend handler
legend_elements = [Line2D([], [], color='w', marker='.',
                          markerfacecolor='tab:blue', markersize=15, label='Last 20%'),
                   Line2D([], [], color='w', marker='.',
                          markerfacecolor='tab:red', markersize=15, label='Top 20%')]
# Scatter plot
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(8, 6))
plt.yticks([])
ax1.scatter(x=psi_data['psi'], y=psi_data['sample'],
            s=10, c=psi_data['color'], label=psi_data['color'],
            marker='.', edgecolors='none')
ax1.set_xlabel(r'PSI ($\Psi$)', fontsize=16)
ax1.set_ylabel('Genes', fontsize=16)
ax1.legend(handles=legend_elements)
ax2.scatter(x=psi_data['zpsi'], y=psi_data['sample'], s=1, c=psi_data['color'], marker='.')
ax2.set_xlabel(r'$Z_{\Psi}$', fontsize=16)
fig.tight_layout()
fig.savefig('psi_scatterplot.png', dpi=300)
plt.show()
psi_scaled.index = psi_df.index
event_counts_per_gene = psi_scaled.T.melt().dropna().groupby('Gene').count()
gene_counts_across_tissue = event_counts_per_gene.reset_index().groupby('value').count()
gene_counts_cdf = 100 * gene_counts_across_tissue.cumsum() / event_counts_per_gene.count().values
# Estimate tissue counts per gene
with sns.axes_style("darkgrid"):
    fig, ax = plt.subplots()
    plt.xlabel('Tissue counts per Gene')
    plt.xticks(range(2, len(gene_counts_across_tissue) + 2))
    ax.bar(gene_counts_across_tissue.index, gene_counts_across_tissue['Gene'], color='#ff7575')
    ax.set_ylabel('Counts (bar)')
    ax2 = plt.twinx()
    ax2 = sns.lineplot(data=gene_counts_cdf, markers='o', legend=False)
    ax2.set_ylabel('Cumulative percentage (line)')
    ax2.grid(None)
    fig.tight_layout()
    fig.savefig('psi_data_points_stat.png', dpi=300)
    plt.show()
x_data_filtered = x_data[psi_data['idx'].astype(int)]
y_filtered = y.iloc[psi_data['idx'].astype(int)].reset_index(drop=True)
y_filtered['ZPSI'] = psi_data.reset_index()['zpsi']
with open('human_weight_rf_data_filtered_quantile.pickle', mode='wb') as fh:
    pickle.dump((x_data_filtered, y_filtered, tf_list), fh)
