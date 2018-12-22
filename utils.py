import errno
import os
import pickle
import sys
from abc import ABC, abstractmethod
from itertools import combinations
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.sparse import issparse, vstack
from scipy.special import comb
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


def _getThreads():
    """ Returns the number of available threads on a posix/win based system """
    if sys.platform == 'win32':
        return (int)(os.environ['NUMBER_OF_PROCESSORS'])
    return (int)(os.popen('grep -c cores /proc/cpuinfo').read())


def delete_emtpy_tf(x, tf_list):
    print('Removing tf columns with no binding in whole dataset...', end='')

    if issparse(x):
        tf_sum = np.asarray(x.sum(axis=1).reshape(x.shape[0] // len(tf_list), len(tf_list)).sum(axis=0))
        del_tf = np.where(tf_sum.reshape(-1) == 0)[0]
        arr = np.arange(0, x.shape[0], len(tf_list))
        new_arr = np.empty((del_tf.shape[0], arr.shape[0]), dtype=np.int)
        for n, v in enumerate(del_tf):
            new_arr[n] = arr + v
        mask = np.ones(x.shape[0], dtype='bool')
        for idx in new_arr:
            mask[idx] = False
        x = x[mask]
    else:
        del_tf = np.where(x.sum(axis=0) == 0)[0]
        x = np.delete(x, del_tf, 1)
    tf_list = np.delete(np.asarray(tf_list), del_tf).tolist()
    print('DONE!')

    return x, tf_list


def filter_by_gene(x, y, gene_list, tf_list):
    n_tf = len(tf_list)
    filtered_idx = y[y['Gene'].isin(gene_list)].index
    if issparse(x):
        mask = np.zeros(x.shape[0], dtype='bool')
        for idx in filtered_idx:
            mask[idx * n_tf: (idx + 1) * n_tf] = True
        x = x[mask]
    else:
        x = x[filtered_idx]
    y = y.loc[filtered_idx].reset_index(drop=True)
    return x, y


def filter_less_than_3(x, y, tf_list):
    """ remove genes with less than 3 tissue """
    print('Removing genes with less than 3 tissue-events...', end='')

    gene_count = y.groupby('Gene')['PSI'].count()
    gene_list = gene_count[gene_count > 2].index
    x, y = filter_by_gene(x, y, gene_list, tf_list)

    # Create psi table
    psi_df = y.pivot(index='Gene', columns='Tissue')['PSI']

    # Calculate stdev
    psi_stdev = psi_df.std(axis=1)

    # Drop genes with no stdev across different tissues
    keep_gene = psi_stdev[psi_stdev != 0].index
    x, y = filter_by_gene(x, y, keep_gene, tf_list)
    print('DONE!')

    return x, y


def filter_low_psi_range(x, y, tf_list):
    """ remove genes with low psi range """
    print('Removing genes with low psi range...', end='')

    # Create psi table
    psi_df = y.pivot(index='Gene', columns='Tissue')['PSI']
    # Cut-off genes with psi range < 0.2
    psi_range = psi_df.max(axis=1) - psi_df.min(axis=1)
    gene_remained_list = psi_range[psi_range >= 0.2]
    x, y = filter_by_gene(x, y, gene_remained_list.index, tf_list)
    print('DONE!')

    return x, y


def psi_z_score(y: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Convert raw PSI to z-score
    :param x: TF binding profile
    :param Y: Raw PSI value
    :return result: tuple of converted (X, Y)
    """
    print('Converting PSI to z-score...', end='')

    # calculate mean
    psi_gene_group = y.groupby('Gene')
    psi_mean = psi_gene_group['PSI'].mean().to_dict()

    # calculate stdev and convert 0 to nan for prevention of ZeroDivisionError
    psi_sd = psi_gene_group['PSI'].std()
    psi_sd[psi_sd == 0.0] = float('nan')
    psi_sd = psi_sd.to_dict()

    # calculate z-score
    z_score = [(psi - psi_mean[gene]) / psi_sd[gene] for gene, psi in zip(y['Gene'], y['PSI'])]
    y['ZPSI'] = z_score
    print('DONE!')

    return y


def quantile_convert(x, y, tf_list):
    print('Applying quantile and Z-score convertion...', end='')

    # Create psi table
    psi_df = y.pivot(index='Gene', columns='Tissue')['PSI']
    # Do data scaling (linear scaling to [0,1])
    data = psi_df.values.copy()
    row_mean = np.nanmean(data, axis=1)
    idx = np.where(np.isnan(data))
    data[idx] = np.take(row_mean, idx[0])
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data.T).T
    data[idx] = np.nan
    data_scaled[idx] = np.nan
    # set 20% ~ 80% to NaN
    data_scaled[(data_scaled > 0.2) & (data_scaled < 0.8)] = np.nan
    psi_scaled = pd.DataFrame(data_scaled, columns=psi_df.columns, index=psi_df.index)
    # Z-score
    psi_mean = psi_df.mean(axis=1)
    psi_stdev = psi_df.std(axis=1)
    zpsi_df = psi_df.apply(lambda x: (x - psi_mean[x.name]) / psi_stdev[x.name], axis=1)
    # save original idx
    idx_df = y.reset_index().pivot(index='Gene', columns='Tissue')['index']
    orig_idx = idx_df.reset_index(drop=True).T.melt()['value']
    # collect all information
    psi_data = zpsi_df.reset_index(drop=True).T.melt(var_name='sample', value_name='zpsi')
    psi_data['psi'] = psi_df.reset_index(drop=True).T.melt(var_name='sample')['value']
    psi_data['psi_scaled'] = psi_scaled.reset_index(drop=True).T.melt(var_name='sample')['value']
    psi_data['idx'] = orig_idx
    psi_data.dropna(inplace=True)
    psi_data['idx'] = psi_data['idx'].astype(int)
    psi_data = psi_data.sort_values('idx')
    # divied into two group
    psi_data['psi_group'] = False
    psi_data.loc[psi_data['psi_scaled'] > 0.5, 'psi_group'] = True
    # filter original data
    if issparse(x):
        mask = np.zeros(x.shape[0], dtype='bool')
        for i in psi_data['idx'].values:
            mask[i * len(tf_list): (i + 1) * len(tf_list)] = True
        x = x[mask]
    else:
        x = x[psi_data['idx']]
    y = y.iloc[psi_data['idx'].astype(int)].reset_index(drop=True)
    y['ZPSI'] = psi_data.reset_index()['zpsi']
    y['psi_group'] = psi_data.reset_index()['psi_group']
    print('DONE!')

    return x, y


class _DeltaConverter(ABC):
    """Basal Class of Delta data converter"""

    def __init__(self, x: np.ndarray, y: pd.DataFrame, tf_list: list) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        :param x: feature data for CART model
        :param y: target data for ML
        :param tf_list: transcription factor list
        :return x: new feature data in delta feature format
        :return y: new target data in delta target format
        """
        self.x = x
        self.y = y
        self.tf_list = tf_list
        gene_count = self.y.groupby('Gene').count()
        self.genes = gene_count[gene_count > 1].dropna()
        self.new_x = None
        self.new_y = None
        print('Performing delta data convention...\n')

        # Create an nan array with estimated event numbers as new features array
        events_count = 0
        for num in self.genes['Tissue']:
            events_count += comb(num, 2)
        self.esti_events_num = int(events_count)
        print('estimated event numbers:', self.esti_events_num)

    @abstractmethod
    def output(self):
        pass

    @abstractmethod
    def converter(self):
        pass


class QuantDeltaConverter(_DeltaConverter):
    """Apply Quantile categorize 20%"""

    def converter(self):
        i = 0
        new_y_dpsi = []
        new_y_gene = []
        new_psi_group = []
        self.new_x = np.zeros((self.esti_events_num, len(self.tf_list)))
        self.new_x[:] = np.nan
        for gene in self.genes.index:
            event_ind_list = self.y[self.y['Gene'] == gene].index
            for event_1, event_2 in combinations(event_ind_list, 2):
                delta_psi = abs(self.y.at[event_1, 'ZPSI'] - self.y.at[event_2, 'ZPSI'])
                if delta_psi:
                    delta_feature = self.x[event_1] ^ self.x[event_2]
                    delta_psi_group = self.y.at[event_1, 'psi_group'] ^ self.y.at[event_2, 'psi_group']
                else:
                    continue
                self.new_x[i] = delta_feature
                new_y_dpsi.append(delta_psi)
                new_psi_group.append(delta_psi_group)
                new_y_gene.append(gene)
                i += 1
        self.new_y = pd.DataFrame(data={'Gene': new_y_gene, 'PSI': new_y_dpsi, 'psi_group': new_psi_group})
        self.new_y['psi_group'] = self.new_y['psi_group'].astype('category')

    def output(self):
        print('Applying delta data converter...', end='')
        self.converter()
        self.new_x = self.new_x[~np.isnan(self.new_x).any(axis=1)].astype('bool')
        self.new_y['Gene'] = self.new_y['Gene'].astype('category')
        print('DONE!')
        return self.new_x, self.new_y


class QuantDeltaConverterSparse(_DeltaConverter):
    """Apply Quantile categorize 20% with sparse matrix"""

    def converter(self):
        if not issparse(self.x):
            raise ValueError('X_data is not sparse matrix')
        i = 0
        n_tf = len(self.tf_list)
        new_y_dpsi = []
        new_y_gene = []
        new_psi_group = []
        self.new_x = []
        for gene in tqdm(self.genes.index):
            event_ind_list = self.y[self.y['Gene'] == gene].index
            for event_1, event_2 in combinations(event_ind_list, 2):
                delta_psi = abs(self.y.at[event_1, 'ZPSI'] - self.y.at[event_2, 'ZPSI'])
                if delta_psi:
                    x_1 = self.x[event_1 * n_tf:(event_1 + 1) * n_tf]
                    x_2 = self.x[event_2 * n_tf:(event_2 + 1) * n_tf]
                    delta_feature = x_1 - x_2
                    delta_psi_group = self.y.at[event_1, 'psi_group'] ^ self.y.at[event_2, 'psi_group']
                else:
                    continue
                self.new_x.append(delta_feature)
                new_y_dpsi.append(delta_psi)
                new_psi_group.append(delta_psi_group)
                new_y_gene.append(gene)
                i += 1
        self.new_y = pd.DataFrame(data={'Gene': new_y_gene, 'PSI': new_y_dpsi, 'psi_group': new_psi_group})
        self.new_y['psi_group'] = self.new_y['psi_group'].astype('category')

    def output(self):
        print('Applying delta data converter...', end='')
        self.converter()
        self.new_x = vstack(self.new_x, format='csr', dtype=np.bool)
        self.new_y['Gene'] = self.new_y['Gene'].astype('category')
        print('DONE!')
        return self.new_x, self.new_y


class DeltaConverter(_DeltaConverter):
    """Converter without categorized label"""

    def converter(self):
        i = 0
        new_y_dpsi = []
        new_y_gene = []
        self.new_x = np.zeros((self.esti_events_num, len(self.tf_list)))
        self.new_x[:] = np.nan
        for gene in self.genes.index:
            event_ind_list = self.y[self.y['Gene'] == gene].index
            for event_1, event_2 in combinations(event_ind_list, 2):
                delta_psi = abs(self.y.at[event_1, 'ZPSI'] - self.y.at[event_2, 'ZPSI'])
                if delta_psi:
                    delta_feature = self.x[event_1] ^ self.x[event_2]
                else:
                    continue
                self.new_x[i] = delta_feature
                new_y_dpsi.append(delta_psi)
                new_y_gene.append(gene)
                i += 1
        self.new_y = pd.DataFrame(data={'Gene': new_y_gene, 'PSI': new_y_dpsi})

    def output(self):
        print('Applying delta data converter...', end='')
        self.converter()
        self.new_x = self.new_x[~np.isnan(self.new_x).any(axis=1)].astype('bool')
        self.new_y['Gene'] = self.new_y['Gene'].astype('category')
        print('DONE!')
        return self.new_x, self.new_y


class nn_X_generator:
    def __init__(self, x, tf_list):
        self.x = x
        self.len = len(tf_list)

    def __getitem__(self, n):
        data = self.x[n * self.len: (n + 1) * self.len]
        return data


def output(var: tuple, filename: str) -> None:
    """
    Save required information into a file
    :param var: information which is required to output
    :param filename: output file name
    """
    try:
        os.makedirs(os.path.dirname(filename))
    except OSError as err:
        if err.errno != errno.EEXIST:
            raise

    with open(filename, mode='wb') as fh:
        pickle.dump((var), fh)
