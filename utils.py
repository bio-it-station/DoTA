import errno
import os
import pickle
import sys
from abc import ABC, abstractmethod
from itertools import combinations
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.special import comb
from scipy.sparse import issparse


def _getThreads():
    """ Returns the number of available threads on a posix/win based system """
    if sys.platform == 'win32':
        return (int)(os.environ['NUMBER_OF_PROCESSORS'])
    return (int)(os.popen('grep -c cores /proc/cpuinfo').read())


def delete_emtpy_tf(x, tf_list):
    if issparse(x):
        tf_sum = np.asarray(x.sum(axis=1).reshape(
            x.shape[0] // len(tf_list), len(tf_list)).sum(axis=0))
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
    return x, tf_list


def filter_by_gene(x, y, gene_list):
    filtered_idx = y[y['Gene'].isin(gene_list)].index
    if issparse(x):
        mask = np.zeros(x.shape[0], dtype='bool')
        for idx in filtered_idx:
            mask[idx * 359: (idx + 1) * 359] = True
        x = x[mask]
    else:
        x = x[filtered_idx]
    y = y[filtered_idx].reset_index(drop=True)
    return x, y


def filter_less_than_3(x, y):
    """ remove genes with less than 3 tissue """
    gene_count = y.groupby('Gene')['PSI'].count()
    gene_list = gene_count[gene_count > 2].index
    x, y = filter_by_gene(x, y, gene_list)

    # Create psi table
    psi_df = y.pivot(index='Gene', columns='Tissue')['PSI']

    # Calculate stdev
    psi_stdev = psi_df.std(axis=1, ddof=0)

    # Drop genes with no stdev across different tissues
    keep_gene = psi_stdev[psi_stdev != 0].index
    x, y = filter_by_gene(x, y, keep_gene)
    return x, y


def filter_low_psi_range(x, y):
    """ remove genes with less than 3 tissue """
    # Create psi table
    psi_df = y.pivot(index='Gene', columns='Tissue')['PSI']
    # Cut-off genes with psi range < 0.2
    psi_range = psi_df.max(axis=1) - psi_df.min(axis=1)
    gene_remained_list = psi_range[psi_range >= 0.2]
    x, y = filter_by_gene(x, y, gene_remained_list.index)
    return x, y


def psi_z_score(y: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Convert raw PSI to z-score
    :param x: TF binding profile
    :param Y: Raw PSI value
    :return result: tuple of converted (X, Y)
    """
    print('Performing Z-score transformation...\n')
    print('Converting PSI to z-score...', end='')

    # calculate mean
    psi_gene_group = y.groupby('Gene')
    psi_mean = psi_gene_group['PSI'].mean().to_dict()

    # calculate stdev and convert 0 to nan for prevention of ZeroDivisionError
    psi_sd = psi_gene_group['PSI'].std()
    psi_sd[psi_sd == 0.0] = float('nan')
    psi_sd = psi_sd.to_dict()

    # calculate z-score
    z_score = [(psi - psi_mean[gene]) / psi_sd[gene]
               for gene, psi in zip(y['Gene'], y['PSI'])]
    y = y.assign(PSI=z_score)
    print('DONE!')

    return y


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
        esti_events_num = 0
        for num in self.genes['Tissue']:
            esti_events_num += comb(num, 2)
        esti_events_num = int(esti_events_num)
        print('estimated event numbers:', esti_events_num)

        self.new_x = np.zeros((esti_events_num, len(self.tf_list)))
        self.new_x[:] = np.nan

        self.converter()
        self.new_x = self.new_x[~np.isnan(
            self.new_x).any(axis=1)].astype('bool')
        self.new_y['Gene'] = self.new_y['Gene'].astype('category')

    def output(self):
        return self.new_x, self.new_y

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
        for gene in self.genes.index:
            event_ind_list = self.y[self.y['Gene'] == gene].index
            for event_1, event_2 in combinations(event_ind_list, 2):
                # delta_psi = abs(y.at[event_1, 'PSI'] - y.at[event_2, 'PSI'])
                delta_psi = abs(
                    self.y.at[event_1, 'ZPSI'] - self.y.at[event_2, 'ZPSI'])
                if delta_psi:
                    delta_feature = self.x[event_1] ^ self.x[event_2]
                    delta_psi_group = self.y.at[event_1,
                                                'psi_group'] ^ self.y.at[event_2, 'psi_group']
                else:
                    continue
                self.new_x[i] = delta_feature
                new_y_dpsi.append(delta_psi)
                new_psi_group.append(delta_psi_group)
                new_y_gene.append(gene)
                i += 1
        self.new_y = pd.DataFrame(
            data={'Gene': new_y_gene, 'PSI': new_y_dpsi, 'psi_group': new_psi_group})
        self.new_y['psi_group'] = self.new_y['psi_group'].astype('category')


class QuantDeltaConverter_sparse(_DeltaConverter):
    """Apply Quantile categorize 20% with sparse matrix"""

    def converter(self):
        if issparse(self.x):
            pass
        else:
            print('Error: X_data is not sparse matrix')
            return
        i = 0
        new_y_dpsi = []
        new_y_gene = []
        new_psi_group = []
        for gene in self.genes.index:
            event_ind_list = self.y[self.y['Gene'] == gene].index
            for event_1, event_2 in combinations(event_ind_list, 2):
                # delta_psi = abs(y.at[event_1, 'PSI'] - y.at[event_2, 'PSI'])
                delta_psi = abs(
                    self.y.at[event_1, 'ZPSI'] - self.y.at[event_2, 'ZPSI'])
                if delta_psi:
                    delta_feature = self.x[event_1] - self.x[event_2]
                    delta_psi_group = self.y.at[event_1,
                                                'psi_group'] ^ self.y.at[event_2, 'psi_group']
                else:
                    continue
                self.new_x[i] = delta_feature
                new_y_dpsi.append(delta_psi)
                new_psi_group.append(delta_psi_group)
                new_y_gene.append(gene)
                i += 1
        self.new_y = pd.DataFrame(
            data={'Gene': new_y_gene, 'PSI': new_y_dpsi, 'psi_group': new_psi_group})
        self.new_y['psi_group'] = self.new_y['psi_group'].astype('category')


class DeltaConverter(_DeltaConverter):
    """Converter without categorized label"""

    def converter(self):
        i = 0
        new_y_dpsi = []
        new_y_gene = []
        for gene in self.genes.index:
            event_ind_list = self.y[self.y['Gene'] == gene].index
            for event_1, event_2 in combinations(event_ind_list, 2):
                # delta_psi = abs(y.at[event_1, 'PSI'] - y.at[event_2, 'PSI'])
                delta_psi = abs(
                    self.y.at[event_1, 'ZPSI'] - self.y.at[event_2, 'ZPSI'])
                if delta_psi:
                    delta_feature = self.x[event_1] ^ self.x[event_2]
                else:
                    continue
                self.new_x[i] = delta_feature
                new_y_dpsi.append(delta_psi)
                new_y_gene.append(gene)
                i += 1
        self.new_y = pd.DataFrame(data={'Gene': new_y_gene, 'PSI': new_y_dpsi})
        self.new_y['psi_group'] = self.new_y['psi_group'].astype('category')


class nn_X_generator(object):
    def __init__(self, x, tf_list):
        self.x = x
        self.len = len(tf_list)

    def __getitem__(self, n):
        data = self.x[n * self.len: (n + 1) * self.len]
        return data


def update_progress_bar(perc: float, option_info: str = None) -> None:
    """
    update progress bar
    :param perc: ratio of current run and whole cycle in percentage
    :option_info: alternative output format
    """
    sys.stdout.write('[{:60}] {:.2f}%, {}\r'.format(
        '=' * int(60 * perc // 100), perc, option_info))
    sys.stdout.flush()


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
