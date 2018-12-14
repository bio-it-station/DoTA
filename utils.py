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


def _getThreads():
    """ Returns the number of available threads on a posix/win based system """
    if sys.platform == 'win32':
        return (int)(os.environ['NUMBER_OF_PROCESSORS'])
    return (int)(os.popen('grep -c cores /proc/cpuinfo').read())


def psi_z_score(X: np.ndarray, Y: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Convert raw PSI to z-score
    :param x: TF binding profile
    :param Y: Raw PSI value
    :return result: tuple of converted (X, Y)
    """
    print('Performing Z-score transformation...\n')
    print('Converting PSI to z-score...', end='')
    # remove genes with less than 2 tissue
    gene_count = Y.groupby('Gene')['PSI'].count()
    gene_list = gene_count[gene_count > 2].index
    filerted_mask = Y['Gene'].isin(gene_list)
    X = X[filerted_mask]
    Y = Y[filerted_mask].reset_index(drop=True)

    # calculate mean
    psi_gene_group = Y.groupby('Gene')
    psi_mean = psi_gene_group['PSI'].mean().to_dict()

    # calculate stdev and convert 0 to nan for prevention of ZeroDivisionError
    psi_sd = psi_gene_group['PSI'].std()
    psi_sd[psi_sd == 0.0] = float('nan')
    psi_sd = psi_sd.to_dict()

    # calculate z-score
    z_score = [(psi - psi_mean[gene]) / psi_sd[gene] for gene, psi in zip(Y['Gene'], Y['PSI'])]
    Y = Y.assign(PSI=z_score)
    print('DONE!')

    return X, Y


class _DeltaConverter(ABC):
    """Basal Class of Delta data converter"""
    def __init__(self, x: np.ndarray, y: pd.DataFrame, tf_list: list):
        self.x = x
        self.y = y
        self.tf_list = tf_list
        gene_count = self.y.groupby('Gene').count()
        self.genes = gene_count[gene_count > 1].dropna()
        self.new_x = None
        self.new_y = None

    def convert(self) -> Tuple[np.ndarray, pd.DataFrame]:
        print('Performing delta data convention...\n')

        # Create an nan array with estimated event numbers as new features array
        esti_events_num = 0
        for num in self.genes['Tissue']:
            esti_events_num += comb(num, 2)
        esti_events_num = int(esti_events_num)
        print('estimated event numbers:', esti_events_num)

        self.new_x = np.zeros((esti_events_num, len(self.tf_list)))
        self.new_x[:] = np.nan

        self.classifier()
        self.new_x = self.new_x[~np.isnan(self.new_x).any(axis=1)].astype('bool')
        self.new_y['Gene'] = self.new_y['Gene'].astype('category')

        return self.new_x, self.new_y

    @abstractmethod
    def classifier(self):
        pass

class QuantDeltaConverter(_DeltaConverter):
    """Apply Quntaile categorize 20%"""
    def classifier(self):
        i = 0
        new_y_dpsi = []
        new_y_gene = []
        new_psi_group = []
        for gene in self.genes.index:
            event_ind_list = self.y[self.y['Gene'] == gene].index
            for event_1, event_2 in combinations(event_ind_list, 2):
                # delta_psi = abs(y.at[event_1, 'PSI'] - y.at[event_2, 'PSI'])
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
        self.new_y = pd.DataFrame(data={'PSI': new_y_dpsi, 'Gene': new_y_gene, 'psi_group': new_psi_group})
        self.new_y['psi_group'] = self.new_y['psi_group'].astype('category')


def delta_data_converter(x: np.ndarray, y: pd.DataFrame, tf_list: list) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Generate data for NN (with position information)
    :param x: feature data for CART model
    :param y: target data for ML
    :param tf_list: transcription factor list
    :param data_order: data order
    :return x: new feature data in delta feature format
    :return y: new target data in delta target format
    """
    print('Performing delta data convention...\n')
    gene_count = y.groupby('Gene').count()
    genes = gene_count[gene_count > 1].dropna()

    # Create an nan array with estimated event numbers as new features array
    esti_events_num = 0
    for num in genes['Tissue']:
        esti_events_num += comb(num, 2)
    esti_events_num = int(esti_events_num)
    print('estimated event numbers:', esti_events_num)

    new_x = np.zeros((esti_events_num, len(tf_list)))
    new_x[:] = np.nan

    # Fill the delta data into array
    i = 0
    new_y_dpsi = []
    new_y_gene = []
    if 'psi_group' in y.columns:
        new_psi_group = []
        for gene in genes.index:
            event_ind_list = y[y['Gene'] == gene].index
            for event_1, event_2 in combinations(event_ind_list, 2):
                # delta_psi = abs(y.at[event_1, 'PSI'] - y.at[event_2, 'PSI'])
                delta_psi = abs(y.at[event_1, 'ZPSI'] - y.at[event_2, 'ZPSI'])
                if delta_psi:
                    delta_feature = x[event_1] ^ x[event_2]
                    delta_psi_group = y.at[event_1, 'psi_group'] ^ y.at[event_2, 'psi_group']
                else:
                    continue
                new_x[i] = delta_feature
                new_y_dpsi.append(delta_psi)
                new_psi_group.append(delta_psi_group)
                new_y_gene.append(gene)
                i += 1
                if i % 1000 == 0:
                    update_progress_bar(i / esti_events_num * 100, '{}/{}'.format(i, esti_events_num))
        y = pd.DataFrame(data={'PSI': new_y_dpsi, 'Gene': new_y_gene, 'psi_group': new_psi_group})
        y['psi_group'] = y['psi_group'].astype('category')
    else:
        for gene in genes.index:
            event_ind_list = y[y['Gene'] == gene].index
            for event_1, event_2 in combinations(event_ind_list, 2):
                # delta_psi = abs(y.at[event_1, 'PSI'] - y.at[event_2, 'PSI'])
                delta_psi = abs(y.at[event_1, 'ZPSI'] - y.at[event_2, 'ZPSI'])
                if delta_psi:
                    delta_feature = x[event_1] ^ x[event_2]
                else:
                    continue
                new_x[i] = delta_feature
                new_y_dpsi.append(delta_psi)
                new_y_gene.append(gene)
                i += 1
                if i % 1000 == 0:
                    update_progress_bar(i / esti_events_num * 100, '{}/{}'.format(i, esti_events_num))
        y = pd.DataFrame(data={'PSI': new_y_dpsi, 'Gene': new_y_gene})
    update_progress_bar(100, '{}/{}'.format(i, i))
    print('\nEvents processed: {} , DONE'.format(i))
    x = new_x[~np.isnan(new_x).any(axis=1)].astype('bool')
    y['Gene'] = y['Gene'].astype('category')

    return x, y


def update_progress_bar(perc: float, option_info: str = None) -> None:
    """
    update progress bar
    :param perc: ratio of current run and whole cycle in percentage
    :option_info: alternative output format
    """
    sys.stdout.write('[{:60}] {:.2f}%, {}\r'.format('=' * int(60 * perc // 100), perc, option_info))
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
