import errno
import os
import pickle
import sys
from itertools import combinations
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.special import comb


def psi_z_score(X: np.ndarray, Y: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Convert raw PSI to z-score
    :param x: TF binding profile
    :param Y: Raw PSI value
    :return result: tuple of converted (X, Y)
    """
    print('Convert PSI to z-score...', end='')
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


def delta_data_converter(x: np.ndarray, y: pd.Series, tf_list: list) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate data for NN (with position information)
    :param x: feature data for CART model
    :param y: target data for ML
    :param tf_list: transcription factor list
    :param data_order: data order
    :return x: new feature data in delta feature format
    :return y: new target data in delta target format
    """
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
    new_y = []
    for gene in genes.index:
        event_ind_list = y[y['Gene'] == gene].index
        for event_1, event_2 in combinations(event_ind_list, 2):
            delta_psi = abs(y.at[event_1, 'PSI'] - y.at[event_2, 'PSI'])
            if delta_psi:
                delta_feature = x[event_1] ^ x[event_2]
            else:
                continue
            new_x[i] = delta_feature
            new_y.append(delta_psi)
            i += 1
            if i == esti_events_num or i % 1000 == 0:
                update_progress_bar(i / esti_events_num * 100,
                                    '{}/{}'.format(i, esti_events_num))
    print('\nEvents processed: {} , DONE'.format(i))
    x = new_x[~np.isnan(new_x).any(axis=1)]
    y = np.asarray(new_y, dtype='float64')

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
