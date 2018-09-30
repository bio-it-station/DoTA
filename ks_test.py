#!/usr/bin/env python3
import argparse
import errno
import os
import pickle
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from statsmodels.stats.multitest import multipletests
from utils import update_progress_bar


def parse_options():
    """
    Argument parser
    :param argv: arguments from sys.argv
    :return: arguments
    """
    parser = argparse.ArgumentParser(
        prog=__file__,
        usage='%(prog)s [Input/Output]',
        description='''Process delta data for ks-test''')

    input_output = parser.add_argument_group('Input/Output')
    input_output.add_argument('--i', metavar='<input-file>', required=True, help='delta data')
    input_output.add_argument('--o', metavar='<output-dir>', default='./output/ks_test/',
                              help='Output file directory (default=\'./output/ks_test/\')')

    return parser.parse_args()


def main():
    args = parse_options()

    with open(args.i, mode='rb') as fh:
        X, Y, Tf_list = pickle.load(fh)
        Y = Y['PSI']
    try:
        os.makedirs(args.o)
    except OSError as err:
        if err.errno != errno.EEXIST:
            raise

    np.set_printoptions(precision=3, suppress=True)

    print('Converting delta data...')
    num_data = len(Tf_list)
    ks_result = {}
    for idx, tf in enumerate(Tf_list):
        feature_0 = Y[np.where(X[:, idx] == 0)[0]]
        feature_1 = Y[np.where(X[:, idx] == 1)[0]]

        if feature_0.shape[0] and feature_1.shape[0]:
            ks_result[tf] = ks_2samp(feature_0.values, feature_1.values)

        else:
            continue
        
        # Plot CDF
        label_0 = '0: {}'.format(len(feature_0))
        label_1 = '1: {}'.format(len(feature_1))
        plt.hist(feature_0, 1000, density=True, histtype='step', cumulative=True, label=label_0)
        plt.hist(feature_1, 1000, density=True, histtype='step', cumulative=True, label=label_1)
        plt.legend(loc='upper left')
        plt.savefig('{}{}.png'.format(args.o, tf), dpi=300)

        idx += 1
        update_progress_bar(idx / num_data * 100, '{}/{}'.format(idx, num_data))

    # Convert to pd.dataframe and do pval correction
    df = pd.DataFrame.from_dict(ks_result, orient='index')
    df['adj_pval'] = multipletests(df.pvalue, alpha=0.05, method='bonferroni')[1]
    df.to_csv(args.o + 'ks_test.csv')

    # Move the CDF plots to pass or not_pass subdirectory depending on its adj_pval
    for tf, row in df.iterrows():
        if row.adj_pval < 1e-30:
            category = 'pass'
        else:
            category = 'not_pass'
        os.rename('{}{}.png'.format(args.o, tf), '{}{}/{}.png'.format(args.o, category, tf))

    print()
    print('ks-test complete!')


if __name__ == '__main__':
    main()
