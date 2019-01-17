#!/usr/bin/env python3
import argparse
import errno
import gc
import os
import pickle
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, dump, load
from scipy.stats import ks_2samp
from statsmodels.stats.multitest import multipletests

from plot import plot_cdf
from utils import _getThreads


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

    param = parser.add_argument_group('Parameters')
    threads = _getThreads()
    param.add_argument('--p', choices=range(1, threads + 1), metavar='<parallel>', type=int, default='1',
                       help='Number of threads to use (range: 1~{}, default=1)'.format(threads))

    return parser.parse_args()


def do_ks_test(X, Y, idx, tf, output_file):
    feature_0 = Y[np.where(X[:, idx] == 0)[0]]
    feature_1 = Y[np.where(X[:, idx] == 1)[0]]
    if feature_0.shape[0] and feature_1.shape[0]:
        data = (feature_0, feature_1)
        plot_cdf(data, output_file)
        return tf, ks_2samp(feature_0, feature_1)
    return None


def main():
    args = parse_options()

    with open(args.i, mode='rb') as fh:
        X, Y, Tf_list = pickle.load(fh)
        Y = Y['PSI'].values
    try:
        os.makedirs(args.o)
    except OSError as err:
        if err.errno != errno.EEXIST:
            raise

    np.set_printoptions(precision=3, suppress=True)

    print('Progressing Ks-test and CDF plots...')

    if args.p != 1:
        # create memmap file in tmpfs
        DEFAULT_TMP_FILE = '/dev/shm'
        # try to use temp folder on tmpfs (/dev/shm)
        temp_folder = DEFAULT_TMP_FILE if os.path.exists(DEFAULT_TMP_FILE) else tempfile.gettempdir()
        try:
            pool_tmp = tempfile.mkdtemp(dir=temp_folder)
            # load X to mmap
            fname = os.path.join(pool_tmp, 'X.mmap')
            dump(X, fname)
            X = load(fname, mmap_mode='r')
            # load Y to mmap
            fname = os.path.join(pool_tmp, 'Y.mmap')
            dump(Y, fname)
            Y = load(fname, mmap_mode='r')
            gc.collect()
            # set OPENBLAS_NUM_THREADS=1 to prevent over-subscription
            os.environ['OPENBLAS_NUM_THREADS'] = '1'
        except (IOError, OSError):
            print('failed to open temp file on ' + temp_folder, file=sys.stderr)
            args.p = 1

    # create pool and exec
    results = Parallel(n_jobs=args.p)(
        delayed(do_ks_test)(
            X, Y, idx, tf, args.o + tf
        ) for idx, tf in enumerate(Tf_list)
    )
    ks_result = {result[0]: result[1] for result in results if result if not None}
    print('\nks-test complete!')

    # cleanup
    if args.p != 1:
        try:
            shutil.rmtree(pool_tmp)
        except (IOError, OSError):
            print('failed to clean-up mmep automatically', file=sys.stderr)

    # Convert to pd.dataframe and do pval correction
    df = pd.DataFrame.from_dict(ks_result, orient='index')
    df['adj_pval'] = multipletests(df.pvalue, alpha=0.05, method='bonferroni')[1]
    df.to_csv(args.o + 'ks_test.csv')

    # Move the CDF plots to accept or reject subdirectory depending on its adj_pval
    print('\nMoving files to corresponding folder...')
    for tf, row in df.iterrows():
        if row.adj_pval < 1e-30:
            category = 'accept'
        else:
            category = 'reject'
        try:
            os.makedirs(args.o + category)
        except OSError as err:
            if err.errno != errno.EEXIST:
                raise
        os.rename('{}{}.png'.format(args.o, tf), '{}{}/{}.png'.format(args.o, category, tf))

    print('\nComplete!')


if __name__ == '__main__':
    main()
