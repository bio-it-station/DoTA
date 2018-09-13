#!/usr/bin/env python3
import argparse
import errno
import os
import pickle

import numpy as np

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
    input_output.add_argument(
        '--i', metavar='<input-file>', help='delta data')
    input_output.add_argument(
        '--o', metavar='<output-dir>', default='./output/ks_test_preprocess/',
        help='Output file directory (default=\'./output/ks_test_preprocess/\')')

    return parser.parse_args()


def main():
    args = parse_options()

    with open(args.i, mode='rb') as fh:
        X, Y, Tf_list = pickle.load(fh)

    try:
        os.makedirs(args.o)
    except OSError as err:
        if err.errno != errno.EEXIST:
            raise

    np.set_printoptions(precision=3, suppress=True)

    print('Converting delta data...')
    num_data = len(Tf_list)
    for n, tf in enumerate(Tf_list):
        feature_0 = Y[np.where(X[:, n] == 0)[0]]
        feature_1 = Y[np.where(X[:, n] == 1)[0]]

        sub_path = '{}{}'.format(args.o, tf)
        with open(sub_path, mode='w') as output:
            print(*feature_0, sep=' ', file=output)
            print(*feature_1, sep=' ', file=output)

        n += 1
        update_progress_bar(n / num_data * 100, '{}/{}'.format(n, num_data))
    print()
    print('ks-test data preprocess complete!')


if __name__ == '__main__':
    main()
