#!/usr/bin/env python3
import argparse
import pickle
from os.path import basename

from scipy.sparse import issparse
import pandas as pd
import numpy as np

from utils import (DeltaConverter, QuantDeltaConverter,
                   QuantDeltaConverterSparse, delete_emtpy_tf,
                   filter_less_than_3, filter_low_psi_range, output,
                   psi_z_score, quantile_convert)


def parse_options():
    """
    Argument parser
    :return: arguments
    """
    parser = argparse.ArgumentParser(
        prog=__file__,
        usage='%(prog)s [Input/Output]',
        description='Do z-score transform from CART-formatted data')

    input_output = parser.add_argument_group('Input/Output')
    input_output.add_argument('--i', metavar='<input-file>', help='CART-formatted data')
    input_output.add_argument('--t', help='Gene-TFBS dictionary')
    input_output.add_argument('--o', metavar='<output-dir>', default='./input/',
                              help='Output file directory (default=\'./input/\')')

    param = parser.add_argument_group('Parameters')
    param = param.add_mutually_exclusive_group(required=True)
    param.add_argument('-z', action='store_true', help='<Optional switch> Normal z-score transformation')
    param.add_argument('-q', action='store_true', help='<Optional switch> Quantile convertion')

    return parser.parse_args()


def z_transform(x, y, tf_list, tfbs_df):
    x, y = filter_less_than_3(x, y, tf_list)
    x, y = filter_low_psi_range(x, y, tf_list)
    x, tf_list = delete_emtpy_tf(x, tf_list)

    y = psi_z_score(y)
    x, y = DeltaConverter(x, y, tf_list, tfbs_df).output()
    return x, y, tf_list


def quantile(x, y, tf_list):
    x, y = filter_less_than_3(x, y, tf_list)
    x, y = filter_low_psi_range(x, y, tf_list)
    x, tf_list = delete_emtpy_tf(x, tf_list)

    x, y = quantile_convert(x, y, tf_list)
    x, y = filter_less_than_3(x, y, tf_list)

    if issparse(x):
        x, y = QuantDeltaConverterSparse(x, y, tf_list).output()
    else:
        x, y = QuantDeltaConverter(x, y, tf_list).output()

    return x, y, tf_list


def main():
    args = parse_options()

    with open(args.i, mode='rb') as fh:
        X, Y, Tf_list = pickle.load(fh)
    with open(args.t, mode='rb') as fh:
        tfbs_dict = pickle.load(fh)
        tfbs_df = (~pd.DataFrame.from_dict(tfbs_dict)[Tf_list]).astype(np.int)

    filename = basename(args.i)
    filename = filename.split('_')
    prefix = filename[:-2]
    datatype = filename[-2:-1]

    if args.z:
        X, Y, Tf_list = z_transform(X, Y, Tf_list, tfbs_df)
        prefix.append('zscore')
    if args.q:
        X, Y, Tf_list = quantile(X, Y, Tf_list)
        prefix.append('quantile')

    datatype.append('delta')
    print('Saving converted delta data...')
    filename = args.o + '_'.join(prefix + datatype + ['data.pickle'])
    output((X, Y, Tf_list), filename)
    print('File saved!')


if __name__ == '__main__':
    main()
