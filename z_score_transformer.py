#!/usr/bin/env python3
import argparse
import pickle
from os.path import basename

from utils import delta_data_converter, output, psi_z_score


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
    input_output.add_argument('--o', metavar='<output-dir>', default='./input/',
                              help='Output file directory (default=\'./input/\')')

    return parser.parse_args()


def main():
    args = parse_options()

    with open(args.i, mode='rb') as fh:
        X, Y, Tf_list = pickle.load(fh)

    X, Y = psi_z_score(X, Y)
    X, Y = delta_data_converter(X, Y, Tf_list)

    filename = basename(args.i)
    prefix = filename[:-14] + 'zscore_'
    datatype = 'delta_data'
    print('Saving converted delta data...')
    filename = args.o + prefix + datatype + '.pickle'
    output((X, Y, Tf_list), filename)
    print('File saved!')


if __name__ == '__main__':
    main()
