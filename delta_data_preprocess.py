import argparse
import errno
import os
import pickle
import sys
from itertools import combinations

import numpy as np
from scipy.special import comb


def parse_options():
    """
    Argument parser
    :param argv: arguments from sys.argv
    :return: arguments
    """
    parser = argparse.ArgumentParser(
        prog=__file__,
        usage='%(prog)s [Input/Output]',
        description='''Generate delta data from CART-formatted data''')

    input_output = parser.add_argument_group('Input/Output')
    input_output.add_argument(
        '--i', metavar='<input-file>', help='CART-formatted data')
    input_output.add_argument(
        '--o', metavar='<output-dir>', default='./input/',
        help='Output file directory (default=\'./input/\')')

    return parser.parse_args()


def delta_data_converter(x, y, tf_list, data_order):
    """
    Generate data for NN (with position information)
    :param x: feature data for CART model
    :param y: target data for ML
    :param tf_list: transcription factor list
    :param data_order: data order
    :return x: new feature data in delta feature format
    :return y: new target data in delta target format
    """
    genes = data_order.groupby('Gene').count()[data_order.groupby('Gene').count() > 1].dropna()

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
        event_ind_list = data_order.index[(data_order['Gene'] == gene) == True].tolist()
        for event_1, event_2 in list(combinations(event_ind_list, 2)):
            delta_psi = abs(y[event_1] - y[event_2])
            if delta_psi:
                delta_feature = x[event_1] ^ x[event_2]
            else:
                continue
            new_x[i] = delta_feature
            new_y.append(delta_psi)
            i += 1
            if i == esti_events_num or i % 1000 == 0:
                update_progress_bar(i / esti_events_num * 100, '{}/{}'.format(i, esti_events_num))
    print('\nEvents processed: {} , DONE'.format(i))
    x = new_x[~np.isnan(new_x).any(axis=1)]
    y = np.asarray(new_y, dtype='float16')

    return x, y


def update_progress_bar(perc, option_info=None):
    """
    update progress bar
    :param perc: ratio of current run and whole cycle in percentage
    :option_info: alternative output format
    """
    sys.stdout.write(
        '[{:60}] {:.2f}%, {}\r'.format('=' * int(60 * perc // 100),
                                       perc,
                                       option_info))
    sys.stdout.flush()


def output(var, filename):
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


def main():
    args = parse_options()

    with open(args.i, mode='rb') as fh:
        X, Y, Tf_list, Data_order = pickle.load(fh)

    print('Converting delta data...')
    X, Y = delta_data_converter(X, Y, Tf_list, Data_order)
    print('Delta data coverting complete!')

    filename = args.o + 'delta_data.pickle'
    output((X, Y, Tf_list), filename)


if __name__ == '__main__':
    main()
