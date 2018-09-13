#!/usr/bin/env python3
import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

from utils import (delta_data_converter, output, psi_z_score,
                   update_progress_bar)


def parse_options():
    """
    Argument parser
    :param argv: arguments from sys.argv
    :return: arguments
    """
    parser = argparse.ArgumentParser(
        prog=__file__,
        usage='%(prog)s ([Input/Output]...) [Data format]...',
        description="""Generate the feature and target data for machine-learning framework.
        The feature data converted from tf-promoter-peaks, and the target data from MISO-reported
        psi of first splicing event of each genes. The data will be converted by numpy and scipy
        into array for neural network (NN) or Classification And Regression Tree (CART).""")

    input_output = parser.add_argument_group('Input/Output')
    input_output.add_argument('--f', metavar='<feature-dir>', default='./input/features/',
                              help='Features files directory (default=\'./input/features/\')')
    input_output.add_argument('--t', metavar='<target-dir>', default='./input/targets/',
                              help='Targets files directory (default=\'./input/targets/\')')
    input_output.add_argument('--w', metavar='<tf_weight>', help='TF weighting matrix files')
    input_output.add_argument('--o', metavar='<output-dir>', default='./input/',
                              help='Output file directory (default=\'./input/\')')

    param = parser.add_argument_group('Parameters')
    param.add_argument('-z', action='store_true', help='<Optional switch> Output z-score of PSI')
    param.add_argument('-d', action='store_true', help='<Optional switch> Convert delta_data')

    # file_format = parser.add_argument_group('Data format')
    # file_format = file_format.add_mutually_exclusive_group(required=True)
    # file_format.add_argument(
    #     '-C', action='store_true', help='<Switch> Genarate feature data in CART format')
    # file_format.add_argument(
    #     '-N', action='store_true', help='<Switch> Generate feature data in NN format')

    return parser.parse_args()


def cart_data_converter(features, y, tf_list, tf_weight=None):
    """
    Generate data for CART (without position information)
    :param features: features from dhs_peaks + tf_motifs + promoter region
    :param y: gene-psi pairs for collected tissues
    :param tf_list: transcription factor list
    :param tf_weight: transcription factor weighting matrix
    :return x: feature data for CART model
    """
    num_data = y.shape[0]
    num_tf = len(tf_list)
    x = np.zeros((num_data, num_tf), dtype='bool')

    for i, [gene, _, tissue] in enumerate(y.values):
        for j, tf in enumerate(tf_list):
            full_id = tissue + gene + tf
            if full_id in features:
                x[i][j] = tf_weight.at[tissue, tf]
        i += 1
        if i == num_data or i % 1000 == 0:
            update_progress_bar(i / num_data * 100, '{}/{}'.format(i, num_data))

    return x


def nn_data_converter(features, y, tf_list):
    """
    Generate data for NN (with position information)
    :param features: features from dhs_peaks + tf_motifs + promoter region
    :param y: gene-psi pairs for collected tissues
    :param tf_list: transcription factor list
    :return x: feature data for NN model
    """
    num_data = y.shape[0]
    num_tf = len(tf_list)
    row = []
    col = []

    for i, [gene, _, tissue] in enumerate(y.values):
        for j, tf in enumerate(tf_list):
            full_id = tissue + gene + tf
            if full_id in features:
                for tfbs in features[full_id]:
                    pos = list(range(tfbs[0], tfbs[1]))
                    col += pos
                    row += [i * num_tf + j] * len(pos)
        i += 1
        if i == num_data or i % 1000 == 0:
            update_progress_bar(i / num_data * 100, '{}/{}'.format(i, num_data))

    data = [True] * len(row)
    x = coo_matrix((data, (row, col)), shape=(num_data * num_tf, 2500), dtype='bool')
    x = x.tocsr()

    return x


def main():
    args = parse_options()

    # Read RNA files and generate Y_data and event-genes list
    targets_tissue_list = set()
    targets_path = args.t
    targets_data = []

    print('Reading targets...\n')
    for file in os.listdir(targets_path):
        if file.endswith('.target'):
            tissue_name = file.split('.')[0]
            targets_tissue_list.add(tissue_name)
            print(tissue_name + '...', end='')
            with open(targets_path + file) as fh:
                for line in fh:
                    targets_data.append(line.strip().split() + [tissue_name])
            print('DONE!')

    Y = pd.DataFrame(targets_data, columns=['Gene', 'PSI', 'Tissue'])
    Y['PSI'] = Y['PSI'].astype('float64')
    Y['Tissue'] = Y['Tissue'].astype('category')
    Y = Y.sort_values(['Tissue', 'Gene'])
    Y = Y.reset_index(drop=True)

    # Read data from feature files and build index for tf binding position
    Tf_list = set()
    Gene_list = set()
    features_tissue_list = set()
    features_path = args.f
    features_data = defaultdict(list)

    print('\n' + '-' * 60 + '\n')
    print('Reading features...\n')
    for file in os.listdir(features_path):
        if file.endswith('.feature'):
            tissue_name = file.split('.')[0]
            features_tissue_list.add(tissue_name)
            print(tissue_name + '...', end='')
            with open(features_path + file) as fh:
                for line in fh:
                    col = line.rstrip().split()
                    Gene_list.add(col[0])
                    Tf_list.add(col[3])
                    features_data[tissue_name + col[0] + col[3]].append((int(col[1]), int(col[2])))
            print('DONE!')

    Tf_list = sorted(Tf_list)
    Gene_list = sorted(Gene_list)

    if features_tissue_list != targets_tissue_list:
        print('Error: features and targets didn\'t matched')
        exit(1)

    # Loading transcription factor weighting matrix file
    if args.w:
        Tf_weight = pd.read_table(args.w, index_col=0)
        Tf_weight = Tf_weight > 1
        prefix = 'weight_'
    else:
        Tf_weight = pd.DataFrame(True, index=targets_tissue_list, columns=Tf_list)
        prefix = ''

    # Convert data for NN or CART
    print('\n' + '-' * 60 + '\n')
    print('Converting data...')
    X = cart_data_converter(features_data, Y, Tf_list, Tf_weight)
    datatype = 'rf_data'
    print('\nDONE!')
    print('Saving converted data...')
    # filename = args.o + prefix + 'rf' + ('_zscore' if args.z else '') + '_data.pickle'
    filename = args.o + prefix + datatype + '.pickle'
    output((X, Y, Tf_list), filename)
    print('File saved!')

    if args.z:
        print('\n' + '-' * 60 + '\n')
        X, Y = psi_z_score(X, Y)
        prefix = prefix + 'zscore_'

    if args.d:
        print('\n' + '-' * 60 + '\n')
        X, Y = delta_data_converter(X, Y, Tf_list)
        datatype = 'delta_data'
        print('Saving converted delta data...')
        filename = args.o + prefix + datatype + '.pickle'
        output((X, Y, Tf_list), filename)
        print('File saved!')


if __name__ == '__main__':
    main()
