#!/usr/bin/env python3
import argparse
import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from plot import plot_confusion_matrix
from utils import _getThreads


def parse_options():
    """
    Argument parser
    :return: arguments
    """
    parser = argparse.ArgumentParser(
        prog=__file__,
        usage='%(prog)s ([Input/Output]...) [Data format]...',
        description="""Tree-based classifier for DoTA.""")

    input_output = parser.add_argument_group('Input')
    input_output.add_argument('--i', metavar='<input>',
                              required=True, help='Categorized NN-formatted data')

    param = parser.add_argument_group('Parameters')
    threads = _getThreads()
    param.add_argument('-p', '--process', metavar='<parallel>',
                       choices=range(1, threads + 1), type=int, default='1',
                       help='Number of threads to use (range: 1~{}, default=1)'.format(threads))

    algo = parser.add_argument_group('algorithm')
    algo = algo.add_mutually_exclusive_group(required=True)
    algo.add_argument('-T', action='store_true', help='Extra-tree classifier')
    algo.add_argument('-X', action='store_true', help='XGBoost')

    return parser.parse_args()


def ImbalancedDatasetSampler(x, y):
    # distribution of classes in the dataset
    label_to_count = {}
    for label in y['psi_group']:
        if label in label_to_count:
            label_to_count[label] += 1
        else:
            label_to_count[label] = 1

    # weight for each sample
    prob_lis = np.empty(y.shape[0])
    for key, count in label_to_count.items():
        prob_lis[y['psi_group'] == key] = 0.5 / count

    selected_idx = np.random.choice(
        np.arange(len(y)), replace=True, size=len(y), p=prob_lis)
    return x[selected_idx], y.iloc[selected_idx]


def main():
    args = parse_options()

    with open(args.i, 'rb') as fh:
        x, y, _ = pickle.load(fh)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=9487)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.125, random_state=9487)

    # Data balancing on training set
    x_train, y_train = ImbalancedDatasetSampler(x_train, y_train)

    y_train = np.asarray(y_train['psi_group'], dtype='bool')
    y_test = np.asarray(y_test['psi_group'], dtype='bool')
    y_val = np.asarray(y_val['psi_group'], dtype='bool')

    if args.T:
        clf = ExtraTreesClassifier(n_estimators=200,
                                   min_impurity_decrease=1e-5,
                                   n_jobs=args.process,
                                   random_state=9487)
        clf = clf.fit(x_train, y_train)
    else:
        param_dist = {'booster': 'gbtree',
                      'eta': 0.1,
                      'n_estimators': 2000,
                      'max_depth': 6,
                      'colsample_bytree': 0.5,
                      'objective': 'binary:logistic',
                      'tree_method': 'gpu_exact'}
        clf = XGBClassifier(**param_dist)

        clf = clf.fit(x_train, y_train,
                      eval_set=[(x_val, y_val)],
                      eval_metric='logloss',
                      early_stopping_rounds=5,
                      verbose=True)

    print('Accuracy on train dataset: ',
          accuracy_score(y_train, clf.predict(x_train)))

    y_pred = clf.predict(x_test)
    print('Accuracy on test dataset: ', accuracy_score(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, classes=['Unchange', 'Change'], normalize=True)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')


if __name__ == '__main__':
    main()
