import itertools
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def delta_data_boxplot(file_in: str, file_out: str) -> None:
    """
    Generate boxplot with delta feature sum and delta psi from delta_data
    :param file_in: input_file_name
    :param file_out: output file name
    """
    with open(file_in, mode='rb') as fh:
        x, y, _ = pickle.load(fh)
        y = y['PSI']
    df = pd.DataFrame(np.sum(x, axis=1, dtype=np.int32),
                      columns=['delta_feature_sum'])
    df['delta_psi'] = y

    df = df[df['delta_feature_sum'] < 50]
    fig = plt.figure(figsize=(12.0, 4.0))
    sns.boxplot(x='delta_feature_sum', y='delta_psi', data=df)
    plt.tight_layout()
    fig.savefig(file_out, dpi=300)
    plt.clf()
    plt.close()
    plt.gcf().clear()


def plot_cdf(data, file_out: str) -> None:
    labels = ['{}:{}'.format(idx, len(x)) for idx, x in enumerate(data)]

    _, ax = plt.subplots()
    for d, label in zip(data, labels):
        n = np.arange(1, len(d) + 1) / np.float(len(d))
        Xs = np.sort(d)
        ax.step(Xs, n, label=label)

    plt.legend(loc='upper left')
    plt.savefig('{}.png'.format(file_out), dpi=300)
    plt.clf()
    plt.close()
    plt.gcf().clear()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
