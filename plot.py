import itertools

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def delta_data_boxplot(data, file_out: str) -> None:
    """
    Generate boxplot with delta feature sum and delta psi from delta_data
    :param file_in: input_file_name
    :param file_out: output file name
    """
    data = data[data['delta_feature_sum'] < 50]
    fig = plt.figure(figsize=(12.0, 4.0))
    sns.boxplot(x='delta_feature_sum', y='delta_psi', data=data)
    plt.tight_layout()
    fig.savefig(file_out, dpi=300)
    plt.clf()
    plt.close()
    plt.gcf().clear()


def tf_wise_boxplot(data, file_out: str) -> None:
    # Creates two subplots and unpacks the output array immediately
    labels = ['{}:{}'.format(idx, len(x)) for idx, x in enumerate(data)]

    fig = plt.figure(figsize=(2.0, 4.0), dpi=300)
    ax = sns.boxplot(data=data, showfliers=False)
    # ax.axes.get_xaxis().set_visible(False)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r'$\Delta Z_{\Psi}$', fontsize=14)
    fig.tight_layout()
    fig.savefig('{}_boxplot.png'.format(file_out), transparent=True)
    plt.clf()
    plt.close()
    plt.gcf().clear()


def tf_wise_cdf(data, file_out: str) -> None:
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
