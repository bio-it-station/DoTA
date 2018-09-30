import pickle

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
    df = pd.DataFrame(np.sum(x, axis=1, dtype=np.int32), columns=['delta_feature_sum'])
    df['delta_psi'] = y

    df = df[df['delta_feature_sum'] < 50]
    fig = plt.figure(figsize=(12.0, 4.0))
    fig = sns.boxplot(x='delta_feature_sum', y='delta_psi', data=df)
    fig.tight_layout()
    fig.savefig(file_out, dpi=300)


def plot_cdf(samp_1, samp_2, file_out: str) -> None:
    label_samp_1 = '0: {}'.format(len(samp_1))
    label_samp_2 = '1: {}'.format(len(samp_2))
    plt.hist(samp_1, 1000, density=True, histtype='step', cumulative=True, label=label_samp_1)
    plt.hist(samp_2, 1000, density=True, histtype='step', cumulative=True, label=label_samp_2)
    plt.legend(loc='upper left')
    plt.savefig('{}.png'.format(file_out), dpi=300)
