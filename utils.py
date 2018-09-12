from typing import Tuple

import numpy as np
import pandas as pd


def psi_z_score(X: np.ndarray, Y: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Convert raw PSI to z-score
    :param x: TF binding profile
    :param Y: Raw PSI value
    :return result: tuple of converted (X, Y)
    """
    print('Convert PSI to z-score...', end='')
    # remove genes with less than 2 tissue
    gene_count = Y.groupby('Gene')['PSI'].count()
    gene_list = gene_count[gene_count > 2].index
    filerted_mask = Y['Gene'].isin(gene_list)
    X = X[filerted_mask]
    Y = Y[filerted_mask].reset_index(drop=True)

    # calculate mean
    psi_gene_group = Y.groupby('Gene')
    psi_mean = psi_gene_group['PSI'].mean().to_dict()

    # calculate stdev and convert 0 to nan for prevention of ZeroDivisionError
    psi_sd = psi_gene_group['PSI'].std()
    psi_sd[psi_sd == 0.0] = float('nan')
    psi_sd = psi_sd.to_dict()

    # calculate z-score
    z_score = [(psi - psi_mean[gene]) / psi_sd[gene] for gene, psi in zip(Y['Gene'], Y['PSI'])]
    Y = Y.assign(PSI=z_score)
    print('DONE!')

    return X, Y
