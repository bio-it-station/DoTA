#!/usr/bin/env python3
import argparse

import pandas as pd


def summarize_tf_exp(acc_df, tf_id, input_path):
    tf_name = list(tf_id.keys())
    tf_id = list(tf_id.values())
    result_dict = {}

    for _, (tissue, acc_list) in acc_df.iterrows():
        acc_list = acc_list.split(',')
        for acc in acc_list:
            df = pd.read_table(input_path + acc + '.tsv', index_col=0)
            tf_exp = df['TPM'][tf_id].values
            result_dict[tissue] = tf_exp

    return pd.DataFrame.from_dict(result_dict, orient='index', columns=tf_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default='./rna_seq_rsem/',
                        help='Folder of RNA-Seq gene quantification result (RSEM)')
    parser.add_argument('-l', '--list', default='gene_quant_acc.tsv',
                        help='List of accession from ENCODE')
    parser.add_argument('-t', '--tf', default='tf_id_name.tsv',
                        help='TF name and it\'s Ensembl gene ID')
    parser.add_argument('-o', '--output', default='tf_exp.tsv',
                        help='Summary of TF expression in tsv format')
    args = parser.parse_args()

    # read list
    acc_df = pd.read_table(args.list, header=None)
    tf_id = pd.read_table(args.tf, header=None, index_col=1).iloc[:, 0].to_dict()
    # summarize data into DataFrame
    tf_exp_df = summarize_tf_exp(acc_df, tf_id, args.input)
    tf_exp_df.to_csv(args.output, sep='\t')


if __name__ == '__main__':
    main()
