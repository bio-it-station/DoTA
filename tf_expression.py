import argparse

import pandas as pd


def summarize_tf_exp(acc_df, tf_id, input_path):
    print(acc_df)


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
    acc_df = pd.read_table(args.list)
    tf_id = pd.read_table(args.tf, index_col=1, header=None).T.to_dict()[0]
    summarize_tf_exp(acc_df, tf_id, args.input)


if __name__ == '__main__':
    main()
