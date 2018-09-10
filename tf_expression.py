import argparse

import pandas as pd


def summarize_tf_exp(acc_df, input_path):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default='./rna_seq_rsem/',
                        help='Folder of RNA-Seq gene quantification result (RSEM)')
    parser.add_argument('-l', '--list', default='gene_quant_acc.tsv')
    parser.add_argument('-o', '--output', default='tf_exp.tsv',
                        help='Summary of TF expression in tsv format')
    args = parser.parse_args()

    # read list
    acc_df = pd.read_table(args.list)
    tf_exp_df = summarize_tf_exp(acc_df, args.input)


if __name__ == '__main__':
    main()
