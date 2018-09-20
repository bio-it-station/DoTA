#!/usr/bin/env python3
import argparse
from plot import delta_data_boxplot


def parse_options():
    """
    Argument parser
    :return: arguments
    """
    parser = argparse.ArgumentParser(
        prog=__file__,
        usage='%(prog)s [Input/Output]',
        description='Generate boxplot with delta feature sum and delta psi from delta_data')

    input_output = parser.add_argument_group('Input/Output')
    input_output.add_argument('--i', metavar='<input-file>', help='delta data')
    input_output.add_argument('--o', metavar='<output-dir>', default='./results/',
                              help='Output file directory (default=\'./results/\')')

    return parser.parse_args()


def main():
    args = parse_options()

    filename = args.o + 'boxplot.png'
    delta_data_boxplot(args.i, filename)


if __name__ == '__main__':
    main()
