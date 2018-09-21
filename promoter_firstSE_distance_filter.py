#!/usr/bin/env python3
import argparse
from collections import defaultdict


def parse_options():
    """
    Argument parser
    :return: arguments
    """
    parser = argparse.ArgumentParser(
        prog=__file__,
        usage='%(prog)s --[]',
        description='Get distance between promoter to first SE of each genes')

    input_output = parser.add_argument_group('Input/Output')
    input_output.add_argument('--a', metavar='<promoter.bed>', help='Bed file of gene promoter')
    input_output.add_argument('--b', metavar='<first_SE.gff>',
                              help='Annotation generated by CATANA')
    input_output.add_argument('--o', metavar='<output-dir>', default='./',
                              help='Output file directory (default=\'./\')')
    return parser.parse_args()


def main():
    args = parse_options()

    promoter_list = defaultdict(int)
    with open(args.a) as fh:
        for line in fh:
            _, start, end, gid, _, strand = line.rstrip().split()
            if strand == '+':
                promoter_list[gid] = int(end)
            else:
                promoter_list[gid] = int(start)

    Gene_dis = defaultdict(int)
    with open(args.b) as fh:
        for line in fh:
            col = line.rstrip().split()
            if col[2] != 'gene':
                continue
            gid = col[8].split(';')[2][4:]
            gid, _ = gid.split('.')
            if col[6] == '+':
                Gene_dis[gid] = abs(promoter_list[gid] - int(col[3]))
            else:
                Gene_dis[gid] = abs(promoter_list[gid] - int(col[4]))

    filename = args.o + 'promoter_firstSE_dis.txt'
    with open(filename, mode='w') as output:
        for k, v in Gene_dis.items():
            print(k, v, sep='\t', file=output)


if __name__ == '__main__':
    main()
