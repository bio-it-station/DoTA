#!/usr/bin/env python3
import os
import pickle
from collections import defaultdict
from multiprocessing import Manager, Pool

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


def chunk_size(dividend, divisor):
    quotient, remainder = divmod(dividend, divisor)
    if 0 != remainder:
        arr = np.array([quotient] * divisor + [remainder])
    else:
        arr = np.array([quotient] * divisor)
    return arr


def read_feature(filename):
    result = defaultdict(list)
    if filename.endswith('.feature'):
        tissue_name = filename.split('.')[0]
        features_tissue_list.add(tissue_name)
    print(tissue_name + '...', end='')
    with open(features_path + filename) as fh:
        for line in fh:
            col = line.rstrip().split()
            Gene_list.add(col[0])
            Tf_list.add(col[3])
            result[tissue_name + col[0] + col[3]].append((int(col[1]), int(col[2])))
    print('DONE!')
    return result


def cart_data_converter(size, num_tf, y, tf_list, features, tf_weight):
    arr = np.zeros((size, num_tf), dtype='bool')
    for i, [gene, _, tissue] in enumerate(y.values):
        for j, tf in enumerate(tf_list):
            full_id = tissue + gene + tf
            if full_id in features:
                arr[i][j] = tf_weight.at[tissue, tf]
    print('DONE!')

    return arr
    # for j, tf in enumerate(tf_list):
    #     full_id = tissue + gene + tf
    #     if full_id in features:
    #         output[j] = tf_weight.at[tissue, tf]


if __name__ == '__main__':
    targets_tissue_list = set()
    targets_path = './input/targets/'
    targets_data = []

    print('Reading targets...\n')
    for file in os.listdir(targets_path):
        if file.endswith('.target'):
            tissue_name = file.split('.')[0]
            targets_tissue_list.add(tissue_name)
            print(tissue_name + '...', end='')
            with open(targets_path + file) as fh:
                for line in fh:
                    targets_data.append(line.strip().split() + [tissue_name])
            print('DONE!')

    Y = pd.DataFrame(targets_data, columns=['Gene', 'PSI', 'Tissue'])
    Y['PSI'] = Y['PSI'].astype('float64')
    Y['Tissue'] = Y['Tissue'].astype('category')
    Y = Y.sort_values(['Tissue', 'Gene'])
    Y = Y.reset_index(drop=True)

    Tf_list = set()
    with open('../DeepTFAS-in-Human/TF/361-tf') as fh:
        for line in fh:
            Tf_list.add(line.rstrip())
    Gene_list = set()
    features_tissue_list = set()
    features_path = './input/features/'
    print('\n' + '-' * 60 + '\n')

    print('Reading features...\n')
    feature_results = Parallel(n_jobs=-1)(
        delayed(read_feature)(filename)
        for filename in os.listdir(features_path)
    )
    manager = Manager()
    # features_data = manager.dict()
    features_data = {}
    for result in feature_results:
        features_data.update(result)
    Tf_list = sorted(Tf_list)
    Gene_list = sorted(Gene_list)
    Tf_weight = pd.DataFrame(True, index=targets_tissue_list, columns=Tf_list)
    print('\n' + '-' * 60 + '\n')

    num_data = Y.shape[0]
    num_tf = len(Tf_list)
    # X = np.zeros((num_data, num_tf), dtype='bool')

    # DEFAULT_TMP_FILE = '/dev/shm'
    # temp_folder = DEFAULT_TMP_FILE
    # pool_tmp = tempfile.mkdtemp(dir=temp_folder)
    # load X to mmap
    # print((num_data, num_tf))
    # X = np.memmap('/dev/shm/X.mmap', dtype='bool', shape=(num_data, num_tf), mode='w+')
    # X[:] = False
    # set OPENBLAS_NUM_THREADS=1 to prevent over-subscription
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    proc = 16

    # data_chunk = chunk_size(num_data, proc)
    # results = []
    # data_start = 0
    # for idx, size in enumerate(data_chunk):
    #     y = Y[data_start: data_start + size]
    #     results.append(cart_data_converter(size, num_tf, y, Tf_list, features_data, Tf_weight))
    #     data_start += size
    # test_output_data = np.concatenate(results)
    # with open('test_temp', 'wb') as fh:
    # pickle.dump(test_output_data, fh)

    data_chunk = chunk_size(num_data, 50)
    # pool = Pool(processes=proc)
    results = []
    data_start = 0
    job_list = []
    for idx, size in enumerate(data_chunk):
        y = Y[data_start: data_start + size]
        job_list.append((cart_data_converter, (size, num_tf, y,
                                               Tf_list, features_data, Tf_weight), {}))
        data_start += size
    output_data = Parallel(n_jobs=proc, verbose=100)(job_list)
    # pool.close()
    # pool.join()
    # res = ([res.get() for res in results])
    # output_data = np.concatenate(res)

    with open('temp', 'wb') as fh:
        pickle.dump(output_data, fh)
    # for idx, row in Y.iterrows():
    #     pool.apply_async(cart_data_converter, (X[idx], Tf_list, features_data, Tf_weight, row['Tissue'], row['Gene']))
    # pool.close()
    # pool.join()

    # shutil.rmtree('/dev/shm/X.mmap')
    # # cleanup
    # try:
    #     shutil.rmtree(pool_tmp)
    # except (IOError, OSError):
    #     print('failed to clean-up mmep automatically', file=sys.stderr)
