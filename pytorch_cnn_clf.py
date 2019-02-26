#!/usr/bin/env python3
import argparse
import os
import pickle

from plot import plot_confusion_matrix

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, Sampler, random_split
from tqdm import tqdm

from utils import _getThreads


def parse_options():
    """
    Argument parser
    :return: arguments
    """
    parser = argparse.ArgumentParser(
        prog=__file__,
        usage='%(prog)s [Input]... ([Parameters]...)',
        description="""NN classifier for DoTA.""")

    input_output = parser.add_argument_group('Input')
    input_output.add_argument('--i', metavar='<input>',
                              required=True, help='Input NN-formatted data')

    param = parser.add_argument_group('Parameters')
    threads = _getThreads()
    param.add_argument('-p', '--process', metavar='<parallel>',
                       choices=range(1, threads + 1), type=int, default='1',
                       help='Number of threads to use (range: 1~{}, default=1)'.format(threads))
    gpu_device = torch.cuda.device_count()
    param.add_argument('-g', '--gpu', metavar='<gpu device>',
                       choices=range(0, gpu_device), type=int, default='0',
                       help='Which gpu device to use (range: 0~{}, default=0)'.format(gpu_device))
    param.add_argument('-b', '--batch', default=128, type=int)
    param.add_argument('-e', '--epoch', default=10, type=int)
    param.add_argument('--checkpoint', help='Prefix of checkoutput output folder')

    model = parser.add_argument_group('Save/Load model')
    model = model.add_mutually_exclusive_group(required=False)
    model.add_argument('-S', metavar='<Save>', help='Path and prefix to save model')
    model.add_argument('-L', metavar='<Load>', help='Load pre-trained model')
    return parser.parse_args()


class DotaDataset(Dataset):
    def __init__(self, n_tf, dataset, target=None):
        self.n_tf = n_tf
        self.dataset = dataset
        self.target = target

    def __len__(self):
        return self.dataset.shape[0] // self.n_tf

    def __getitem__(self, index):
        input_tensor = torch.Tensor(
            self.dataset[index * self.n_tf:(index + 1) * self.n_tf].toarray())
        if self.target is None:
            return input_tensor
        target_tensor = torch.Tensor(self.target[index])
        return input_tensor, target_tensor


class DotaNet(nn.Module):
    def __init__(self, n_tf, con_size=50, dilation=1, pool_size=10):
        super(DotaNet, self).__init__()
        self.con_size = con_size
        self.dilation = dilation
        self.pool_size = pool_size
        self.conv1 = nn.Conv1d(n_tf, 128, self.con_size, dilation=self.dilation)
        self.pool1 = nn.MaxPool1d(self.pool_size)
        self.drop1 = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear((2500 - self.con_size * self.dilation) // self.pool_size * 128, 128)
        self.drop2 = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(128, 128)
        self.drop3 = nn.Dropout(p=0.25)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.drop1(self.pool1(F.relu(self.conv1(x))))
        x = x.view(-1, 128 * (2500 - self.con_size * self.dilation) // self.pool_size)  # flatten
        x = self.drop2(F.relu(self.fc1(x)))
        x = self.drop3(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


class ImbalancedDatasetSampler(Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):

        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset
        dataset_idx = dataset.dataset.target[dataset.indices].ravel()
        label_to_count = {}
        for idx in self.indices:
            label = dataset_idx[idx]
            label_to_count[label] = label_to_count.setdefault(label, 0) + 1

        # weight for each sample
        for key, val in label_to_count.items():
            dataset_idx[dataset_idx == key] = (1.0 / val)
        self.weights = torch.DoubleTensor(dataset_idx)

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


def main():
    args = parse_options()

    print('Loading data...')
    with open(args.i, mode='rb') as fh:
        x, y, tf_list = pickle.load(fh)
    x = x.astype(np.float32)
    y = np.array(y['psi_group'], dtype=np.float32).reshape(-1, 1)
    print('DONE!')

    print('Spliting data...')
    n_tf = len(tf_list)
    full_dataset = DotaDataset(n_tf, x, y)
    torch.manual_seed(9487)
    train_dataset, val_dataset, test_dataset = random_split(full_dataset,
                                                            [round(len(full_dataset) * 0.7),
                                                             round(len(full_dataset) * 0.1),
                                                             round(len(full_dataset) * 0.2)])
    print('DONE!')

    print('Balancing data...')
    balancer = ImbalancedDatasetSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch, sampler=balancer, num_workers=args.process)
    val_loader = DataLoader(val_dataset, batch_size=args.batch,
                            shuffle=True, num_workers=args.process)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch, shuffle=True, num_workers=args.process)
    print('DONE!')

    cuda = torch.device(f'cuda:{args.gpu}')

    if not args.L:
        if args.checkpoint and not os.path.exists(args.checkpoint):
            os.makedirs(args.checkpoint)
        net = DotaNet(n_tf=n_tf, dilation=2).to(cuda)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(net.parameters())
        print('Start Training...')
        for epoch in range(args.epoch):
            net.train()
            # progress bar
            train_loader_pbar = tqdm(
                train_loader, miniters=100,
                bar_format='{desc} [{n_fmt:>4}/{total_fmt}] {percentage:3.0f}%|{bar}|{postfix} [{elapsed}<{remaining}]')
            train_loader_pbar.set_description_str(
                f'Epoch [{epoch + 1:2d}/{args.epoch}]')
            for data in train_loader_pbar:
                inputs, labels = data
                inputs = inputs.to(cuda)
                labels = labels.to(cuda)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loader_pbar.set_postfix_str(f'Loss: {loss.item():.3f}')
            train_loader_pbar.close()

            with torch.no_grad():
                correct = 0
                total = 0
                net.eval()
                for data in val_loader:
                    inputs, labels = data
                    inputs = inputs.to(cuda)
                    labels = labels.to(cuda)
                    outputs = net(inputs)
                    predicted = torch.sigmoid(outputs).round()

                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                print(
                    f'Accuracy on val dataset after {epoch + 1} epoch: {100 * correct / total:.3f}%')

            if args.checkpoint:
                filename = f'{args.checkpoint}/checkpoint_{epoch + 1}.pt'
                torch.save({'epoch': epoch,
                            'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss}, filename)

        print('Finished Training')
        if args.S:
            filename = args.S + '.pt'
            torch.save(net.state_dict(), filename)

    else:
        net = DotaNet(n_tf=n_tf)
        net.load_state_dict(torch.load(args.L, map_location=cuda))
        net.to(cuda)

    with torch.no_grad():
        correct = 0
        total = 0
        y_true = torch.empty(0, 1)
        y_pred = torch.empty(0, 1)
        net.eval()
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.to(cuda)
            labels = labels.to(cuda)
            outputs = net(inputs)
            predicted = torch.sigmoid(outputs).round()

            # Return shuffled true and predict list of labels for confusion matrix
            y_true = torch.cat((y_true, labels.cpu()))
            y_pred = torch.cat((y_pred, predicted.cpu()))

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy on test dataset: {100 * correct / total:.3f}%')

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, classes=['Unchange', 'Change'], normalize=True)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')


if __name__ == "__main__":
    main()
