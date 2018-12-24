#!/usr/bin/env python3
import argparse
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split


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
    def __init__(self):
        super(DotaNet, self).__init__()
        self.conv1 = nn.Conv1d(298, 128, 50)
        self.pool1 = nn.MaxPool1d(10)
        self.drop1 = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(245 * 128, 128)
        self.drop2 = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(128, 128)
        self.drop3 = nn.Dropout(p=0.25)
        self.fc3 = nn.Linear(128, 128)
        self.drop4 = nn.Dropout(p=0.25)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.drop1(self.pool1(F.relu(self.conv1(x))))
        x = x.view(-1, 128 * 245)  # flatten
        x = self.drop2(F.relu(self.fc1(x)))
        x = self.drop3(F.relu(self.fc2(x)))
        x = self.drop4(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch', default='128')
    parser.add_argument('-p', '--process', default='6')
    parser.add_argument('-g', '--gpu', default='0')
    args = parser.parse_args()

    print('Loading data...')
    fname = '/dev/shm/wezscore_delta_data.pickle'
    with open(fname, mode='rb') as fh:
        x, y, tf_list = pickle.load(fh)
    x = x.astype(np.float32)
    y = np.array(y['psi_group'], dtype=np.float32).reshape(-1, 1)
    print('DONE!')

    n_tf = len(tf_list)
    full_dataset = DotaDataset(n_tf, x, y)
    train_dataset, test_dataset = random_split(full_dataset, [round(
        len(full_dataset) * 0.8), len(full_dataset) - round(len(full_dataset) * 0.8)])
    train_dataset, val_dataset = random_split(train_dataset, [round(
        len(train_dataset) * 0.875), len(train_dataset) - round(len(train_dataset) * 0.875)])

    n_proc = int(args.process)
    bsize = int(args.batch)
    train_loader = DataLoader(train_dataset, batch_size=bsize, shuffle=True, num_workers=n_proc)
    val_loader = DataLoader(val_dataset, batch_size=bsize, shuffle=True, num_workers=n_proc)
    test_loader = DataLoader(test_dataset, batch_size=bsize, shuffle=True, num_workers=n_proc)

    cuda = torch.device('cuda:' + args.gpu)
    net = DotaNet().to(cuda)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters())
    print('Start Training...')
    for epoch in range(10):
        running_loss = 0.0
        net.train()
        for i, data in enumerate(train_loader, 0):
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

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 200 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

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
            print(f'Accuracy on val dataset after {epoch + 1} epoch: {100 * correct / total}%')

    print('Finished Training')

    with torch.no_grad():
        correct = 0
        total = 0
        net.eval()
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.to(cuda)
            labels = labels.to(cuda)
            outputs = net(inputs)
            predicted = torch.sigmoid(outputs).round()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy on test dataset: {100 * correct / total}%')


if __name__ == "__main__":
    main()
