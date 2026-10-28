import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from collections import OrderedDict
import math
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

import glob as glob
import time

from torchsummary import summary
from IPython.display import clear_output
import collections


class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(5 * 5 * 64, num_classes)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = self.relu(self.max_pool(self.conv1(x)))
        x = self.relu(self.max_pool(self.conv2_drop(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16  # 16
        self.conv = conv3x3(3, 16)  # 16
        self.bn = nn.BatchNorm2d(16)  # 16
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 8, layers[0])  # 8
        self.layer2 = self.make_layer(block, 32, layers[1], 2)  # 32
        self.layer3 = self.make_layer(block, 64, layers[2], 2)  # 64
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)  # 64

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out1 = self.relu(out)
        out2 = self.layer1(out1)  # out
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out = self.avg_pool(out4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out  # , (out1,out2,out3,out4)


class MLP1x(nn.Module):
    def __init__(self, dim, hidd, num_classes=10):
        super(MLP1x, self).__init__()
        self.fc1 = nn.Linear(dim, hidd)
        self.fc2 = nn.Linear(hidd, num_classes)
        self.relu = nn.ReLU(inplace=True)
        # self.fc2 = nn.Linear(hidd, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class LogReg(nn.Module):
    def __init__(self, dim, num_classes=10):
        super(LogReg, self).__init__()
        self.fc1 = nn.Linear(dim, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        return out


class Bottleneck_nobn(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck_nobn, self).__init__()
        # self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4 * growth_rate, kernel_size=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        # out = self.conv1(F.relu(self.bn1(x)))
        # out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv1(F.relu(x))
        out = self.conv2(F.relu(out))
        out = torch.cat([out, x], 1)
        return out


class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        # out = self.conv1(F.relu(x))
        # out = self.conv2(F.relu(out))
        out = torch.cat([out, x], 1)
        return out


class Transition_nobn(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition_nobn, self).__init__()
        # self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        # out = self.conv(F.relu(self.bn(x)))
        out = self.conv(F.relu(x))
        out = F.avg_pool2d(out, 2)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        # out = self.conv(F.relu(x))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, block, transition, nblocks, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans1 = transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans2 = transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans3 = transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3] * growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        # out = F.avg_pool2d(F.relu(out), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def param_proc(model, images, labels):
    outputs = model(images)

    loss = criterion(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    # optimizer.step()

    dx_all = []

    got = 0
    for f in model.parameters():
        if got == 0:
            dx = f.grad.cpu().detach().numpy().flatten()
            dx_all.append(f.grad.cpu().detach().numpy())
            dnorm = np.linalg.norm(f.data.cpu().detach().numpy())
            got = 1
        else:
            dx = np.append(dx, f.grad.cpu().detach().numpy().flatten())
            dx_all.append(f.grad.cpu().detach().numpy())
            act = f.data.cpu().detach().numpy()
            dnorm = np.append(dnorm, np.linalg.norm(f.data.cpu().detach().numpy()))
    return outputs, dx, dnorm, dx_all


def grad_proc(criterion, optimizer, model, images, labels):
    outputs = model(images)
    loss = criterion(outputs, labels)
    optimizer.zero_grad()
    loss.backward()

    got = 0
    for f in model.parameters():
        if got == 0:
            dx = f.grad.cpu().detach().numpy().flatten()
            # dx_all.append(f.grad.cpu().detach().numpy())
            # dnorm = np.linalg.norm(f.data.cpu().detach().numpy())
            got = 1
        else:
            dx = np.append(dx, f.grad.cpu().detach().numpy().flatten())
            # dx_all.append(f.grad.cpu().detach().numpy())
            # act = f.data.cpu().detach().numpy()
            # dnorm = np.append(dnorm,np.linalg.norm(f.data.cpu().detach().numpy()))
    return dx


def torch_cifar_loader():
    transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor()])

    train_dataset = torchvision.datasets.CIFAR10(root='data/',
                                                 train=True,
                                                 transform=transform,
                                                 download=True)

    valid_dataset = torchvision.datasets.CIFAR10(root='data/',
                                                 train=True,
                                                 transform=transform,
                                                 download=True)

    test_dataset = torchvision.datasets.CIFAR10(root='data/',
                                                train=False,
                                                transform=transforms.ToTensor(),
                                                download=True)

    idx = np.random.permutation(50000)

    train_dataset.data = train_dataset.data[idx[:25000]]
    train_dataset.targets = list(np.array(train_dataset.targets)[idx[:25000]])

    valid_dataset.data = valid_dataset.data[idx[25000:]]
    valid_dataset.targets = list(np.array(valid_dataset.targets)[idx[25000:]])

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=100,
                                               shuffle=True)

    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                               batch_size=100,
                                               shuffle=False)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=100,
                                              shuffle=False)

    grad_valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                                    batch_size=1,
                                                    shuffle=False)

    grad_test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                   batch_size=1,
                                                   shuffle=False)

    return train_loader, valid_loader, test_loader, grad_valid_loader, grad_test_loader


def torch_cifar_loader_basic(bs=100):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='data/',
                                                 train=True,
                                                 transform=transform_train,
                                                 download=True)

    valid_dataset = torchvision.datasets.CIFAR10(root='data/',
                                                 train=True,
                                                 transform=transform_train,
                                                 download=True)

    test_dataset = torchvision.datasets.CIFAR10(root='data/',
                                                train=False,
                                                transform=transform_test,
                                                download=True)

    train_dataset.data = train_dataset.data[:25000]
    train_dataset.targets = list(np.array(train_dataset.targets)[:25000])

    valid_dataset.data = valid_dataset.data[25000:]
    valid_dataset.targets = list(np.array(valid_dataset.targets)[25000:])

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=bs,
                                               shuffle=True)

    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                               batch_size=bs,
                                               shuffle=False)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=bs,
                                              shuffle=False)

    grad_valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                                    batch_size=1,
                                                    shuffle=False)

    grad_test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                   batch_size=1,
                                                   shuffle=False)

    return train_loader, valid_loader, test_loader, grad_valid_loader, grad_test_loader


def torch_cifar_loader_basic_small():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='data/',
                                                 train=True,
                                                 transform=transform_train,
                                                 download=True)

    valid_dataset = torchvision.datasets.CIFAR10(root='data/',
                                                 train=True,
                                                 transform=transform_train,
                                                 download=True)

    test_dataset = torchvision.datasets.CIFAR10(root='data/',
                                                train=False,
                                                transform=transform_test,
                                                download=True)

    train_dataset.data = train_dataset.data[:25000]
    train_dataset.targets = list(np.array(train_dataset.targets)[:25000])

    valid_dataset.data = valid_dataset.data[45000:]
    valid_dataset.targets = list(np.array(valid_dataset.targets)[45000:])

    test_dataset.data = test_dataset.data[:1000]
    test_dataset.targets = list(np.array(test_dataset.targets)[:1000])

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=100,
                                               shuffle=True)

    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                               batch_size=100,
                                               shuffle=False)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=100,
                                              shuffle=False)

    grad_valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                                    batch_size=1,
                                                    shuffle=False)

    grad_test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                   batch_size=1,
                                                   shuffle=False)

    return train_loader, valid_loader, test_loader, grad_valid_loader, grad_test_loader


def torch_cifar_loader_small():
    transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor()])

    train_dataset = torchvision.datasets.CIFAR10(root='data/',
                                                 train=True,
                                                 transform=transform,
                                                 download=True)

    valid_dataset = torchvision.datasets.CIFAR10(root='data/',
                                                 train=True,
                                                 transform=transform,
                                                 download=True)

    test_dataset = torchvision.datasets.CIFAR10(root='data/',
                                                train=False,
                                                transform=transforms.ToTensor(),
                                                download=True)

    idx = np.random.permutation(50000)

    train_dataset.data = train_dataset.data[idx[:25000]]
    train_dataset.targets = list(np.array(train_dataset.targets)[idx[:25000]])

    valid_dataset.data = valid_dataset.data[idx[49000:]]
    valid_dataset.targets = list(np.array(valid_dataset.targets)[idx[49000:]])

    idx = np.random.permutation(10000)

    test_dataset.data = test_dataset.data[idx[9000:]]
    test_dataset.targets = list(np.array(test_dataset.targets)[idx[9000:]])

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=100,
                                               shuffle=True)

    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                               batch_size=100,
                                               shuffle=False)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=100,
                                              shuffle=False)

    grad_valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                                    batch_size=1,
                                                    shuffle=False)

    grad_test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                   batch_size=1,
                                                   shuffle=False)

    return train_loader, valid_loader, test_loader, grad_valid_loader, grad_test_loader


def grad_stats(model, device, criterion, optimizer, grad_valid_loader, dbg, target=-1):
    i = 0
    valid_set_size = len(grad_valid_loader.dataset.targets)

    if target > 0:
        for images, labels in grad_valid_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            if target == -1:
                act_labels = torch.tensor([int((np.random.random() - 0.0000001) * 10)]).to(device)
            else:
                act_labels = torch.tensor([target]).to(device)

            loss = criterion(outputs, act_labels)
            optimizer.zero_grad()
            loss.backward()

            n = 0
            for f in model.parameters():
                if n == 0:
                    dx = f.grad.reshape(-1)
                else:
                    dx2 = f.grad.reshape(-1)
                    dx = torch.cat((dx, dx2), 0)
                n += 1
            if i == 0:
                dim = len(dx)
                dx_max = dx
                dx_min = dx
                dx_avg = dx / valid_set_size
            else:
                dx_max = torch.max(torch.cat((dx.view(-1, dim), dx_max.view(-1, dim)), 0), 0).values
                dx_min = torch.min(torch.cat((dx.view(-1, dim), dx_max.view(-1, dim)), 0), 0).values
                dx_avg = torch.add(dx_avg, dx / valid_set_size)
            i += 1

        i = 0
        for images, labels in grad_valid_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            if target == -1:
                act_labels = torch.tensor([int((np.random.random() - 0.0000001) * 10)]).to(device)
            else:
                act_labels = torch.tensor([target]).to(device)

            loss = criterion(outputs, act_labels)
            optimizer.zero_grad()
            loss.backward()

            n = 0
            for f in model.parameters():
                if n == 0:
                    dx = f.grad.reshape(-1)
                else:
                    dx2 = f.grad.reshape(-1)
                    dx = torch.cat((dx, dx2), 0)
                n += 1
            if i == 0:
                dx_std = torch.pow(dx - dx_avg, 2) / valid_set_size
            else:
                dx_std = torch.add(dx_std, torch.pow(dx - dx_avg, 2) / valid_set_size)
            i += 1

        dx_std = torch.sqrt(dx_std)
        for k in range(len(dx_std)):
            if dx_std[k] == 0:
                dx_std[k] = 1

        dx_diff = dx_max - dx_min
        for k in range(len(dx_diff)):
            if dx_diff[k] == 0:
                dx_diff[k] = 1
    else:
        valid_set_size = 10 * valid_set_size
        for t in range(10):
            for images, labels in grad_valid_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)

                act_labels = torch.tensor([t]).to(device)

                loss = criterion(outputs, act_labels)
                optimizer.zero_grad()
                loss.backward()

                n = 0
                for f in model.parameters():
                    if n == 0:
                        dx = f.grad.reshape(-1)
                    else:
                        dx2 = f.grad.reshape(-1)
                        dx = torch.cat((dx, dx2), 0)
                    n += 1
                if i == 0:
                    dim = len(dx)
                    dx_max = dx
                    dx_min = dx
                    dx_avg = dx / valid_set_size
                else:
                    dx_max = torch.max(torch.cat((dx.view(-1, dim), dx_max.view(-1, dim)), 0), 0).values
                    dx_min = torch.min(torch.cat((dx.view(-1, dim), dx_max.view(-1, dim)), 0), 0).values
                    dx_avg = torch.add(dx_avg, dx / valid_set_size)
                i += 1

            i = 0
            for images, labels in grad_valid_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)

                act_labels = torch.tensor([target]).to(device)

                loss = criterion(outputs, act_labels)
                optimizer.zero_grad()
                loss.backward()

                n = 0
                for f in model.parameters():
                    if n == 0:
                        dx = f.grad.reshape(-1)
                    else:
                        dx2 = f.grad.reshape(-1)
                        dx = torch.cat((dx, dx2), 0)
                    n += 1
                if i == 0:
                    dx_std = torch.pow(dx - dx_avg, 2) / valid_set_size
                else:
                    dx_std = torch.add(dx_std, torch.pow(dx - dx_avg, 2) / valid_set_size)
                i += 1

            dx_std = torch.sqrt(dx_std)
            for k in range(len(dx_std)):
                if dx_std[k] == 0:
                    dx_std[k] = 1

            dx_diff = dx_max - dx_min
            for k in range(len(dx_diff)):
                if dx_diff[k] == 0:
                    dx_diff[k] = 1
    return dx_max, dx_min, dx_diff, dx_avg, dx_std


def base_train_step(model, device, optimizer, criterion, train_loader):
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return


def base_model_eval(model, device, test_loader, dbg):
    total = 0
    correct = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # print("base model acc: ",correct,total,correct/total)
    # print("base model acc: ",correct,total,correct/total,file=dbg)
    # dbg.flush()
    return correct, total