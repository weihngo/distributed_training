import numpy as np
import torch
import torch.utils.data.distributed
import torchvision
from torchvision import transforms
from math import ceil
from random import Random


class Partition(object):
    """ Dataset-like object, but only access a subset of it. """
    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=2020, isNonIID=False, alpha=0):
        self.data = data
        if isNonIID:
            self.partitions, self.ratio = self.__getDirichletData__(data, sizes, seed, alpha)
        else:
            self.partitions = []
            self.ratio = [0] * len(sizes)
            rng = Random()
            rng.seed(seed)
            data_len = len(data)
            indexes = [x for x in range(0, data_len)]
            rng.shuffle(indexes)

            for frac in sizes:
                part_len = int(frac * data_len)
                self.partitions.append(indexes[0:part_len])
                indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])

    def __getNonIIDdata__(self, data, sizes, seed):
        labelList = data.train_labels
        rng = Random()
        rng.seed(seed)
        a = [(label, idx) for idx, label in enumerate(labelList)]
        # Same Part
        labelIdxDict = dict()
        for label, idx in a:
            labelIdxDict.setdefault(label, [])
            labelIdxDict[label].append(idx)
        labelNum = len(labelIdxDict)
        labelNameList = [key for key in labelIdxDict]
        labelIdxPointer = [0] * labelNum
        # sizes = number of nodes
        partitions = [list() for i in range(len(sizes))]
        eachPartitionLen = int(len(labelList) / len(sizes))
        majorLabelNumPerPartition = ceil(labelNum / len(partitions))
        basicLabelRatio = 0.4

        interval = 1
        labelPointer = 0

        # basic part
        for partPointer in range(len(partitions)):
            requiredLabelList = list()
            for _ in range(majorLabelNumPerPartition):
                requiredLabelList.append(labelPointer)
                labelPointer += interval
                if labelPointer > labelNum - 1:
                    labelPointer = interval
                    interval += 1
            for labelIdx in requiredLabelList:
                start = labelIdxPointer[labelIdx]
                idxIncrement = int(basicLabelRatio * len(labelIdxDict[labelNameList[labelIdx]]))
                partitions[partPointer].extend(labelIdxDict[labelNameList[labelIdx]][start:start + idxIncrement])
                labelIdxPointer[labelIdx] += idxIncrement

        # random part
        remainLabels = list()
        for labelIdx in range(labelNum):
            remainLabels.extend(labelIdxDict[labelNameList[labelIdx]][labelIdxPointer[labelIdx]:])
        rng.shuffle(remainLabels)
        for partPointer in range(len(partitions)):
            idxIncrement = eachPartitionLen - len(partitions[partPointer])
            partitions[partPointer].extend(remainLabels[:idxIncrement])
            rng.shuffle(partitions[partPointer])
            remainLabels = remainLabels[idxIncrement:]
        return partitions

    def __getDirichletData__(self, data, psizes, seed, alpha):
        sizes = len(psizes)
        labelList = data.train_labels
        rng = Random()
        rng.seed(seed)
        a = [(label, idx) for idx, label in enumerate(labelList)]
        # Same Part
        labelIdxDict = dict()
        for label, idx in a:
            labelIdxDict.setdefault(label, [])
            labelIdxDict[label].append(idx)
        labelNum = len(labelIdxDict)  # 10
        labelNameList = [key for key in labelIdxDict]
        # rng.shuffle(labelNameList)
        labelIdxPointer = [0] * labelNum
        # sizes = number of nodes
        partitions = [list() for i in range(sizes)]  # of size (m)
        np.random.seed(seed)
        distribution = np.random.dirichlet([alpha] * sizes, labelNum).tolist()  # of size (10, m)

        # basic part
        for row_id, dist in enumerate(distribution):
            subDictList = labelIdxDict[labelNameList[row_id]]
            rng.shuffle(subDictList)
            totalNum = len(subDictList)
            dist = self.handlePartition(dist, totalNum)
            for i in range(len(dist) - 1):
                partitions[i].extend(subDictList[dist[i]:dist[i + 1] + 1])

        # random part
        a = [len(partitions[i]) for i in range(len(partitions))]
        ratio = [a[i] / sum(a) for i in range(len(a))]
        return partitions, ratio

    def handlePartition(self, plist, length):
        newList = [0]
        canary = 0
        for i in range(len(plist)):
            canary = int(canary + length * plist[i])
            newList.append(canary)
        return newList

def partition_dataset(rank, size, args):
    if args.dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(root='../data',
                                                train=True,
                                                download=False,
                                                transform=transform_train)

        partition_sizes = [1.0 / size for _ in range(size)]
        partition = DataPartitioner(trainset, partition_sizes, isNonIID=False, alpha=0.5)
        ratio = partition.ratio
        partition = partition.use(rank)
        train_loader = torch.utils.data.DataLoader(partition,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   pin_memory=True)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        testset = torchvision.datasets.CIFAR10(root='../data',
                                               train=False,
                                               download=False,
                                               transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset,
                                                  batch_size=64,
                                                  shuffle=False,
                                                  num_workers=size)
    return train_loader, test_loader

from torchvision.datasets import CIFAR10
def create_dataloader(data_dir,ngpus_per_node,batch_size):
    #data transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToTensor(),normalize])
    #load dataset
    trainset=CIFAR10(data_dir, train=True, transform=transform, target_transform=None, download=False)
    testset = CIFAR10(data_dir, train=False,download=False, transform=transform)
    #change to dataloader
    train_sampler= torch.utils.data.distributed.DistributedSampler(trainset)
    batch_size=int(batch_size/ngpus_per_node)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=(train_sampler is None), num_workers=8,sampler=train_sampler)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False, num_workers=8)
    return trainloader,testloader,train_sampler

