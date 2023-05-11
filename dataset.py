from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, STL10, ImageFolder
from typing import Callable, Optional

import numpy as np


class CIFAR10(CIFAR10):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        triplet: bool = False,
    ) -> None:
        super().__init__(root, train, transform, target_transform, download)
        self.triplet = triplet

    @staticmethod
    def get_transform(train):
        if train:
            return transforms.Compose([
                transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)
            if self.triplet:
                pos_3 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.triplet:
            return (pos_1, pos_2, pos_3), target
        else:
            return (pos_1, pos_2), target

class CIFAR100(CIFAR100):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        triplet: bool = False,
    ) -> None:
        super().__init__(root, train, transform, target_transform, download)
        self.triplet = triplet

    @staticmethod
    def get_transform(train):
        if train:
            return transforms.Compose([
                transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
            ])

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)
            if self.triplet:
                pos_3 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.triplet:
            return (pos_1, pos_2, pos_3), target
        else:
            return (pos_1, pos_2), target

class STL10(STL10):
    def __init__(
        self,
        root: str,
        split: str = "train",
        folds: Optional[int] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        triplet: bool = False,
    ) -> None:
        super().__init__(root, split, folds, transform, target_transform, download)
        self.triplet = triplet

    @staticmethod
    def get_transform(train):
        if train:
            return transforms.Compose([
                transforms.RandomResizedCrop(64, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.4467, 0.4398, 0.4066], [0.2603, 0.2565, 0.2712])
            ])
        else:
            return transforms.Compose([
                transforms.Resize(64),
                transforms.ToTensor(),
                transforms.Normalize([0.4467, 0.4398, 0.4066], [0.2603, 0.2565, 0.2712]),
            ])

    def __getitem__(self, index):
        img, target = self.data[index], int(self.labels[index])
        img = Image.fromarray(np.transpose(img, (1,2,0)))

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)
            if self.triplet:
                pos_3 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.triplet:
            return (pos_1, pos_2, pos_3), target
        else:
            return (pos_1, pos_2), target

class TinyImageNet(ImageFolder):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        triplet: bool = False,
    ):
        super().__init__(root, transform)
        self.triplet = triplet
    
    @staticmethod
    def get_transform(train):
        if train:
            return transforms.Compose([
                transforms.RandomResizedCrop(64, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ])

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.transform is not None:
            pos_1 = self.transform(sample)
            pos_2 = self.transform(sample)
            if self.triplet:
                pos_3 = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.triplet:
            return (pos_1, pos_2, pos_3), target
        else:
            return (pos_1, pos_2), target