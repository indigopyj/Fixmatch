import numpy as np
from PIL import Image
import torchvision
import torch


def get_cifar10(root, n_labeled,
                 transform_train=None, transform_val=None,
                 download=True, mode="train"):

    base_dataset = torchvision.datasets.CIFAR10(root, train=True, download=download)
    train_labeled_idxs, train_unlabeled_idxs, val_idxs = train_val_split(base_dataset.targets, int(n_labeled/10), num_val=500)
    
    if mode =="train":
        train_labeled_dataset = CIFAR10_labeled(root, train_labeled_idxs, train=True, transform=transform_train)
        train_unlabeled_dataset = CIFAR10_unlabeled(root, train_unlabeled_idxs, train=True, transform=transform_train)
        val_dataset = CIFAR10_labeled(root, val_idxs, train=True, transform=transform_val, download=True)
        print (f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_idxs)} #Val: {len(val_idxs)}")
        return train_labeled_dataset, train_unlabeled_dataset, val_dataset
    elif mode=="test":
        test_dataset = CIFAR10_labeled(root, train=False, transform=transform_val, download=True)
        return test_dataset

def train_val_split(labels, n_labeled_per_class, num_val):
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    val_idxs = []

    for i in range(10):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:n_labeled_per_class])
        train_unlabeled_idxs.extend(idxs[n_labeled_per_class:-num_val])
        val_idxs.extend(idxs[-num_val:])
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    np.random.shuffle(val_idxs)
    
    return train_labeled_idxs, train_unlabeled_idxs, val_idxs

def gender_train_val_split(train_labels, val_labels, n_labeled_per_class):
    train_labels = np.array(train_labels)
    val_labels = np.array(val_labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    val_idxs = []

    for i in range(2):
        t_idxs = np.where(train_labels == i)[0]
        v_idxs = np.where(val_lables == i)[0]
        np.random.shuffle(t_idxs)
        np.random.shuffle(v_idxs)
        train_labeled_idxs.extend(t_idxs[:n_labeled_per_class])
        train_unlabeled_idxs.extend(t_idxs[n_labeled_per_class:])
        val_idxs.extend(v_idxs)
    
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    np.random.shuffle(val_idxs)

    return train_labeled_idxs, train_unlabeled_idxs, val_idxs


cifar10_mean = (0.4914, 0.4822, 0.4465) # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616) # equals np.std(train_set.train_data, axis=(0,1,2))/255
svhn_mean = (0.5071, 0.4867, 0.4408)
svhn_std = (0.2675, 0.2565, 0.2761)
def normalise(x, mean=cifar10_mean, std=cifar10_std):
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    
    x -= mean*255
    x *= 1.0/(255*std)
    return x

class CIFAR10_labeled(torchvision.datasets.CIFAR10):

    def __init__(self, root, indexs=None, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR10_labeled, self).__init__(root, train=train,
                 transform=transform, target_transform=target_transform,
                 download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR10_unlabeled(CIFAR10_labeled):

    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR10_unlabeled, self).__init__(root, indexs, train=train,
                 transform=transform, target_transform=target_transform,
                 download=download)
        self.targets = np.array([-1 for i in range(len(self.targets))])




class SVHN_labeled(torchvision.datasets.SVHN):
    def __init__(self, root, indexs=None, split="train",
                 transform=None, target_transform=None,
                 download=False):
        super(SVHN_labeled, self).__init__(root, split=split,
                 transform=transform, target_transform=target_transform,
                 download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.labels = np.array(self.labels)[indexs]

        self.data = transpose(self.data, source="NCHW", target="NHWC")
        self.data = transpose(normalise(self.data, mean=svhn_mean, std=svhn_std))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.labels[index])
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class SVHN_unlabeled(SVHN_labeled):

    def __init__(self, root, indexs, split="train", extra=False,
                 transform=None, target_transform=None,
                 download=False):
        super(SVHN_unlabeled, self).__init__(root, indexs, split=split,
                 transform=transform, target_transform=target_transform,
                 download=download)
        
        if(extra):
            extra_dataset = torchvision.datasets.SVHN(root, split="extra", download=True)
            self.data = np.concatenate([self.data, extra_dataset.data], axis=0)
            np.random.shuffle(self.data)
            total_length = len(np.array(self.labels)) + len(np.array(extra_dataset.labels))
            self.labels = np.array([-1 for i in range(total_length)])
        else:
            self.labels = np.array([-1 for i in range(len(self.labels))])


class Gender_labeled(torochvision.datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, loader=default_loader, is_valid_file=None):
        super(Gender_labeled, self).__init__(root, transform=transform,
                target_transform=target_transform)

        if indexs is not None:
            self.samples = self.samples[indexs]
            self.data = np.array([self.loader(s[0]) for s in samples])
            self.targets = np.array([self.loader(s[1]) for s in samples])

        self.data = transpose(normalise(self.data)) # convert to CHW

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class TransformFix(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=32, padding=int(32*0.125), padding_mode='reflect')
            ])
        self.strong = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=32, padding=int(32*0.125), padding_mode='reflect')
                RandAugmentMC(n=2, m=10)
            ])
        self.normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
    
    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)
