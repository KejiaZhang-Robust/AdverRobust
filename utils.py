import glob
import math
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import torchvision
from torchvision import datasets, models, transforms
from PIL import Image
from tqdm import tqdm, tqdm_notebook
import csv
import numpy as np
import os
import scipy.sparse as sp
import scipy.stats as st
import random
from PIL import Image
from torch.utils.data import Subset
from torch.utils.data import TensorDataset, DataLoader, random_split

from random import triangular
from typing import Callable, Union
import os
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Set, Tuple

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_txt(path :str) -> list:
    return [line.rstrip('\n') for line in open(path)]

corruptions = load_txt('./data/corruptions.txt')

class CIFAR10C(datasets.VisionDataset):
    def __init__(self, root :str, name :str, 
                 transform=None, target_transform=None):
        # corruptions = load_txt('./src/corruptions.txt')
        assert name in corruptions
        super(CIFAR10C, self).__init__(
            root, transform=transform,
            target_transform=target_transform
        )
        data_path = os.path.join(root, name + '.npy')
        target_path = os.path.join(root, 'labels.npy')
        
        self.data = np.load(data_path)
        self.targets = np.load(target_path)
        
    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            targets = self.target_transform(targets)
            
        return img, targets
    
    def __len__(self):
        return len(self.data)
    
class TinyImageNet(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        self.root_dir = root_dir
        self.mode = mode
        self.image_paths = sorted(glob.iglob(os.path.join(self.root_dir, self.mode, '**', '*.%s' % 'JPEG'), recursive=True))
        self.transform = transform

        self.labels = {}
        #label2dict
        with open(os.path.join(self.root_dir, 'wnids.txt'), 'r') as fp:
            self.label_texts = sorted([text.strip() for text in fp.readlines()])
        self.label_text_to_number = {text: i for i, text in enumerate(self.label_texts)}
        if self.mode == 'train':
            for label_text, i in self.label_text_to_number.items():
                for cnt in range(500):
                    self.labels['%s_%d.%s' % (label_text, cnt, 'JPEG')] = i
                    
        elif self.mode == 'val':
            with open(os.path.join(self.root_dir, 'val', 'val_annotations.txt'), 'r') as fp:
                for line in fp.readlines():
                    terms = line.split('\t')
                    file_name, label_text = terms[0], terms[1]
                    self.labels[file_name] = self.label_text_to_number[label_text]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        file_path = self.image_paths[idx]
        img = Image.open(file_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[os.path.basename(file_path)]
    
def shuffle_labels(label):
    max_val = torch.max(label).item()
    shuffled = torch.randint(0, max_val + 1, label.size()).to(device)
    shuffled[label == shuffled] = (shuffled[label == shuffled] + 1) % (max_val + 1)
    return shuffled

def extract_subset(dataset, num_subset :int, random_subset :bool):
    if random_subset:
        random.seed(0)
        indices = random.sample(list(range(len(dataset))), num_subset)
    else:
        indices = [i for i in range(num_subset)]
    return Subset(dataset, indices)

def record_path_words(record_path, record_words):
    print(record_words)
    with open(record_path, "a+") as f:
        f.write(record_words)
    f.close()
    return 

def format_time(seconds):
    """
    cur_time = time.time()
    time.sleep(64)
    last_time = time.time()
    step_time = format_time(last_time-cur_time)
    print(step_time)
    """
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    return f

def create_dataloader(dataset, Norm):
    if dataset == "TinyImageNet":
        if Norm == True:
            transform_train = transforms.Compose([
                    transforms.RandomCrop(64, padding=8),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        else:
            transform_train = transforms.Compose([
                    transforms.RandomCrop(64, padding=8),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor()
                ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
            ])
        train_dataset = TinyImageNet('./data/tiny-imagenet-200','train',transform=transform_train)
        testset = TinyImageNet('./data/tiny-imagenet-200','val',transform=transform_test)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)
        return train_loader, test_loader
    if dataset == "Imagenette":
        if Norm == True:
            transform_train = transforms.Compose([
                    transforms.RandomCrop(160),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        else:
            transform_train = transforms.Compose([
                    transforms.RandomCrop(160),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
            ])
        train_dataset = torchvision.datasets.ImageFolder(root='./data/imagenette2-160/train',transform=transform_train)
        testset = torchvision.datasets.ImageFolder(root='./data/imagenette2-160/val',transform=transform_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)
        return train_loader, test_loader
    if dataset == "CIFAR10":
        if Norm == True:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
            ])
        else:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
            ])
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4)
        return train_loader, test_loader
    if dataset == "CIFAR100":
        if Norm == True:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])
        else:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
            ])
        train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4)
        return train_loader, test_loader

    # if dataset == "CIFAR10-C":
    #     if Norm == True:
    #         transform_test = transforms.Compose([
    #             transforms.ToTensor(),
    #             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
    #         ])
    #     else:
    #         transform_test = transforms.Compose([
    #             transforms.ToTensor(),
    #         ])
    #     corruptions = load_txt('./src/corruptions.txt')
    #     test_dataset = CIFAR10C(os.path.join('./data', 'CIFAR-10-C'),
    #                 corruptions, transform=transform_test
    #             )
    #     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4)
    #     return test_loader

def create_loader_with_val_CIFAR_10(val_size=2000):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load the datasets
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    # Split the training dataset into training and validation
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4)

    return train_loader, test_loader, val_loader

def create_loader_with_val_CIFAR_100(val_size=0):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load the datasets
    dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    # Split the training dataset into training and validation
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4)

    return train_loader, test_loader, val_loader