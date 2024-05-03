import os
import pdb
from os.path import join
import random
from collections import defaultdict

from PIL import ImageFilter
import torch
from torchvision import datasets, transforms


# Extended version of ImageFolder to return index of image too.
class ImageFolderEx(datasets.ImageFolder):
    def __init__(self, root, *args, **kwargs):
        super(ImageFolderEx, self).__init__(root, *args, **kwargs)

    def __kshot__(self, k, seed):
        """Convert dataset to contain k-samples per class. Randomly\
        subsample images per class to obtain the k samples."""
        data_dict = defaultdict(list)
        k_samples = []
        targets = []
        for sample in self.samples:
            data_dict[sample[1]].append(sample[0])
        for i in range(len(self.classes)):
            # Select k-samples per class
            gen = torch.random.manual_seed(seed=seed)
            chosen = torch.randperm(len(data_dict[i]), generator=gen)[:k]
            # chosen = torch.randint(0, len(self.classes), (k, ), generator=gen)
            for j in chosen:
                k_samples.append([data_dict[i][j], i])
                targets.append(i)
        self.samples = k_samples
        # print(k_samples[:10])
        self.imgs = k_samples
        self.targets = targets

    def __getitem__(self, index):
        sample, target = super(ImageFolderEx, self).__getitem__(index)
        # return index, sample, target
        return sample, target


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def load_dataset(cfg):
    traindir = os.path.join(cfg.train_data_path, 'train')
    valdir = os.path.join(cfg.val_data_path, 'val')
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)

    augmentation_train = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    augmentation_val = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]

    train_dataset = ImageFolderEx(traindir, transforms.Compose(augmentation_train))
    if cfg.kshot > 0:
        train_dataset.__kshot__(cfg.kshot, cfg.kshot_seed)
        # print(train_dataset.samples)
    print(train_dataset)
    val_dataset = ImageFolderEx(valdir, transforms.Compose(augmentation_val))
    test_dataset = ImageFolderEx(valdir, transforms.Compose(augmentation_val))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=256, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )
    return train_loader, val_loader, val_loader
