from argparse import Namespace
import torch
from PIL import Image
from tifffile import TiffFile
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from torchvision import transforms
from torchvision.transforms import Compose
import PIL
import random
import sys

data_dir = Path.home() / 'data/isbi2012/pngs'
args = Namespace(
    path_img_train=data_dir / 'train/images',
    path_target_train=data_dir / 'train/masks',
    path_img_val=data_dir / 'val/images',
    path_target_val=data_dir / 'val/masks',
    path_img_test=data_dir / 'test/images',
    TRAIN='train',
    VAL='val',
    TEST='test'
)


def get_transforms_initial():
    train_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    val_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    return train_transforms, val_transforms


def get_transforms_stanford():
    """
    We started from the example of a project:
    http://cs230.stanford.edu/projects_winter_2019/reports/15810728.pdf
    code for this project is not available, but we have code for a close project:

    :return: train_transforms, val_transforms
    """
    train_transforms = transforms.Compose([
        transforms.RandomAffine(degrees=.2,
                                translate=(.05, .05),
                                shear=.05,
                                resample=PIL.Image.NEAREST),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    val_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    return train_transforms, val_transforms


class ISBI2012Dataset(Dataset):

    def __init__(self,
                 transforms=None,
                 split=args.TRAIN):

        self.split = split

        if split == args.TRAIN:
            self.images = [Image.open(f) for f in args.path_img_train.glob('*')]
            self.targets = [Image.open(f) for f in args.path_target_train.glob('*')]
        elif split == args.VAL:
            self.images = [Image.open(f) for f in args.path_img_val.glob('*')]
            self.targets = [Image.open(f) for f in args.path_target_val.glob('*')]
        elif split == args.TEST:
            self.images = [Image.open(f) for f in args.path_img_test.glob('*')]
        else:
            raise IndexError('illegal type')

        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.split != args.TEST:
            target = self.targets[idx]
        else:
            target = np.zeros_like(img)

        if self.transforms:
            # it's necessary to apply the same transforms to target!
            seed = random.randrange(sys.maxsize)

            random.seed(seed)
            torch.manual_seed(seed)
            img = self.transforms(img)

            random.seed(seed)
            torch.manual_seed(seed)
            target = self.transforms(target)

        return img, target


def fetch_dataloader(splits, params):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        splits: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        params: (Params) hyperparameters

    Returns:
        dataloaders: (dict) contains the DataLoader object for each type in types
    """

    # train_transforms, val_transforms = get_transforms_initial()
    train_transforms, val_transforms = get_transforms_stanford()

    dataloaders = {}

    for split in splits:

        if split == args.TRAIN:
            dl = DataLoader(ISBI2012Dataset(transforms=train_transforms,
                                            split=args.TRAIN),
                            batch_size=params.batch_size,
                            shuffle=True,
                            num_workers=0,
                            pin_memory=params.cuda)
        elif split == args.VAL:
            dl = DataLoader(ISBI2012Dataset(transforms=val_transforms,
                                            split=args.VAL),
                            batch_size=params.batch_size,
                            shuffle=False,
                            num_workers=0,
                            pin_memory=params.cuda)
        elif split == args.TEST:
            dl = DataLoader(ISBI2012Dataset(transforms=val_transforms,
                                            split=args.TEST),
                            batch_size=params.batch_size,
                            shuffle=False,
                            num_workers=0,
                            pin_memory=params.cuda)
        else:
            raise IndexError('incorrect split')

        dataloaders[split] = dl

    return dataloaders
