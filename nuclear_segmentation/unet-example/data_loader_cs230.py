from argparse import Namespace
import torch
from tifffile import TiffFile
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from torchvision import transforms


data_dir = Path.home() / 'data/isbi2012/'
args = Namespace(
    path_img_train=data_dir / 'train-volume.tif',
    path_target_train=data_dir / 'train-labels.tif',
    path_img_test=data_dir / 'test-volume.tif',
    n_valid=5,
    TRAIN='train',
    VAL='val',
    TEST='test'
)


def get_transforms():
    train_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    val_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    return train_transforms, val_transforms


class ISBI2012Dataset(Dataset):

    def __init__(self,
                 path_img,
                 path_target=None,
                 transforms=None,
                 split=args.TRAIN,
                 n_valid=args.n_valid):

        # read .tif files into numpy arrays (30, 512, 512) - so called 'an image with scalar data'
        # and add a dimension (30, 512, 512, 1), where 30 is the number of training images
        self.split = split

        self.images = np.expand_dims(TiffFile(path_img).asarray(), axis=-1)
        self.length = len(self.images)

        self.targets = None
        if path_target:
            self.targets = np.expand_dims(TiffFile(path_target).asarray(), axis=-1)

        if split == args.TRAIN:
            self.images = self.images[:(self.length-n_valid)]
            self.targets = self.targets[:(self.length-n_valid)]
        elif split == args.VAL:
            self.images = self.images[(self.length-n_valid):]
            self.targets = self.targets[(self.length-n_valid):]
        elif split == args.TEST:
            pass
        else:
            raise IndexError('illegal type')

        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        target = np.zeros_like(img)
        if self.split != args.TEST:
            target = self.targets[idx]

        if self.transforms:
            img = self.transforms(img)
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
    train_transforms, val_transforms = get_transforms()
    dataloaders = {}

    for split in splits:

        if split == args.TRAIN:
            dl = DataLoader(ISBI2012Dataset(path_img=args.path_img_train,
                                            path_target=args.path_target_train,
                                            transforms=train_transforms,
                                            split=args.TRAIN),
                            batch_size=params.batch_size,
                            shuffle=True,
                            num_workers=0,
                            pin_memory=params.cuda)
        elif split == args.VAL:
            dl = DataLoader(ISBI2012Dataset(path_img=args.path_img_train,
                                            path_target=args.path_target_train,
                                            transforms=train_transforms,
                                            split=args.VAL),
                            batch_size=params.batch_size,
                            shuffle=True,
                            num_workers=0,
                            pin_memory=params.cuda)
        elif split == args.TEST:
            dl = DataLoader(ISBI2012Dataset(path_img=args.path_img_test,
                                            path_target=None,
                                            transforms=val_transforms,
                                            split=args.TEST),
                            batch_size=params.batch_size,
                            shuffle=False,
                            num_workers=0,
                            pin_memory=params.cuda)
        else:
            raise IndexError('incorrect split')

        dataloaders[split] = dl

    return dataloaders
