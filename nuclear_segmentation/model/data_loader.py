from argparse import Namespace

from tifffile import TiffFile
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path

data_dir = Path.home() / 'data/isbi2012/'
args = Namespace(
    path_img_train=data_dir / 'train-volume.tif',
    path_target_test=data_dir / 'train-labels.tif',
    path_img_test=data_dir / 'test-volume.tif'
)


def get_transforms():
    train_transforms = None
    val_transforms = None

    return train_transforms, val_transforms


class ISBI2012Dataset(Dataset):

    def __init__(self, path_img, path_target=None, transforms=None):

        # read .tif files into numpy arrays (30, 512, 512) - so called 'an image with scalar data'
        # and add a dimension (30, 512, 512, 1), where 30 is the number of training images
        self.train = np.expand_dims(TiffFile(path_img).asarray(), axis=-1)
        self.targets = None

        if path_target:
            self.targets = np.expand_dims(TiffFile(path_target).asarray(), axis=-1)

        self.transforms = transforms

    def __len__(self):
        return len(self.train)

    def __getitem__(self, idx):
        img = self.train[idx]
        target = None
        if self.targets:
            target = self.targets[idx]

        if self.transforms:
            img = self.transforms(img)
            if self.targets:
                target = self.transforms(target)

        return img, target


def fetch_dataloader(types, params):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        params: (Params) hyperparameters

    Returns:
        dataloaders: (dict) contains the DataLoader object for each type in types
    """
    train_transforms, val_transforms = get_transforms()
    dataloaders = {}

    for split in types:

        if split == 'train':
            dl = DataLoader(ISBI2012Dataset(path_img=args.path_img_train,
                                            path_target=args.path_target_train,
                                            transforms=train_transforms),
                            batch_size=params.batch_size,
                            shuffle=True,
                            num_workers=0,
                            pin_memory=params.cuda)
        elif split == 'val':
            raise NotImplementedError
        elif split == 'test':
            dl = DataLoader(ISBI2012Dataset(path_img=args.path_img_test,
                                            transforms=val_transforms),
                            batch_size=params.batch_size,
                            shuffle=False,
                            num_workers=0,
                            pin_memory=params.cuda)
        else:
            raise IndexError('incorrect split')

        dataloaders[split] = dl

    return dataloaders
