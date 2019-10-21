import tables
import scipy
import sys
import random
import numpy as np

from torchvision import transforms
from torch.utils.data import DataLoader
import PIL
from pathlib import Path


# this defines our dataset class which will be used by the dataloader
class Dataset(object):
    def __init__(self,
                 fname,
                 img_transform=None,
                 mask_transform=None,
                 edge_weight=0.):
        # nothing special here, just internalizing the constructor parameters
        self.fname = fname
        self.edge_weight = edge_weight

        self.img_transform = img_transform
        self.mask_transform = mask_transform

        self.tables = tables.open_file(self.fname)
        self.numpixels = self.tables.root.numpixels[:]
        self.nitems = self.tables.root.img.shape[0]
        self.tables.close()

        self.img = None
        self.mask = None

    def __getitem__(self, index):
        # opening should be done in __init__ but seems to be
        # an issue with multithreading so doing here
        with tables.open_file(self.fname, 'r') as db:
            self.img = db.root.img
            self.mask = db.root.mask

            # get the requested image and mask from the pytable
            img = self.img[index, :, :, :]
            mask = self.mask[index, :, :]

        # the original Unet paper assignes increased weights to the edges of the annotated objects
        # their method is more sophistocated, but this one is faster, we simply dilate the mask and
        # highlight all the pixels which were "added"
        if (self.edge_weight):
            weight = scipy.ndimage.morphology.binary_dilation(mask == 1, iterations=2) & ~mask
        else:  # otherwise the edge weight is all ones and thus has no affect
            weight = np.ones(mask.shape, dtype=mask.dtype)

        mask = mask[:, :, None].repeat(3, axis=2)  # in order to use the transformations given by torchvision
        weight = weight[:, :, None].repeat(3,
                                           axis=2)  # inputs need to be 3D, so here we convert from 1d to 3d by repetition

        img_new = img
        mask_new = mask
        weight_new = weight

        seed = random.randrange(sys.maxsize)  # get a random seed so that we can reproducibly do the transofrmations
        if self.img_transform is not None:
            random.seed(seed)  # apply this seed to img transforms
            img_new = self.img_transform(img)

        if self.mask_transform is not None:
            random.seed(seed)
            mask_new = self.mask_transform(mask)
            mask_new = np.asarray(mask_new)[:, :, 0].squeeze()

            random.seed(seed)
            weight_new = self.mask_transform(weight)
            weight_new = np.asarray(weight_new)[:, :, 0].squeeze()

        return img_new, mask_new, weight_new

    def __len__(self):
        return self.nitems


def get_transforms(patch_size=256):
    # note that since we need the transofrmations to be reproducible for both masks and images
    # we do the spatial transformations first, and afterwards do any color augmentations
    img_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=(patch_size, patch_size), pad_if_needed=True),
        # these need to be in a reproducible order, first affine transforms and then color
        transforms.RandomResizedCrop(size=patch_size),
        transforms.RandomRotation(180),
        transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=.5),
        transforms.RandomGrayscale(),
        transforms.ToTensor()
    ])

    mask_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=(patch_size, patch_size), pad_if_needed=True),
        # these need to be in a reproducible order, first affine transforms and then color
        transforms.RandomResizedCrop(size=patch_size, interpolation=PIL.Image.NEAREST),
        transforms.RandomRotation(180),
    ])

    return img_transform, mask_transform


def get_dataloader(phases=("train", "val"),
                   dataname="epistroma",
                   edge_weight=1.1,
                   batch_size=3):
    dataset = {}
    dataLoader = {}
    img_transform, mask_transform = get_transforms()
    data_dir_str = str(Path.home() / 'data/epi')

    # now for each of the phases, we're creating the dataloader
    for phase in phases:
        dataset[phase] = Dataset(f"{data_dir_str}/{dataname}_{phase}.pytable",
                                 img_transform=img_transform,
                                 mask_transform=mask_transform,
                                 edge_weight=edge_weight)

        dataLoader[phase] = DataLoader(dataset[phase],
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=8,
                                       pin_memory=True)
    return dataLoader
