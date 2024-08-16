import math
import os
from typing import Any, Callable, Optional, Tuple
from monai import data, transforms as med
from monai.data import load_decathlon_datalist
import PIL.Image as PImage
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
from torchvision.transforms import transforms
from torch.utils.data import Dataset
import torch
import numpy as np
import cv2
try:
    from torchvision.transforms import InterpolationMode
    interpolation = InterpolationMode.BICUBIC
except:
    import PIL
    interpolation = PIL.Image.BICUBIC
from monai.transforms.transform import LazyTransform, MapTransform, RandomizableTransform
import random


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f: img: PImage.Image = PImage.open(f).convert('RGB')
    return img


class ImageNetDataset(DatasetFolder):
    def __init__(
            self,
            imagenet_folder: str,
            train: bool,
            transform: Callable,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        imagenet_folder = os.path.join(imagenet_folder, 'train' if train else 'val')
        super(ImageNetDataset, self).__init__(
            imagenet_folder,
            loader=pil_loader,
            extensions=IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=None, is_valid_file=is_valid_file
        )
        
        self.samples = tuple(img for (img, label) in self.samples)
        self.targets = None # this is self-supervised learning so we don't need labels
    
    def __getitem__(self, index: int) -> Any:
        img_file_path = self.samples[index]
        return self.transform(self.loader(img_file_path))


def build_dataset_to_pretrain(dataset_path, input_size) -> Dataset:
    """
    You may need to modify this function to return your own dataset.
    Define a new class, a subclass of `Dataset`, to replace our ImageNetDataset.
    Use dataset_path to build your image file path list.
    Use input_size to create the transformation function for your images, can refer to the `trans_train` blow. 
    
    :param dataset_path: the folder of dataset
    :param input_size: the input size (image resolution)
    :return: the dataset used for pretraining
    """
    trans_train = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.67, 1.0), interpolation=interpolation),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])
    
    dataset_path = os.path.abspath(dataset_path)
    for postfix in ('train', 'val'):
        if dataset_path.endswith(postfix):
            dataset_path = dataset_path[:-len(postfix)]
    
    dataset_train = ImageNetDataset(imagenet_folder=dataset_path, transform=trans_train, train=True)
    print_transform(trans_train, '[pre-train]')
    return dataset_train


def build_meddataset_to_pretrain(dataset_path, input_size) -> Dataset:
    """
    You may need to modify this function to return your own dataset.
    Define a new class, a subclass of `Dataset`, to replace our ImageNetDataset.
    Use dataset_path to build your image file path list.
    Use input_size to create the transformation function for your images, can refer to the `trans_train` blow.

    :param dataset_path: the folder of dataset
    :param input_size: the input size (image resolution)
    :return: the dataset used for pretraining
    """
    trans_train = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.67, 1.0), interpolation=interpolation),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])

    dataset_path = os.path.abspath(dataset_path)


    dataset_train = MedicalDataSets(base_dir=dataset_path, transform=trans_train)
    print_transform(trans_train, '[pre-train]')
    return dataset_train



class MedicalDataSets(Dataset):
    def __init__(
        self,
        base_dir=None,
        transform=None,
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.sample_list = os.listdir(self._base_dir)
        self.transform = transform
        print("total {}".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        img = PImage.open(os.path.join(self._base_dir, case)).convert('RGB')
        aug = self.transform(img)
        return aug

def print_transform(transform, s):
    print(f'Transform {s} = ')
    for t in transform.transforms:
        print(t)
    print('---------------------------\n')


class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank : self.total_size : self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

class RandScaleCropdPlusScaleByMidDimSampled(MapTransform):
    def __init__(self, keys, mode='area', max_size=128,allow_missing_keys=False,num_samples=4,max_radio=0.8,min_radio=0.5):
        self.keys = keys
        self.mode = mode
        self.allow_missing_keys = allow_missing_keys
        self.max_size=max_size
        self.num_samples = num_samples
        self.max_radio=max_radio
        self.min_radio=min_radio

    def __call__(self, data):
        outputs = []
        for i in range(self.num_samples):
            random_number = round(random.uniform(self.min_radio, self.max_radio), 2)
            _data = dict(data)
            for key in self.keys:
                cropper= med.RandScaleCropd(keys=[key],roi_scale=random_number)
                _data[key] = cropper(_data)[key]
                ct_tensor = _data[key]
                sorted_numbers = sorted(ct_tensor.shape[1:])
                scale_factor = self.max_size / sorted_numbers[1]
                new_size = [int(d * scale_factor)
                            for d in ct_tensor.shape[1:]]

                resizer = med.Resized(keys=[key],
                                             spatial_size=new_size,
                                             mode=self.mode,
                                             allow_missing_keys=self.allow_missing_keys)
                _data[key] = resizer(_data)[key]

            outputs.append(_data)

        return outputs




def get_loader(data_dir, size):
    datalist_json = os.path.join(data_dir, "dataset.json")
    train_transform = med.Compose(
    [
        med.LoadImaged(keys=["image"], allow_missing_keys=True),
        med.AddChanneld(keys=["image"], allow_missing_keys=True),
        med.Orientationd(keys=["image"], axcodes="RAS", allow_missing_keys=True),
        med.Spacingd(keys=["image"], pixdim=(1.5, 1.5, 1.5), mode="bilinear", allow_missing_keys=True),
        med.ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        med.CropForegroundd(keys=["image"], source_key="image", allow_missing_keys=True),
        med.SpatialPadd(keys=["image"], spatial_size=(size, size, size), mode='constant'),
        med.RandCropByPosNegLabeld(
            spatial_size=(size, size, size),
            keys=["image"],
            label_key="image",
            pos=1,
            neg=0,
            num_samples=4,
        ),
        med.RandFlipd(keys=["image"],
                            prob=0.2,
                            spatial_axis=0),
        med.RandFlipd(keys=["image"],
                            prob=0.2,
                            spatial_axis=1),
        med.RandFlipd(keys=["image"],
                            prob=0.1,
                            spatial_axis=2),
        med.ToTensord(keys=["image"]),
    ])
    # val_transform = transforms.Compose(
    #     [
    #         transforms.LoadImaged(keys=["image", "label"]),
    #         transforms.AddChanneld(keys=["image", "label"]),
    #         transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
    #         transforms.Spacingd(
    #             keys=["image", "label"], pixdim=(1, 1, 1), mode=("bilinear", "nearest")
    #         ),
    #         transforms.ScaleIntensityRanged(
    #             keys=["image"], a_min=-175.0, a_max=250.0, b_min=0.0, b_max=1.0, clip=True
    #         ),
    #         transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
    #         transforms.ToTensord(keys=["image", "label"]),
    #     ]
    # )

    datalist = load_decathlon_datalist(datalist_json, True, "training", base_dir=data_dir)
    # train_ds = data.Dataset(data=datalist, transform=train_transform)
    # train_ds = data.CacheDataset(data=datalist, transform=train_transform)
    # train_ds = data.SmartCacheDataset(data=datalist, transform=train_transform, replace_rate=0.7, cache_num=256,  num_init_workers=4, num_replace_workers=4)
    train_ds= data.CacheNTransDataset(data=datalist, transform=train_transform, cache_n_trans=6, cache_dir="/fenghetang/3d/pretrain/MM/cache_dataset")
    return train_ds



