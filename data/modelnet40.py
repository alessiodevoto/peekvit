import glob
import os
from typing import List, Optional

import h5py
import numpy as np
#import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import requests
import pathlib
import zipfile

############################################################### TRANSFORMS ###############################################################

def random_point_dropout(pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    # for b in range(batch_pc.shape[0]):
    dropout_ratio = np.random.random()*max_dropout_ratio # 0~0.875    
    drop_idx = np.where(np.random.random((pc.shape[0]))<=dropout_ratio)[0]
    # print ('use random drop', len(drop_idx))

    if len(drop_idx)>0:
        pc[drop_idx,:] = pc[0,:] # set to the first point
    return pc

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

############################################################### CUSTOM DATASET CLASS ###############################################################

class ModelNet40Ply2048(Dataset):
    def __init__(
        self,
        root,
        split="train",
    ):
        assert split == "train" or split == "test"

        self.split = split
        self.data_list = []
        self.labels_list = []
        for h5_name in glob.glob(os.path.join(root, "ply_data_%s*.h5" % split)):
            with h5py.File(h5_name, "r") as f:
                self.data_list.append(f["data"][:].astype(np.float32))  # type: ignore
                self.labels_list.append(f["label"][:].astype(np.int64))  # type: ignore
        self.data = np.concatenate(self.data_list, axis=0)
        self.labels = np.concatenate(self.labels_list, axis=0).squeeze(-1)

    def __getitem__(self, item):
        points = self.data[item]
        label = self.labels[item]
        if self.split == 'train':
            points = random_point_dropout(points) # open for dgcnn not for our idea  for all
            points = translate_pointcloud(points)
            np.random.shuffle(points)
        return points, label

    def __len__(self):
        return self.data.shape[0]

############################################################### MODELNET40 DATASET ###############################################################

class ModelNet40:
    """
    A class representing the ModelNet40 dataset.

    Args:
        root (str): The root directory of the dataset.
        train_transform (callable, optional): A function/transform that takes in an image and returns a transformed version for training. Default is None.
        test_transform (callable, optional): A function/transform that takes in an image and returns a transformed version for testing. Default is None.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it. Default is None.

    Attributes:
        root (str): The root directory of the dataset.
        train_transform (callable, optional): A function/transform that takes in an image and returns a transformed version for training.
        test_transform (callable, optional): A function/transform that takes in an image and returns a transformed version for testing.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        train_dataset (Dataset): The training dataset.
        val_dataset (Dataset): The validation dataset.
    """

    MODELNET40_URL = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'

    MODELNET40_CLASSES = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']

    def __init__(self, root, num_points = 1024, num_classes = 40, train_transform=None, test_transform=None, target_transform=None, augmentation_ops=2, augmentation_magnitude=9, **kwargs):
        self.root = root
        self.num_points = num_points
        self.num_classes = num_classes
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.target_transform = target_transform
        self.augmentation_ops = augmentation_ops
        self.augmentation_magnitude = augmentation_magnitude
        self.train_dataset, self.val_dataset, self.train_transform, self.test_transform = self.get_modelnet40(root, train_transform, test_transform, target_transform)

        if 'num_classes' in kwargs:
            print(f'Warning: num_classes is not used for {self.__class__.__name__} dataset. Ignoring the argument and using default number of classes.')

    def get_modelnet40_transforms(self):
        """
        Returns the default train and test transforms for the ModelNet40 dataset.

        Args:
            image_size (int, optional): The size of the images. Default is 224.

        Returns:
            tuple: A tuple containing (train transform, test transform).
        
        """
        return (None, None) # TODO

    def get_modelnet40(self, root, train_transform=None, test_transform=None, target_transform=None):
        """
        Retrieves the ModelNet40 dataset from the specified root directory.
        If transforms are not specified, the default transforms are used.

        Args:
            root (str): The root directory to store the dataset.
            train_transform (callable, optional): A function/transform that takes in a training sample and returns a transformed version. Default is None.
            test_transform (callable, optional): A function/transform that takes in a test sample and returns a transformed version. Default is None.
            target_transform (callable, optional): A function/transform that takes in the target and transforms it. Default is None.
        
        Returns:
            tuple: A tuple containing (train, validation, default train transform, default test transform).
        """

        # get default transforms if not specified
        _train_transform, _test_transform = self.get_modelnet40_transforms()
        train_transform = train_transform or _train_transform
        test_transform = test_transform or _test_transform

        #create directory to store the dataset
        os.makedirs(root, exist_ok=True)

        #dowlnload dataset to the directory
        downloaded_file = pathlib.Path(root) / "modelnet40.zip"
        if not os.path.exists(downloaded_file):
            print(f'Downloading ModelNet40 dataset to {downloaded_file}')
            downloaded_file.write_bytes(requests.get(self.MODELNET40_URL, verify=False).content)
        else:
            print(f'Archive found at {downloaded_file}, skipping download')

        # unzip the downloaded file
        extracted_file = pathlib.Path(root) / "modelnet40"
        if not os.path.exists(extracted_file):
            print(f'Extracting ModelNet40 dataset to {extracted_file}')
            with zipfile.ZipFile(downloaded_file, 'r') as zip_ref:
                zip_ref.extractall(root)
        else:
            print(f'Archive already extracted to {extracted_file}, skipping extraction')
        
        #create ModelNet40 dataset
        ds_path = pathlib.Path(root) / "modelnet40_ply_hdf5_2048"
        #train_path, val_path = [pathlib.Path(root) / folder for folder in ['modelnet40/train', 'modelnet40/test']]
        train_dataset = ModelNet40Ply2048(ds_path, split="train")
        val_dataset = ModelNet40Ply2048(ds_path, split="test")

        return train_dataset, val_dataset, train_transform, test_transform



################################################################ OLD CODE ################################################################

"""
def random_point_dropout(pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    # for b in range(batch_pc.shape[0]):
    dropout_ratio = np.random.random()*max_dropout_ratio # 0~0.875    
    drop_idx = np.where(np.random.random((pc.shape[0]))<=dropout_ratio)[0]
    # print ('use random drop', len(drop_idx))

    if len(drop_idx)>0:
        pc[drop_idx,:] = pc[0,:] # set to the first point
    return pc

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

class ModelNet40Ply2048(Dataset):
    def __init__(
        self,
        root,
        split="train",
    ):
        assert split == "train" or split == "test"

        self.split = split
        self.data_list = []
        self.labels_list = []
        for h5_name in glob.glob(os.path.join(root, "ply_data_%s*.h5" % split)):
            with h5py.File(h5_name, "r") as f:
                self.data_list.append(f["data"][:].astype(np.float32))  # type: ignore
                self.labels_list.append(f["label"][:].astype(np.int64))  # type: ignore
        self.data = np.concatenate(self.data_list, axis=0)
        self.labels = np.concatenate(self.labels_list, axis=0).squeeze(-1)

    def __getitem__(self, item):
        points = self.data[item]
        label = self.labels[item]
        if self.split == 'train':
            points = random_point_dropout(points) # open for dgcnn not for our idea  for all
            points = translate_pointcloud(points)
            np.random.shuffle(points)
        return points, label

    def __len__(self):
        return self.data.shape[0]


class ModelNet40Ply2048DataModule(pl.LightningDataModule):
    
#    size: 12308
#    train: 9840
#    test: 2468
    

    def __init__(
        self,
        data_dir: str = "/home/alebai/projects/Datasets/PointClouds/modelnet40_ply_hdf5_2048",
        batch_size: int = 32,
        drop_last: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        with open(os.path.join(data_dir, "shape_names.txt"), "r") as f:
            self._label_names = [line.rstrip() for line in f]

    def setup(self, stage: Optional[str] = None):
        self.modelnet_train = ModelNet40Ply2048(self.hparams.data_dir, split="train")  # type: ignore
        self.modelnet_test = ModelNet40Ply2048(self.hparams.data_dir, split="test")  # type: ignore

    def train_dataloader(self):
        return DataLoader(
            self.modelnet_train,
            batch_size=self.hparams.batch_size,  # type: ignore
            shuffle=True,
            drop_last=self.hparams.drop_last,  # type: ignore
            num_workers=4,
        )

    def val_dataloader(self):
        return DataLoader(
            self.modelnet_test,
            batch_size=self.hparams.batch_size,  # type: ignore
            num_workers=4,
        )

    @property
    def num_classes(self):
        return 40

    @property
    def label_names(self) -> List[str]:
        return self._label_names
    
    @property
    def num_points(self):
        return 2048

"""