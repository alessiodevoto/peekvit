import torch
import torchvision
import os
import requests
import tarfile
import requests
import pathlib


IMAGENETTE_URL = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz'


def get_imagenette(root, train_transform=None, test_transform=None, target_transform=None):
    """
    Retrieves the Imagenette dataset from the specified root directory.
    
    Args:
      root (str): The root directory to store the dataset.
      train_transform (callable, optional): A function/transform that takes in a training sample and returns a transformed version. Default is None.
      test_transform (callable, optional): A function/transform that takes in a test sample and returns a transformed version. Default is None.
      target_transform (callable, optional): A function/transform that takes in the target and transforms it. Default is None.
    
    Returns:
      tuple: A tuple containing the train and validation datasets.
    """
    

    # create directory to store the dataset
    os.makedirs(root, exist_ok=True)

    # download the dataset to the directory
    downloaded_file = pathlib.Path(root) / "imagenette.zip"
    if not os.path.exists(downloaded_file):
      print(f'Downloading Imagenette dataset to {downloaded_file}')
      downloaded_file.write_bytes(requests.get(IMAGENETTE_URL).content)
    else:
      print(f'Archive found in {downloaded_file}, skipping download')

    # unzip the downloaded file
    extracted_file = pathlib.Path(root) / 'imagenette2-160'
    if not os.path.exists(extracted_file):
      print('Extracting archive')
      with tarfile.open(downloaded_file) as file:
          file.extractall(path=root)
    else:
      print(f'Extracted file found at {extracted_file}, skipping extraction')
    

    # create the Imagenette dataset
    train_path, val_path = [pathlib.Path(root) / folder for folder in ['imagenette2-160/train', 'imagenette2-160/val']]
    train_dataset = torchvision.datasets.ImageFolder(train_path, train_transform, target_transform)
    val_dataset = torchvision.datasets.ImageFolder(val_path, test_transform, target_transform)

    return train_dataset, val_dataset