import shutil
import torchvision
from torchvision import transforms as T
import os
import requests
import zipfile
import requests
import pathlib
import os
import re

######################################################### TINY IMAGENET DATASET #########################################################


class TinyImageNet:
  """
  A class representing the Imagenette dataset.

  Args:
    root (str): The root directory of the dataset.
    train_transform (callable, optional): A function/transform that takes in an image and returns a transformed version for training. Default is None.
    test_transform (callable, optional): A function/transform that takes in an image and returns a transformed version for testing. Default is None.
    target_transform (callable, optional): A function/transform that takes in the target and transforms it. Default is None.
    image_size (int, optional): The size of the images. Default is 160.

  Attributes:
    root (str): The root directory of the dataset.
    train_transform (callable, optional): A function/transform that takes in an image and returns a transformed version for training.
    test_transform (callable, optional): A function/transform that takes in an image and returns a transformed version for testing.
    target_transform (callable, optional): A function/transform that takes in the target and transforms it.
    image_size (int): The size of the images.
    train_dataset (Dataset): The training dataset.
    val_dataset (Dataset): The validation dataset.
  """

  TINY_IMAGENET_URL = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip' 


  TINY_IMAGENET_DENORMALIZE_TRANSFORM = T.Compose([
                                T.Normalize(mean=[0, 0, 0], std=[1/0.229, 1/0.224, 1/0.225]),
                                T.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1])])
                    
  IMAGENETTE_CLASSES = None #TODO

  def __init__(self, root, train_transform=None, test_transform=None, target_transform=None, image_size: int = 224, **kwargs):
    self.root = root
    self.train_transform = train_transform
    self.test_transform = test_transform
    self.target_transform = target_transform
    self.image_size = image_size
    self.denormalize_transform = self.TINY_IMAGENET_DENORMALIZE_TRANSFORM
    self.train_dataset, self.val_dataset, self.train_transform, self.test_transform = self.get_tiny_imagenet(root, train_transform, test_transform, target_transform, image_size)
    
    if 'num_classes' in kwargs:
      print(f'Warning: num_classes is not used for {self.__class__.__name__} dataset')

  @staticmethod
  def normalize_tin_val_folder_structure(path,
                                        images_folder='images',
                                        annotations_file='val_annotations.txt'):
      # From https://gist.github.com/lromor/bcfc69dcf31b2f3244358aea10b7a11b
      # Check if files/annotations are still there to see
      # if we already run reorganize the folder structure.
      images_folder = os.path.join(path, images_folder)
      annotations_file = os.path.join(path, annotations_file)

      # Exists
      if not os.path.exists(images_folder) \
        and not os.path.exists(annotations_file):
          if not os.listdir(path):
              raise RuntimeError('Validation folder is empty.')
          return

      # Parse the annotations
      with open(annotations_file) as f:
          for line in f:
              values = line.split()
              img = values[0]
              label = values[1]
              img_file = os.path.join(images_folder, values[0])
              label_folder = os.path.join(path, label)
              os.makedirs(label_folder, exist_ok=True)
              try:
                  shutil.move(img_file, os.path.join(label_folder, img))
              except FileNotFoundError:
                  continue

      os.sync()
      assert not os.listdir(images_folder)
      shutil.rmtree(images_folder)
      os.remove(annotations_file)
      os.sync()


  def get_imagenet_transforms(self, image_size: int = 224):
      """
      Returns the default train and test transforms for the TinyImagenet dataset.
      
      Args:
        image_size (int, optional): The size of the images. Default is 224.
      
      Returns:
        tuple: A tuple containing (train transform, test transform).

      """
      test_transform  =  T.Compose([
                          T.Resize(image_size),
                          T.CenterCrop(image_size),
                          T.ToTensor(),
                          T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

      train_transform =  T.Compose([
                          T.RandomResizedCrop(image_size),
                          T.RandomHorizontalFlip(),
                          T.ToTensor(),
                          T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
      
      return train_transform, test_transform
  


  def get_tiny_imagenet(self, root, train_transform=None, test_transform=None, target_transform=None, image_size: int = 160):
      """
      Retrieves the TinyImagenet dataset from the specified root directory.
      If transforms are not specified, the default transforms are used.
      
      Args:
        root (str): The root directory to store the dataset.
        train_transform (callable, optional): A function/transform that takes in a training sample and returns a transformed version. Default is None.
        test_transform (callable, optional): A function/transform that takes in a test sample and returns a transformed version. Default is None.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it. Default is None.
        image_size (int, optional): The size of the images. Default is 160. 
      
      Returns:
        tuple: A tuple containing (train, validation, default train transform, default test transform).

      """

      # get default transforms if not specified
      _train_transform, _test_transform = self.get_imagenet_transforms(image_size=image_size)
      train_transform = train_transform or _train_transform
      test_transform = test_transform or _test_transform

      # create directory to store the dataset
      os.makedirs(root, exist_ok=True)

      # download the dataset to the directory
      downloaded_file = pathlib.Path(root) / "tiny_imagenet.zip"
      if not os.path.exists(downloaded_file):
        print(f'Downloading Tiny Imagenet dataset to {downloaded_file}')
        downloaded_file.write_bytes(requests.get(self.TINY_IMAGENET_URL).content)
      else:
        print(f'Archive found at {downloaded_file}, skipping download')

      # unzip the downloaded file
      extracted_file = pathlib.Path(root) / 'tiny-imagenet-200'
      if not os.path.exists(extracted_file):
        print('Extracting  archive')
        with zipfile.ZipFile(downloaded_file, 'r') as zip_ref:
          zip_ref.extractall(root)
      else:
        print(f'Extracted file found at {extracted_file}, skipping extraction')
      
      # create the Imagenette dataset
      train_path, val_path = [pathlib.Path(root) / folder for folder in ['tiny-imagenet-200/train/', 'tiny-imagenet-200/val/']]
      
      print('Reorganizing validation folder structure. Might take a while...')
      self.normalize_tin_val_folder_structure(val_path)
      train_dataset = torchvision.datasets.ImageFolder(train_path, train_transform, target_transform)
      val_dataset = torchvision.datasets.ImageFolder(val_path, test_transform, target_transform)

      return train_dataset, val_dataset, train_transform, test_transform
