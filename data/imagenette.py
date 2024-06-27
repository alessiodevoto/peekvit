import torchvision
from torchvision import transforms as T
import os
import requests
import tarfile
import requests
import pathlib

######################################################### IMAGENETTE DATASET #########################################################


class Imagenette:
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

  IMAGENETTE_URL = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz'


  IMAGENETTE_DENORMALIZE_TRANSFORM = T.Compose([
                                T.Normalize(mean=[0, 0, 0], std=[1/0.229, 1/0.224, 1/0.225]),
                                T.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1])])
                    
  IMAGENETTE_CLASSES = ['tench', 'English springer', 'cassette player', 'chain saw', 'church', 'French horn', 'garbage truck', 'gas pump', 'golf ball', 'parachute']


  def __init__(self, root, train_transform=None, test_transform=None, target_transform=None, image_size: int = 160, augmentation_ops=2, augmentation_magnitude=9, verbose:bool=True, **kwargs):
    self.root = root
    self.verbose = verbose
    self.train_transform = train_transform
    self.test_transform = test_transform
    self.target_transform = target_transform
    self.image_size = image_size
    self.augmentation_ops = augmentation_ops
    self.augmentation_magnitude = augmentation_magnitude
    self.denormalize_transform = self.IMAGENETTE_DENORMALIZE_TRANSFORM
    self.train_dataset, self.val_dataset, self.train_transform, self.test_transform = self.get_imagenette(root, train_transform, test_transform, target_transform)

    if 'num_classes' in kwargs:
       print(f'Warning: num_classes is not used for {self.__class__.__name__} dataset. \nIgnoring the argument and using default number of classes in this dataset (10).')


  def get_imagenette_transforms(self):
      """
      Returns the default train and test transforms for the Imagenette dataset.
      
      Args:
        image_size (int, optional): The size of the images. Default is 160.
      
      Returns:
        tuple: A tuple containing (train transform, test transform).

      """
      test_transform  =  T.Compose([
                          T.Resize((self.image_size, self.image_size)),
                          T.CenterCrop(self.image_size),
                          T.ToTensor(),
                          T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

      train_transform =  T.Compose([
                          T.RandAugment(num_ops=self.augmentation_ops, magnitude=self.augmentation_magnitude),
                          T.Resize((self.image_size, self.image_size)),
                          T.ToTensor(),
                          T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
      
      return train_transform, test_transform


  def get_imagenette(self, root, train_transform=None, test_transform=None, target_transform=None):
      """
      Retrieves the Imagenette dataset from the specified root directory.
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
      _train_transform, _test_transform = self.get_imagenette_transforms()
      train_transform = train_transform or _train_transform
      test_transform = test_transform or _test_transform

      # create directory to store the dataset
      os.makedirs(root, exist_ok=True)

      # download the dataset to the directory
      downloaded_file = pathlib.Path(root) / "imagenette.zip"
      if not os.path.exists(downloaded_file):
        if self.verbose: print(f'Downloading Imagenette dataset to {downloaded_file}')
        downloaded_file.write_bytes(requests.get(self.IMAGENETTE_URL).content)
      else:
        if self.verbose: print(f'Archive found at {downloaded_file}, skipping download')

      # unzip the downloaded file
      extracted_file = pathlib.Path(root) / 'imagenette2-160'
      if not os.path.exists(extracted_file):
        if self.verbose: print('Extracting archive')
        with tarfile.open(downloaded_file) as file:
            file.extractall(path=root)
      else:
        if self.verbose: print(f'Extracted file found at {extracted_file}, skipping extraction')
      

      # create the Imagenette dataset
      train_path, val_path = [pathlib.Path(root) / folder for folder in ['imagenette2-160/train', 'imagenette2-160/val']]
      train_dataset = torchvision.datasets.ImageFolder(train_path, train_transform, target_transform)
      val_dataset = torchvision.datasets.ImageFolder(val_path, test_transform, target_transform)

      return train_dataset, val_dataset, train_transform, test_transform



# map imagenette to imagenet classes via a transform
class ImagenetToImagenetteLabel(object):
    def __init__(self):
        super().__init__()
        self.mapping = {
            0: 0,       # tench
            1: 217,     # english springer
            2: 482,     # cassette player
            3: 491,     # chainsaw
            4: 497,     # church
            5: 566,     # french horn
            6: 569,     # garbage
            7: 571,     # gas
            8: 574,     # golf ball
            9: 701,     # parachute
        }

    def __call__(self, label):
        return self.mapping[label]