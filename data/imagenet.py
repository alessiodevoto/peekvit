from torchvision.datasets import ImageNet
from torchvision import transforms as T
from typing import Literal


class ImageNetDataset:

    def __init__(
        self,
        root,
        split: Literal["train", "val", "train+val"] = "val",
        train_transform=None,
        test_transform=None,
        target_transform=None,
        image_size: int = 224,
        augmentation_ops: int = 2,
        augmentation_magnitude: int = 9,
        **kwargs,
    ):

        # get default transforms if not specified
        self.image_size = image_size
        self.augmentation_ops = augmentation_ops
        self.augmentation_magnitude = augmentation_magnitude
        _train_transform, _test_transform = self.get_imagenet_transforms()
        train_transform = train_transform or _train_transform
        test_transform = test_transform or _test_transform

        if "num_classes" in kwargs:
            print(
                f"Warning: num_classes is not used for {self.__class__.__name__}. \nIgnoring the argument and using default number of classes in this dataset (1000)."
            )

        self.train_dataset, self.val_dataset = None, None
        if "train" in split:
            self.train_dataset = ImageNet(
                root,
                split="train",
                transform=train_transform,
                target_transform=target_transform,
            )
        if "val" in split:
            self.val_dataset = ImageNet(
                root,
                split="val",
                transform=test_transform,
                target_transform=target_transform,
            )

    def get_imagenet_transforms(self):
        """
        Returns the default train and test transforms for the Imagenette dataset.

        Args:
          image_size (int, optional): The size of the images. Default is 160.

        Returns:
          tuple: A tuple containing (train transform, test transform).

        """
        test_transform = T.Compose(
            [
                T.Resize((self.image_size, self.image_size)),
                T.CenterCrop(self.image_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        train_transform = T.Compose(
            [
                T.RandAugment(
                    num_ops=self.augmentation_ops, magnitude=self.augmentation_magnitude
                ),
                T.Resize((self.image_size, self.image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        return train_transform, test_transform
