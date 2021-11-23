import os
from typing import Union, Optional, Tuple

import torch.utils.data as data
from torchvision import transforms
import numpy as np
import torch
from torch import Tensor
from pandas import Series


class AttributeDataset(data.Dataset):
    """Torch dataset for attributes

    Parameters
    ----------
    castors : Pandas Series, article identifier.
    labels : Pandas Series, attribute labels.
    inference : bool, inference mode.
    multilabel : bool, flag for multi label attributes.
    n_classes : int, number of classes in attribute.
    transform : torchvision Compose, image transformations, optional.
    img_path : str, path to stored images.
    """

    def __init__(
        self,
        castors: Series,
        labels: Optional[Series] = None,
        inference: bool = False,
        multilabel: bool = False,
        n_classes: Optional[int] = None,
        transform: Optional[transforms.Compose] = None,
        root_path: str = "/dbfs/mnt/pim",
    ):
        self.castors = castors
        self.labels = labels
        self.multilabel = multilabel
        self.inference = inference
        self.n_classes = n_classes
        self.transform = transform
        self.img_path = f"{root_path}/images/"

        print("reading images for training")
        self.img_list = []
        for _, castor in self.castors.items():
            img_path = os.path.join(self.img_path, f"{castor}.npy")
            img = np.load(img_path)
            self.img_list.append(img)
        if not inference and labels is None:
            raise ValueError("Labels can only be 'None' for inference mode.")

    def __len__(self) -> int:
        length: int = self.castors.shape[0]
        return length

    def __getitem__(self, index: Union[int, torch.Tensor]) -> Tuple[Tensor, Tensor]:

        if self.labels is None:
            raise AttributeError("labels is None, please check the parameter value.")
        if self.n_classes is None:
            raise AttributeError("n_classes is None, please check the parameter value.")

        if isinstance(index, torch.Tensor):
            index = int(index.item())
        #img_path = os.path.join(self.img_path, f"{self.castors.iat[idx]}.npy")

        #img = np.load(img_path)
        if self.transform:
            img = self.transform(self.img_list[index])

        if self.multilabel:
            label = torch.zeros(self.n_classes)
            if not self.inference:
                label[self.labels.iat[index]] = 1
        else:
            if self.inference:
                label = torch.tensor(0)
            else:
                label = torch.tensor(self.labels.iat[index])

        return img, label


