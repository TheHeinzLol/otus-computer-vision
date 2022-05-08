import os.path
import random

from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from torchvision.datasets import MNIST, CIFAR10
import torchvision.transforms as T

def set_seed(seed=None, seed_torch=True):
    """ Фиксируем генератор случайных чисел
    
    Параметры
    ---------
    seed : int
    seed_torch : bool
      Если True, то будет зафиксирован PyTorch
    """
    
    if seed is None:
        seed = np.random.choice(2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
    if seed_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    
    
class MNISTDataSet(Dataset):
    """ Датасет MNIST
    
    Параметры
    ---------
    base_transforms : BaseImageTransform
      Базовый набор преобразований изображения
    train_transforms : BaseImageTransform
      Набор преобразований изображения для обучения
    seed : int
      Фиксация генератора случайных чисел.
    """

    def __init__(self, base_transforms, train_transforms=None, seed=None):

        if seed is not None:
            set_seed(seed)
        
        self._mnist_train_dataset = MNIST('mnist/train', train=True, download=True)
        self._train_size = len(self._mnist_train_dataset)
        self._mnist_val_dataset = MNIST('mnist/val', train=False, download=True)        
        self._val_size = len(self._mnist_val_dataset)
        
        self._train_transforms = train_transforms
        self._base_transform = base_transforms
        
        self._lookup_dict = {
            'train': (self._mnist_train_dataset, self._train_size, train_transforms),
            'val': (self._mnist_val_dataset, self._val_size, base_transforms),
        }
        
        # По умолчанию включаем режим обучения
        self.set_split('train')

    def set_split(self, split='train'):
        """ Выбор режима датасета
        
        Параметры
        ---------
        split : str
          Выбор режима train или val
        """
        split = str(split).strip().lower()
        assert split in ('train', 'val'), "split может принимать значения train или val"

        self._target_split = split
        self._target_dataset, self._target_size, self._target_transforms = self._lookup_dict[split]

    def __len__(self):
        return self._target_size

    def __getitem__(self, index):
        """ Возвращает элемент датасета в формате:
        {
        'x_data': Тензор изображения размером N x C x H x W,
        'target': числовая метка
        }
        """

        # Получаем строку датафрейма по его индексу
        image, target = self._target_dataset[index]
        image = np.asarray(image)

        # Словарь, который будем возвращать
        model_data = {}
        
        if self._target_transforms is not None:
            transform_img = self._target_transforms(image=image)
        else:
            transform_img = self._base_transform(image)

        # Добавляем в возвращаемый словарь исходное изображение с базовым преобразованием
        model_data['img_source'] = self._base_transform(image)
        # Добавляем в возвращаемый словарь аугментированное изображение
        model_data['img_transform'] = transform_img
        # Добавляем в возвращаемый словарь путь к исходному изображению
        model_data['target'] = target
        
        return model_data
                   
                   
class MNISTDataModule(pl.LightningDataModule):
    """ Модуль изображений PyTorch Lighting для датасета MNIST
    
    Параметры
    ---------
    dataset_params : dict
      Словарь для создания экземпляров MNISTDataSet.
      Предназначен для передачи разных типов аугментаций.
      Содержит параметры инициализации класса MNISTDataSet:
        - base_transforms
        - train_transforms
        - val_transforms
        - seed
    train_loader_params : dict
      Словарь для параметров DataLoader обучающего датасета.
      Имеет следующие ключи:
        - batch_size (по умолчанию 16)
        - shuffle (по умолчанию True)
        - num_workers (по умолчанию 2)
        - drop_last (по умолчанию True)
    val_loader_params : dict
      Словарь для параметров DataLoader валидационного датасета.
      Имеет следующие ключи:
        - batch_size (по умолчанию 16)
        - shuffle (по умолчанию False)
        - num_workers (по умолчанию 2)
        - drop_last (по умолчанию False)
    seed : int
      Фиксация генератора случайных чисел.
    """

    def __init__(self, dataset_params, train_loader_params=None, val_loader_params=None, seed=None):

        super().__init__()
        
        assert set(dataset_params.keys()) == set(['base_transforms', 'train_transforms'])

        if seed is not None:
            set_seed(seed)
            
        if not train_loader_params:
            train_loader_params = {
                'batch_size': 16,
                'shuffle': True,
                'num_workers': 2,
                'drop_last': True,
            }

        if not val_loader_params:
            val_loader_params = {
                'batch_size': 16,
                'shuffle': False,
                'num_workers': 2,
                'drop_last': False
            }

        self.train_loader_params = train_loader_params
        self.val_loader_params = val_loader_params
        
        self._dataset_params = dataset_params
        
    def setup(self, stage=None):
        self._mnist_train_dataset = MNISTDataSet(**self._dataset_params)
        self._mnist_train_dataset.set_split('train')
        self._mnist_val_dataset = MNISTDataSet(**self._dataset_params)
        self._mnist_val_dataset.set_split('val')

    def train_dataloader(self):
        return DataLoader(self._mnist_train_dataset, batch_size=self.train_loader_params['batch_size'],
                          shuffle=self.train_loader_params['shuffle'], drop_last=self.train_loader_params['drop_last'],
                          num_workers=self.train_loader_params['num_workers'])

    def val_dataloader(self):
        return DataLoader(self._mnist_val_dataset, batch_size=self.val_loader_params['batch_size'],
                          drop_last=self.val_loader_params['drop_last'], shuffle=self.val_loader_params['shuffle'],
                          num_workers=self.val_loader_params['num_workers'])