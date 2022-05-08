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

        
class ParrotDataSet(Dataset):
    """ Датасет Cockatiel VS Cockatoo
    
    Параметры
    ---------
    parrot_df : pandas dataframe
      Общий датафрейм с данными (обучение + тест).
      Датафрейм должен обязательно содержать следующие колонки:
      - filename (название файла изображения)
      - target (целевая метка в виде числа)
      - parrot_name (порода попугая)
      - split (используется для разделения данных по выборкам,
               принимает значения train или val)
    photo_dir : str
      Путь к папке с изображениями.
    base_transforms : BaseImageTransform
      Базовый набор преобразований изображения
    train_transforms : BaseImageTransform
      Набор преобразований изображения для обучения
    seed : int
      Фиксация генератора случайных чисел.
    """

    def __init__(self, parrot_df, base_transforms, train_transforms=None, seed=None):

        if seed is not None:
            set_seed(seed)

        assert 'filename' in parrot_df.columns, "Датафрейм parrot_df должен содержать колонку filename"
        assert 'target' in parrot_df.columns, "Датафрейм parrot_df должен содержать колонку target"
        assert 'split' in parrot_df.columns, "Датафрейм parrot_df должен содержать колонку split"
        assert 'parrot_name' in parrot_df.columns, "Датафрейм parrot_df должен содержать колонку parrot_name"

        self._parrot_df = parrot_df

        self._train_df = parrot_df[parrot_df.split == 'train'].copy()
        self._train_size = len(self._train_df)
        self._val_df = parrot_df[parrot_df.split == 'val'].copy()
        self._val_size = len(self._val_df)
        
        self._train_transforms = train_transforms
        self._base_transform = base_transforms
        
        self._lookup_dict = {
            'train': (self._train_df, self._train_size, train_transforms),
            'val': (self._val_df, self._val_size, base_transforms),
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
        self._target_df, self._target_size, self._target_transforms = self._lookup_dict[split]

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
        row = self._target_df.iloc[index, :]

        # Словарь, который будем возвращать
        model_data = {}

        # Формируем ссылку к изображению и проверяем на доступность файла
        image_path = row.filename
        if not os.path.exists(image_path):
            raise Exception(f"Файл {image_path} не существует")

        # Читаем изображение и применяем функцию обработки изображения
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self._target_transforms is not None:
            transform_img = self._target_transforms(image=img)
        else:
            transform_img = self._base_transform(img)

        # Добавляем в возвращаемый словарь исходное изображение с базовым преобразованием
        model_data['img_source'] = self._base_transform(img)
        # Добавляем в возвращаемый словарь аугментированное изображение
        model_data['img_transform'] = transform_img
        # Добавляем в возвращаемый словарь путь к исходному изображению
        model_data['image_path'] = image_path
        # Добавляем в возвращаемый словарь целевую метку
        model_data['target'] = row.target
        
        return model_data
        
class ParrotDataModule(pl.LightningDataModule):
    """ Загрузчик PyTorch Lighting для датасета ParrotDataSet
    
    Параметры
    ---------
    parrot_df : pandas dataframe
      Общий датафрейм с данными (обучение + тест).
      Датафрейм должен обязательно содержать следующие колонки:
      - filename (полный пусть к файлу изображения)
      - target (целевая метка в виде числа)
      - parrot_name (порода попугая)
      - split (используется для разделения данных по выборкам,
               принимает значения train или val)
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
    dataset_params : dict
      Словарь для создания экземпляров ParrotDataSet
      Предназначен для передачи разных типов аугментаций.
      Должен содержать все следующие ключи:
        - base_transforms
        - train_transforms
        - val_transforms
    seed : int
      Фиксация генератора случайных чисел.
    """

    def __init__(self, parrot_df, dataset_params, train_loader_params=None, val_loader_params=None, seed=None):

        super().__init__()

        if seed is not None:
            set_seed(seed)
            
        assert set(dataset_params.keys()) == set(['base_transforms', 'train_transforms'])

        self._parrot_df = parrot_df

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

        self.train_dataset = ParrotDataSet(parrot_df=self._parrot_df, **self._dataset_params)
        self.train_dataset.set_split('train')

        self.val_dataset = ParrotDataSet(parrot_df=self._parrot_df, **self._dataset_params)
        self.val_dataset.set_split('val')


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_loader_params['batch_size'],
                          shuffle=self.train_loader_params['shuffle'], drop_last=self.train_loader_params['drop_last'],
                          num_workers=self.train_loader_params['num_workers'])

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_loader_params['batch_size'],
                          drop_last=self.val_loader_params['drop_last'], shuffle=self.val_loader_params['shuffle'],
                          num_workers=self.val_loader_params['num_workers'])
    

    @classmethod
    def create_datamodule(cls, parrot_xlsx, photo_dir, train_loader_params=None, val_loader_params=None, 
                          dataset_params=None, seed=None):   
        """
        Создание экземпляра класса загрузчика ParrotDataLoader
        
        Изначально в файле разметки нет полного пути к фото, функция дополняет ипя файла полным путем к нему 
        и возвращает инициализированный класс ParrotDataLoader
        
        Параметры
        ---------
        parrot_xlsx : str
          Пусть к файлу train.csv
        photo_dir : str
          Путь к папке с фото для обучения
        train_loader_params : dict
          Словарь для параметров DataLoader обучающего датасета. 
          См. код инициализации класса.
        val_loader_params : dict
          Словарь для параметров DataLoader валидационного датасета. 
          См. код инициализации класса.
        dataset_params : dict
          Словарь для создания экземпляров ParrotDataSet
          См. код инициализации класса.
        seed : int
          Фиксация генератора случайных чисел.
        Результат
        ---------
        parrotdataloader : ParrotDataLoader
        """     

        parrot_df = pd.read_excel(parrot_xlsx)
        
        if os.path.sep != photo_dir[-1]:
            photo_dir += os.path.sep
            
        parrot_df.filename = parrot_df.filename.apply(lambda x: photo_dir + x)
        
        return cls(parrot_df=parrot_df, train_loader_params=train_loader_params, val_loader_params=val_loader_params,
                   dataset_params=dataset_params, seed=seed)