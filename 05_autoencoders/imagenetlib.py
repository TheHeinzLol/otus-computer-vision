from pathlib import Path
import json
import os.path
import random

from tqdm.notebook import tqdm
import requests
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


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

class ImageTransform(nn.Module):
    """ Базовый класс для трансформации изображений на основе Albumentations
    """
    def __init__(self):
        
        super().__init__()
        self._transforms = A.Compose([])
        
    def forward(self, image):
        return self._transforms(image=image)['image'].type(torch.float32)
    
    def add_transform(self, transform, index=0):
        """ Добавить преобразование
        
        Параметры
        ---------
        transform : albumentations transform
          Инициализированный класс преобразования albumentations
        index : int
          Позиция для вставки в текущие преобразования.
          По умолчанию 0.
        """
        assert isinstance(index, int)
        new_transforms = []
        for idx, c_transform in enumerate(self._transforms):
            if index == idx:
                new_transforms.append(transform)
            new_transforms.append(c_transform)
        self._transforms = A.Compose(new_transforms)
        
    def delete_transform(self, index):
        """ Удалить преобразование
        
        Параметры
        ---------
        index : int или list
          Индекс или список индексов аугментаций для удаления
        """
            
        new_transforms = []
        for idx, c_transform in enumerate(self._transforms):
            if idx == index or idx in index:
                continue
            new_transforms.append(c_transform)
        self._transforms = A.Compose(new_transforms)
        
    def __repr__(self):
        return(str(self._transforms))
    
    def __str__(self):
        return(str(self._transforms))
      
      
class BaseImageTransform(ImageTransform):
    """ Класс с минимальными преобразованиями, которые можно использовать
    как для обучения, так и для валидации.
    
    - Изменение размера изображения
    - Нормализация изображения с параметрами от ImageNet
    - Преобразование изображения в тензор PyTorch
    - Преобразование значений изображения в тип torch.float32
    
    Параметры
    ---------
    height : int
      Высота изображения для операции Resize
    width : int
      Ширина изображения для операции Resize
    """
    def __init__(self, height=224, width=224, mean=None, std=None):
        
        super().__init__()
        
        if mean is None:
            # Средние значения для IMAGE NET, используются при стандартизации изображений
            mean = [0.485, 0.456, 0.406]
        if std is None:
            # СКО для IMAGE NET, используются при стандартизации изображений
            std = [0.229, 0.224, 0.225]
        
        self._transforms = A.Compose([
            A.Resize(height, width),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

def get_imagenette_dataloader(imagenette_path, image_size=224, split='val', shuffle=False, batch_size=32,
                              std=(0.229, 0.224, 0.225), mean=(0.485, 0.456, 0.406), interpolation='bicubic'):
    """ Получение загрузчика изображений
    
    Параметры
    ---------
    imagenette_path : str
      Путь к папке с датасетом Imagenette
    image_size : int or (int, int)
      Высота и ширина выходного изображения.
      Если int, то ширина и высота будут одинаковыми.
      По умолчанию 224.
    split : str
      Режим работы датасета.
      Принимает значение train или val.
      По умолчанию val.
    shuffle : bool
      Если True, то датасет будет перемешан. По умолчанию False.
    std : tuple
      Коэффициенты стандартного отклонения для нормализации изображения.
      По умолчанию (0.229, 0.224, 0.225)
    mean : tuple
      Коэффициенты среднего значения для нормализации изображения
      По умолчанию (0.485, 0.456, 0.406)
    interpolation : str
      Интерполяция для изменения размера изображения.
      По умолчанию bicubic.
    
    Результат
    ---------
    dataloader : Dataloader
    """
                                                   
    dataset = ImageNetteDataset(imagenette_path=imagenette_path, image_size=image_size, std=std, mean=mean,
                                interpolation=interpolation)                                       
    dataset.set_split(split)
    
    return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)


def get_imagenet_category_names(type_='code', json_class_index_file=None):
    """
    Получаем словари категорий ImageNet
    
    Функция формирует 2 словаря:
    - index_to_name
    - name_to_index
    
    Параметры
    type_ : str
      Тип имени в словаре.
      Примает значение code или name.
    json_class_index_file : str
      Путь к локальному файлу imagenet_class_index.json
      в формате {index: {код, имя}}.
      Если None, то будет попытка найти файл imagenet_class_index.json
      в корневой директори запуска программы.
      Если файл не будет найден, то будет попытка загрузки с веб-ресурса
      https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json
      
    Результат
    ---------
    index_to_name, name_to_index : (dict, dict)
    """
    
    type_ = str(type_).lower().strip()
    assert type_ in ('code', 'name')
    
    if json_class_index_file is None:
        default_json_class_index_file = 'imagenet_class_index.json'
        if os.path.exists(default_json_class_index_file):
            json_class_index_file = default_json_class_index_file
    
    if json_class_index_file is None:
        url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
        json_class_index = requests.get(url, timeout=8).json()
    else:
        with open(json_class_index_file, 'r', encoding='utf-8') as f:
            json_class_index = json.load(f)
          
    if type_ == 'code':
        name_index = 0
    else:
        name_index = 1
        
    index_to_name = {int(idx):names_list[name_index] for idx, names_list in json_class_index.items()}
    name_to_index = {item:key for key, item in index_to_name.items()}
    
    return index_to_name, name_to_index


class ImagenetteDataset(Dataset):
    """ Датасет Imagenette
    
    Работает с датасетами imagenette из https://github.com/fastai/imagenette
    Класс тестировался на датасете https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz
    
    Параметры
    ---------
    imagenette_path : str
      Путь к папке с датасетом Imagenette
    image_size : int or (int, int)
      Высота и ширина выходного изображения.
      Если int, то ширина и высота будут одинаковыми.
    std : tuple
      Коэффициенты стандартного отклонения для нормализации изображения
    mean : tuple
      Коэффициенты среднего значения для нормализации изображения
    interpolation : str
      Интерполяция для изменения размера изображения
    """
    
    def __init__(self, imagenette_path, base_transforms, image_size=224, train_transforms=None,
                 std=(0.229, 0.224, 0.225), mean=(0.485, 0.456, 0.406), interpolation='bicubic'):
        
        self._imagenette_path = Path(imagenette_path)
        self._index_to_name, self._name_to_index = get_imagenet_category_names(type_='code')
        
        df_imagenette = self._make_imagenette_dataframe()
        
        self._train_df = df_imagenette[df_imagenette.split == 'train'].copy()
        self._train_size = len(self._train_df)
        self._val_df = df_imagenette[df_imagenette.split == 'val'].copy()
        self._val_size = len(self._val_df)
        
        if not hasattr(image_size, '__iter__'):
            image_size = [image_size, image_size]
            
        if interpolation == 'bicubic':
            interpolation = InterpolationMode.BICUBIC
        elif interpolation == 'bilinear':
            interpolation = InterpolationMode.BILINEAR
        elif interpolation == 'nearest':
            interpolation = InterpolationMode.NEAREST
        else:
            raise Exception(f"Неизвестный тип интерполяции '{interpolation}'")
            
        self._train_transforms = train_transforms
        self._base_transform = base_transforms
            
        self._lookup_dict = {
            'train': (self._train_df, self._train_size, train_transforms),
            'val': (self._val_df, self._val_size, self._base_transform),
        }

        # По умолчанию включаем режим обучения
        self.set_split('train')
          
    def _make_imagenette_dataframe(self):
        """ Готовим датафрейм из изображений в папке датасета
        
        Результат
        df_imagenette : pd.DataFrame
        """
        
        imagenette_train_path = self._imagenette_path / 'train'
        imagenette_val_path = self._imagenette_path / 'val'
        df_imagenette = pd.DataFrame()
        
        assert imagenette_train_path.exists(), "Отсутствует папка train в папке датасета imagenette"
        assert imagenette_val_path.exists(), "Отсутствует папка val в папке датасета imagenette"
        
        filepath = []
        split = []
        target = []
        
        for split_type in ('train', 'val'): 
            for class_path in (self._imagenette_path / split_type).iterdir():
                if not class_path.is_dir():
                    continue
                class_index = self._name_to_index[class_path.name]
                for photo_path in class_path.iterdir():
                    if not photo_path.is_file():
                        continue
                    filepath.append(str(photo_path.absolute()))
                    split.append(split_type)
                    target.append(class_index)
                    
        df_imagenette = pd.DataFrame({'filepath': filepath, 'split': split, 'target': target})
        
        return df_imagenette
    
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
        
        # Получаем строку датафрейма по его индексу
        row = self._target_df.iloc[index, :]

        # Словарь, который будем возвращать
        model_data = {}

        # Формируем ссылку к изображению и проверяем на доступность файла
        image_path = row.filepath
        if not os.path.exists(image_path):
            raise Exception(f"Файл {image_path} не существует")

        # Читаем изображение и применяем функцию обработки изображения
        img = cv2.imread(image_path)
        
        # Некоторые изображения в датасете одноканальные, преобразуем в RGB
        if img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
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
    
    
class ImagenetteDataModule(pl.LightningDataModule):
    """ Загрузчик PyTorch Lighting для датасета ImageNette
    
    Параметры
    ---------
    imagenette_path : str
      Путь к папке с датасетом Imagenette
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
      Словарь для создания экземпляров ImagenetteDataSet.
      Предназначен для передачи разных типов аугментаций.
      Должен содержать все следующие ключи или один из:
        - image_size
        - train_transforms, 
        - base_transforms=None,
        - std
        - mean
    seed : int
      Фиксация генератора случайных чисел.
    """

    def __init__(self, imagenette_path, train_loader_params=None, val_loader_params=None,
                 dataset_params=None, seed=None):

        super().__init__()
        
        assert set(dataset_params.keys()) == set(['base_transforms', 'train_transforms'])

        if seed is not None:
            set_seed(seed)

        self._imagenette_path = imagenette_path

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

        if not dataset_params:
            dataset_params = {}

        self.train_loader_params = train_loader_params
        self.val_loader_params = val_loader_params
        self._dataset_params = dataset_params

        self.make_split_dict()

    def make_split_dict(self):

        self.train_dataset = ImagenetteDataset(imagenette_path=self._imagenette_path, **self._dataset_params)
        self.train_dataset.set_split('train')

        self.val_dataset = ImagenetteDataset(imagenette_path=self._imagenette_path, **self._dataset_params)
        self.val_dataset.set_split('val')


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_loader_params['batch_size'],
                          shuffle=self.train_loader_params['shuffle'], drop_last=self.train_loader_params['drop_last'],
                          num_workers=self.train_loader_params['num_workers'])

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_loader_params['batch_size'],
                          drop_last=self.val_loader_params['drop_last'], shuffle=self.val_loader_params['shuffle'],
                          num_workers=self.val_loader_params['num_workers'])    