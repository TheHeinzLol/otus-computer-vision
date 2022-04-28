from pathlib import Path
import json
import os.path

from tqdm.notebook import tqdm
import requests
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

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


class ImageNetteDataset(Dataset):
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
    
    def __init__(self, imagenette_path, image_size=224, std=(0.229, 0.224, 0.225), mean=(0.485, 0.456, 0.406),
                interpolation='bicubic'):
        
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
            
        self._image_transforms = T.Compose(
                [T.Resize([image_size[0], image_size[1]], interpolation=interpolation),
                 T.ConvertImageDtype(torch.float32),
                 T.Normalize(mean=mean, std=std),
                ]
        )
        
        self._lookup_dict = {
            'train': (self._train_df, self._train_size),
            'val': (self._val_df, self._val_size),
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
        """ Выбор режима работы датасета
        
        Параметры
        ---------
        split : str
          Режим работы датасета.
          Принимает значение train или val
        """

        split = str(split).strip().lower()
        assert split in ('train', 'val'), "split может принимать значения train или val"

        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]
        
    def __len__(self):
        return self._target_size
    
    def __getitem__(self, index):
        
        row = self._target_df.iloc[index]
        img = cv2.imread(row.filepath)
        
        # Некоторые изображения в датасете одноканальные, преобразуем в RGB
        if img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
           
        img = torch.tensor(img)
        img = img.permute(2,0,1)
        
        assert img.shape[0] == 3, f"Изображение не имеет 3 канала ({row.filepath})"
        img = self._image_transforms(img)
        
        return {'target': row.target, 'image': img}