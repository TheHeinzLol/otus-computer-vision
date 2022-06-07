import os.path
import os
import random
from copy import copy
from pathlib import Path
from itertools import combinations

import dlib
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
from IPython import display
import pandas as pd
import numpy as np
import torchvision.transforms as T
from torchvision.io import read_image
from torchvision import models
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sklearn.preprocessing import StandardScaler
import timm
from sklearn.metrics import accuracy_score

from tqdm import tqdm


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
        

class TransferNet:
    """ Создание модели для трансферного обучения

    В качестве экстратктора признаков берется сверточная сеть из VGG16, ResNet18, GoogleNet и AlexNet.
    В экстракторе признаков замораживаются градиенты, кроме небольшой части сверточных блоков в конце экстрактора.
    И кроме слоев Batch Normalization. К выходу экстрактора признаков добавляется полносвязный слой.
    Параметры полносвязного слоя регулируются параметрами класса.

    Параметры
    ---------
    output_dims : list
      Структура полносвязной сети на выходе экстратора признаков.
      Длина списка - количество полносвязных слоев.
      Значения списка - кол-во нейронов в каждом полносвязном слое.
      Пример: [1024, 512, 256, 1] - четыре полносвязных слоя.
              Первый слой с 1024 нейронами, второй слой с 512 нейронами,
              третий слой с 256 нейронами и последний слой с 1 нейроном.
      По умолчанию [128, 256, 1].
    pretrained : bool
      Если True, то будут загружена предобученная модель на ImageNet.
      По умолчанию True.
      !!! ВАЖНО: Класс не тестировался на не предобученных моделях.
          Структура модели при значении pretrained=False может отличаться, поведение класса может быть некорректным.
    full_trainable : bool
      Сделать ли модель полностью обучаемой. По умолчанию False. Если False, то для обучения доступна только
      полносвязная сеть и около 10% окончания экстрактора признаков
    seed : int
      Фиксация генератора случайных чисел.
    dropout : float
      Значение вероятности для всех слоев Dropout полносвязной сети.
      По умолчанию 0.5
    """

    def __init__(self, output_dims=None, dropout=0.5, pretrained=True, full_trainable=False, seed=None):

        if seed is not None:
            set_seed(seed)

        if not output_dims:
            output_dims = [128, 256, 1]

        output_dims =list(output_dims)

        assert 0.0 <= dropout <= 1.0, f"Значение dropout может находитсья в диапазоне от 0 до 1. Текущее значение: ({dropout})"
        assert isinstance(pretrained, bool), f"Значение pretrained должно иметь тип bool. Текущее значение: ({pretrained})"
        assert len(output_dims) > 0, f"Список output_dims должен быть больше 0. Текущее значение: ({output_dims})"
        assert all([isinstance(x, int) for x in output_dims]), f"Все значения в output_dims должны быть целыми. Текущее значение: ({output_dims})"

        self.output_dims = output_dims
        self.dropout = dropout
        self.pretrained = pretrained
        self.full_trainable = full_trainable

    def _make_fc(self, input_dim, first_dropout=True):
        """" Создание полносвязного слоя

        Количество полносвязных слоев и количество нейронов берется из переменной self.output_dims
        Значение вероятности Dropout берется из self.dropout

        Параметры
        ---------
        input_dim : int
          Размер входа полносвязной сети
        first_dropout : bool
          Если True, то первый слой полносвязной сети - Dropout. Иначе Linear

        Результат
        ---------
        fc : torch Sequential
          Последовательность слоев типа nn.Linear
        """

        if first_dropout:
            layers = [nn.Dropout(self.dropout)]
        else:
            layers = []

        # На основе self.output_dims конфигурируем каждый полносвязный слой
        for index, output_dim in enumerate(self.output_dims):
            layers.append(nn.Linear(input_dim, output_dim))
            # Если слой не последний, то после него добавляем Relu и Dropout
            if index != len(self.output_dims) - 1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self.dropout))
            input_dim = output_dim

        return nn.Sequential(*layers)

    def _prepare_transfernet(self, transfer_model, no_grad_layer=None, first_dropout=True):
        """ Подготовка сети для трансферного обучения

        Параметры
        ---------
        transfer_model : torch model
          Модель из torchvision.models
        no_grad_layer : str
          Название слоя, вплоть до которого будет выключено обучение параметров. По умолчанию None.
          Если None, то все параметры модели будут обучаться.
        first_dropout : bool
          Если True, то первый слой полносвязной сети - Dropout. Иначе Linear

        Результат
        ---------

        """

        if no_grad_layer is not None:

            # Замораживаем градиенты у модели, отключаем обучение параметров
            for name, param in transfer_model.named_parameters():

                # Выключаем градиенты во всех блоках модели вплоть до no_grad_layer
                if no_grad_layer in name:
                    break
                # Не выключаем обучение параметров у всех блоков Batch Normalization
                if 'bn' in name:
                    continue

                param.requires_grad = False

        # Находим размер выхода экстрактора признаков модели
        # Значение находится в свойстве in_features у первого слоя типа Linear после экстрактора
        # Модели не однотипные, полносвязный слой может храниться в параметре fc или classifier
        if 'fc' in transfer_model.__dir__():
            fc_name = 'fc'
            fc_in_features = transfer_model.fc.in_features
        elif 'head' in transfer_model.__dir__():
            fc_name = 'head'
            fc_in_features = transfer_model.head.in_features
        elif 'classifier' in transfer_model.__dir__():
            fc_name = 'classifier'
            if isinstance(transfer_model.classifier, nn.Linear):
                fc_in_features = transfer_model.classifier.in_features
            else:
                # Обычно модели с блоком classifier содержат последовательность слоев.
                # Бывает такое, что первым слоем идет Dropout и у него нет параметра in_features
                try:
                    fc_in_features = transfer_model.classifier[0].in_features
                except:
                    fc_in_features = transfer_model.classifier[1].in_features
        else:
            raise Exception("В модели не найден блок полносвязной сети")

        # Заменяем у модели полносвязный слой на сгенерированный
        setattr(transfer_model, fc_name, self._make_fc(input_dim=fc_in_features, first_dropout=first_dropout))

        return transfer_model

    def _make_swin_base_patch4_window7_224(self):
        """ Создание модели на базе swin_base_patch4_window7_224

        Используетя параметр self.pretrained для загрузки предобученной модели.

        Результат
        ---------
        model : torch model
          Модель swin_base_patch4_window7_224
        """

        transfer_model = timm.create_model('swin_base_patch4_window7_224', pretrained=self.pretrained)
        no_grad_layer = None if self.full_trainable else "FULL_BLOCK_MODEL"
        transfer_model = self._prepare_transfernet(transfer_model=transfer_model, no_grad_layer=no_grad_layer)

        return transfer_model
    
    def _make_swin_base_patch4_window7_224_in22k(self):
        """ Создание модели на базе swin_base_patch4_window7_224_in22k

        Используетя параметр self.pretrained для загрузки предобученной модели.

        Результат
        ---------
        model : torch model
          Модель swin_base_patch4_window7_224
        """

        transfer_model = timm.create_model('swin_base_patch4_window7_224_in22k', pretrained=self.pretrained)
        no_grad_layer = None if self.full_trainable else "FULL_BLOCK_MODEL"
        transfer_model = self._prepare_transfernet(transfer_model=transfer_model, no_grad_layer=no_grad_layer)

        return transfer_model

    def _make_swin_tiny_patch4_window7_224(self):
        """ Создание модели на базе tiny_patch4_window7_224

        Используетя параметр self.pretrained для загрузки предобученной модели.

        Результат
        ---------
        model : torch model
          Модель tiny_patch4_window7_224
        """

        transfer_model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=self.pretrained)
        no_grad_layer = None if self.full_trainable else "FULL_BLOCK_MODEL"
        transfer_model = self._prepare_transfernet(transfer_model=transfer_model, no_grad_layer=no_grad_layer)

        return transfer_model
    
    def _make_swin_large_patch4_window12_384(self):
        """ Создание модели на базе tiny_patch4_window7_224

        Используетя параметр self.pretrained для загрузки предобученной модели.

        Результат
        ---------
        model : torch model
          Модель swin_large_patch4_window12_384
        """

        transfer_model = timm.create_model('swin_large_patch4_window12_384', pretrained=self.pretrained)
        no_grad_layer = None if self.full_trainable else "FULL_BLOCK_MODEL"
        transfer_model = self._prepare_transfernet(transfer_model=transfer_model, no_grad_layer=no_grad_layer)

        return transfer_model
    
    def _make_swin_large_patch4_window7_224(self):
        """ Создание модели на базе swin_large_patch4_window7_224

        Используетя параметр self.pretrained для загрузки предобученной модели.

        Результат
        ---------
        model : torch model
          Модель swin_large_patch4_window7_224
        """

        transfer_model = timm.create_model('swin_large_patch4_window7_224', pretrained=self.pretrained)
        no_grad_layer = None if self.full_trainable else "FULL_BLOCK_MODEL"
        transfer_model = self._prepare_transfernet(transfer_model=transfer_model, no_grad_layer=no_grad_layer)

        return transfer_model

    def _make_vit_base_patch16_224(self):
        """ Создание модели на базе swin_large_patch4_window12_384

        Используетя параметр self.pretrained для загрузки предобученной модели.

        Результат
        ---------
        model : torch model
          Модель swin_large_patch4_window12_384
        """

        transfer_model = timm.create_model('vit_base_patch16_224', pretrained=self.pretrained)
        no_grad_layer = None if self.full_trainable else "FULL_BLOCK_MODEL"
        transfer_model = self._prepare_transfernet(transfer_model=transfer_model, no_grad_layer=no_grad_layer,
                                                   first_dropout=False)

        return transfer_model

    def _make_efficientnetv2_s(self):
        """ Создание модели на базе efficientnetv2_s

                Используетя параметр self.pretrained для загрузки предобученной модели.

                Результат
                ---------
                model : torch model
                  Модель efficientnetv2_l
                """

        transfer_model = timm.create_model('tf_efficientnetv2_s', pretrained=self.pretrained)
        no_grad_layer = None if self.full_trainable else 'blocks.5.13'
        transfer_model = self._prepare_transfernet(transfer_model=transfer_model, no_grad_layer=no_grad_layer)

        return transfer_model

    def _make_efficientnetv2_m(self):
        """ Создание модели на базе efficientnetv2_s

                Используетя параметр self.pretrained для загрузки предобученной модели.

                Результат
                ---------
                model : torch model
                  Модель efficientnetv2_l
                """

        transfer_model = timm.create_model('tf_efficientnetv2_m', pretrained=self.pretrained)
        no_grad_layer = None if self.full_trainable else 'blocks.6.0'
        transfer_model = self._prepare_transfernet(transfer_model=transfer_model, no_grad_layer=no_grad_layer)

        return transfer_model

    def _make_efficientnetv2_b1(self):
        """ Создание модели на базе efficientnetv2_b1

                Используетя параметр self.pretrained для загрузки предобученной модели.

                Результат
                ---------
                model : torch model
                  Модель efficientnetv2_b1
                """

        transfer_model = timm.create_model('tf_efficientnetv2_b1', pretrained=self.pretrained)
        no_grad_layer = None if self.full_trainable else 'blocks.5.5'
        transfer_model = self._prepare_transfernet(transfer_model=transfer_model, no_grad_layer=no_grad_layer)

        return transfer_model

    def _make_efficientnetv2_b3(self):
        """ Создание модели на базе efficientnetv2_b3

                Используетя параметр self.pretrained для загрузки предобученной модели.

                Результат
                ---------
                model : torch model
                  Модель efficientnetv2_l
                """

        transfer_model = timm.create_model('tf_efficientnetv2_b3', pretrained=self.pretrained)
        no_grad_layer = None if self.full_trainable else 'blocks.5.8'
        transfer_model = self._prepare_transfernet(transfer_model=transfer_model, no_grad_layer=no_grad_layer)

        return transfer_model

    def _make_resnet18(self):
        """ Создание модели на базе ResNet18

        Используетя параметр self.pretrained для загрузки предобученной модели.

        Результат
        ---------
        model : torch model
          Модель ResNet18
        """

        transfer_model = models.resnet18(pretrained=self.pretrained)
        no_grad_layer = None if self.full_trainable else 'layer4'
        transfer_model = self._prepare_transfernet(transfer_model=transfer_model, no_grad_layer=no_grad_layer)

        return transfer_model

    def _make_vgg16(self):
        """ Создание модели на базе VGG16

            Используетя параметр self.pretrained для загрузки предобученной модели.

            Результат
            ---------
            model : torch model
              Модель VGG16
        """

        transfer_model = models.vgg16(pretrained=self.pretrained)
        no_grad_layer = None if self.full_trainable else 'features.24'
        transfer_model = self._prepare_transfernet(transfer_model=transfer_model, no_grad_layer=no_grad_layer)

        return transfer_model

    def _make_googlenet(self):
        """ Создание модели на базе GoogleNet

            Используетя параметр self.pretrained для загрузки предобученной модели.

            Результат
            ---------
            model : torch model
              Модель GoogleNet
        """

        transfer_model = models.googlenet(pretrained=self.pretrained)
        transfer_model.dropout.p = 0.0
        no_grad_layer = None if self.full_trainable else 'inception5a'
        transfer_model = self._prepare_transfernet(transfer_model=transfer_model, no_grad_layer=no_grad_layer)

        return transfer_model

    def _make_alexnet(self):
        """ Создание модели на базе AlexNet

            Используетя параметр self.pretrained для загрузки предобученной модели.

            Результат
            ---------
            model : torch model
              Модель AlexNet
        """

        transfer_model = models.alexnet(pretrained=self.pretrained)
        no_grad_layer = None if self.full_trainable else 'features.8'
        transfer_model = self._prepare_transfernet(transfer_model=transfer_model, no_grad_layer=no_grad_layer)

        return transfer_model

    def make_model(self, name):
        """ Создание модели для трансферного обучения.

        Параметры
        ---------
        name : str
          Название модели. Может принимать значение resnet18, vgg16, alexnet, googlenet или efficientnetv2_b3

        Результат
        ---------
        model : torch model
        """

        name = str(name).lower().strip()

        if name == 'resnet18':
            return self._make_resnet18()
        elif name == 'vgg16':
            return self._make_vgg16()
        elif name == 'alexnet':
            return self._make_alexnet()
        elif name == 'googlenet':
            return self._make_googlenet()
        elif name == 'efficientnetv2_s':
            return self._make_efficientnetv2_s()
        elif name == 'efficientnetv2_m':
            return self._make_efficientnetv2_m()
        elif name == 'efficientnetv2_b1':
            return self._make_efficientnetv2_b1()
        elif name == 'efficientnetv2_b3':
            return self._make_efficientnetv2_b3()
        elif name == 'swin_tiny_patch4_window7_224':
            return self._make_swin_tiny_patch4_window7_224()
        elif name == 'swin_base_patch4_window7_224':
            return self._make_swin_base_patch4_window7_224()
        elif name == 'swin_base_patch4_window7_224_in22k':
            return self._make_swin_base_patch4_window7_224_in22k()
        elif name == 'swin_large_patch4_window12_384':
            return self._make_swin_large_patch4_window12_384()
        elif name == 'swin_large_patch4_window7_224':
            return self._make_swin_large_patch4_window7_224()
        elif name == 'vit_base_patch16_224':
            return self._make_vit_base_patch16_224()
        else:
            raise AttributeError("Параметр name принимает неизвестное значение")

    def __call__(self, name):
        return self.make_model(name=name)
        
        
class Vocabulary:
    """ Класс для кодирования слов и символов

    Параметры
    ---------
    token_to_idx : dict
      Словарь в формате {токен: индекс токена}
      По умолчанию None, будет создан чистый словарь.
    """
    def __init__(self, token_to_idx=None):

        if token_to_idx is None:
            token_to_idx = {}

        self.token_to_idx = token_to_idx
        self.idx_to_token = {i: k for k, i in token_to_idx.items()}

    def add_token(self, token):
        """ Добавление токена в словарь объекта

        Параметры
        ---------
        token : str
          Токен для добавления в словарь

        Результат 
        ---------
        index : int
          Индекс токена в словаре
        """
        if token in self.token_to_idx:
            index = self.token_to_idx[token]
        else:
            index = len(self)
            self.token_to_idx[token] = index
            self.idx_to_token[index] = token
        return index

    def lookup_token(self, token):
        """ Поиск индекса для токена в словаре 

        Параметры
        ---------
        token : str
          Токен для поиска в словаре

        Результат
        ---------
        index : int
          Индекс токена в словаре
        """

        assert token in self.token_to_idx, f"Символ {token} отсутствует в словаре"
        return self.token_to_idx[token]

    def lookup_index(self, index):
        """ Поиск словая по индексу в словаре 

        Параметры
        ---------
        index : int
          Индекс для поиска в словаре

        Результат
        ---------
        token : str
          Токен для заданного индекса
        """
        assert index in self.idx_to_token, f"Индекс {index} отсутствует в словаре"
        return self.idx_to_token[index]

    def to_serializable(self):
        """ Формирование словаря с сериализованными параметрами объекта класса """
        return {'token_to_idx': self._token_to_idx}

    @classmethod
    def from_serializable(cls, contents):
        """ Объект класса  Vocabulary из сериализованного словаря"""
        return cls(**contents)

    def __len__(self):
        return len(self.token_to_idx)

    def __str__(self):
        return f"<Vocabulary(len={len(self)})>"
    
    
class FaceDetector:
    """ Детектор лиц и лицевых ориентиров"""
    
    def __init__(self):
        
        five_landmarks_predictor_path = 'shape_predictor_5_face_landmarks.dat'
        sixty_eight_landmarks_predictor_path = 'shape_predictor_68_face_landmarks.dat'
        
        # Детектор границ лица
        self._detector = dlib.get_frontal_face_detector()
        # Детектор 5 ориентиров лица
        self._sp_5 = dlib.shape_predictor(five_landmarks_predictor_path)
        # Детектор 68 ориентиров лица
        self._sp_68 = dlib.shape_predictor(sixty_eight_landmarks_predictor_path)
        
    @staticmethod
    def _rectangle2numpy(dets):
        """ Преобразуем предсказанные боксы dlib класса rectangle в массив numpy
        
        Параметры
        ---------
        dets : list
          Список предсказанных боксов dlib для изображения
        
        Результат
        ---------
        numpy_bboxes : np.ndarray
          Массив в формате [[x1, y1, x2, y2], [], ...]
        """
        numpy_bboxes = np.zeros(shape=(len(dets), 4), dtype=np.int16)
        for idx, rectangle in enumerate(dets):
            x1, y1 = rectangle.left(), rectangle.top()
            x2, y2 = rectangle.right(), rectangle.bottom()
            numpy_bboxes[idx] = x1, y1, x2, y2

        return numpy_bboxes
    
    @staticmethod
    def _landmarks2numpy(predict):
        """ Преобразуем предсказанные лицевые ориентиры dlib в массив numpy
        
        Параметры
        ---------
        predict : list
          Список предсказанных ландмарок dlib для одного бокса
        
        Результат
        ---------
        predict_np : np.ndarray
          Массив в формате [[x1, y1], [x2, y2], [], ...]
        """
        predict_np = np.zeros((68, 2), dtype="int")
        for i in range(0, 68):
            predict_np[i] = (predict.part(i).x, predict.part(i).y)
            
        return predict_np
    
    def _draw_bboxes(self, img, dets):
        """ Рисуем предсказанные границы (боксы) лица
        
        Параметры
        ---------
        img : np.ndarray
          Изображение в формате HxWxC
        dets : list
          Список предсказаний боксов лица из dlib
          
        Результат
        ---------
        img : np.ndarray
          Изображение с отрисованными боксами
        """
    
        numpy_bboxes = self._rectangle2numpy(dets)
        img = np.copy(img)

        for bbox in numpy_bboxes:
            x1, y1 = int(bbox[0]), int(bbox[1]) 
            x2, y2 = int(bbox[2]), int(bbox[3]) 

        cv2.rectangle(img, (x1, y1), (x2, y2), [255,0,0], 5)

        return img
    
    def _face_align(self, img, dets):
        """ Выравнивание лица по горизонтали
        
        Параметры
        ---------
        img : np.ndarray
          Изображение в формате HxWxC
        dets : list
          Список предсказаний боксов лица из dlib
          
        Результат
                ---------
        images : list
          Список выровненных изображений лиц в формате HxWxC
        """
    
        faces = dlib.full_object_detections()
        for detection in dets:
            faces.append(self._sp_5(img, detection))
        images = dlib.get_face_chips(img, faces)

        return images
    
    @staticmethod
    def _resize_image(img, size):
        """ Изменение размера изображения """
        
        img = cv2.resize(img, size)
        
        return img
    
    @staticmethod
    def _rgb2grayscale(img):
        """ Конвертация из RGB в Grayscale """
        
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        return img
    
    def _get_landmarks(self, img, dets):
        """" Получение лицевых ориентиров
        
        Параметры
        ---------
        img : np.ndarray
          Изображение в формате HxWxC
        dets : list
          Список предсказаний боксов лица из dlib
          
        Результат
                ---------
        landmarks : list
          Список ландмарок в формате массива numpy
          Каждый массив ландмарок соответствует порядку лиц в dets
        """
        
        landmarks = []
        
        for detection in dets:
            predict = self._sp_68(img, detection)
            predict = self._landmarks2numpy(predict)
            landmarks.append(predict)
            
        return landmarks
    
    @staticmethod
    def _get_centroids(landmarks):
        """ Поиск центральной точки для ландмарок каждого лица
        
        Центроид находится с помощью нахождения среднего вектора всех лицевых ориентиров.
        
        Параметры
        ---------
        landmarks : list
          Список ландмарок в формате массива numpy. 
          Каждый массив numpy - ландмарки для одного лица.
          
        Результат
        ---------
        centroids : list
          Список центроидов в формате [[x, y], [x, y], [], ...]
          
        """
        centroids = []
        for landmark in landmarks:
            centroid = np.array(np.sum(landmark, axis=0) / len(landmark), dtype=np.int16)
            centroids.append(centroid)
            
        return centroids
    
    def _crop_faces(self, img, dets):
        """ Вырезка изображений лиц по результатам детекции
        
        Параметры
        ---------
        img : np.ndarray
          Изображение в формате HxWxC
        dets : list
          Список предсказаний боксов лица из dlib
          
        Результат
                ---------
        cropped_images : list
          Список изображений лиц в формате HxWxC
        """
        
        bboxes = self._rectangle2numpy(dets)
        cropped_images = []
        
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox.tolist()
            cropped_image = img[y1:y2, x1:x2]
            cropped_images.append(cropped_image)
            
        return cropped_images
    
    def detect(self, img, align=True, landmarks=False, size=None, grayscale=False):
        """ Запуск детекции лица
        
        В качестве результата выдает изображения найденных лиц и их лицевые ориентиры
        
        Параметры
        ---------
        img : np.ndarray
          Изображение в формате HxWxC
        align : bool
          Если True, то будет выполнено выравнивание лица
        landmarks : bool
          Если True, то метод вернет воторым элементом список лицевых ориентиров.
          Иначе только список изображений лиц.
        size : tuple
          Размер изображений лиц для resize в формате (W, H)
        grayscale : bool
          Если True, то обнаруженные изображения лиц будут конвертированы в Grayscale
          
        Результат
        ---------
        face_images, landmarks : tuple 
          При условии landmarks=True
          
        face_images : list 
          При условии landmarks=False
        """
        
        self.dets_ = self._detector(img, 1)
        num_faces = len(self.dets_)
        
        if num_faces == 0:
            if landmarks:
                return None, None
            else:
                return None
        
        if align:
            face_images = self._face_align(img, self.dets_)
        else:
            face_images = self._crop_faces(img, self.dets_)
         
        if size is not None:
            for idx, image in enumerate(face_images):
                face_images[idx] = self._resize_image(face_images, size)
                
        if grayscale:
            for idx, image in enumerate(face_images):
                face_images[idx] = self._rgb2grayscale(image)
         
        if not landmarks:
            return face_images
        
        landmarks_ = self._get_landmarks(img, self.dets_)
        
        return face_images, landmarks_
    
    def calculate_landmarks_features(self, landmarks, image_size):
        """ Расчет признаков мимики человека на основе ландмарок
        
        Данный принцип был взять из статьи 
        Facial Expression Recognition using Facial Landmark Detection and Feature Extraction via Neural Network
        по адресу https://arxiv.org/pdf/1812.04510.pdf
        
        Рассчитываем длины векторов между ландмарками в рамках своего кластера.
        Будут взяты кластера бровей, глаз и рта.
        А также расчитываем длины вектора каждой ландмарки с центром начала координат в виде центроида всех ландмарок лица.
        
        Параметры
        ---------
        landmarks : list
          Список предсказанных ландмарок для каждого лица в формате numpy
          
        Результат
        ---------
        result_features : np.ndarray
          Массив признаков на основе ландмарок лиц в формате Nx262,
          где N - количество лиц
        """
        
        def calc_norm_for_combinations(combinations):
            """ Находим Евклидово расстояние между комбинациями точек """
            distances = []
            for combo in combinations:
                dist = np.sqrt(np.sum(np.square(combo[0] - combo[1])))
                distances.append(dist)   
            return np.array(distances)
        
        def calc_norm_for_centroid(landmarks, centroid):
            """ Находим Евклидово расстоение между центроидом и лицевым ориентиром """
            distances = []
            for landmark in landmarks:
                dist = np.sqrt(np.sum(np.square(centroid - landmark)))
                distances.append(dist)
            return np.array(distances)
        
        # Результирующий список признаков для лиц на основе лицевых ориентиров
        result_features = []
        
        # Все ландмарки признаков для визуализации
        all_feature_landmarks = []
        
        x_scale = image_size[1]
        y_scale = image_size[0]
        
        # Пробегаемся по каждому лицу
        for idx, face_landmarks in enumerate(landmarks):
            
            face_landmarks[:, 0] = np.clip(face_landmarks[:, 0], 0, x_scale)
            face_landmarks[:, 1] = np.clip(face_landmarks[:, 1], 0, y_scale)
            
            # Группа ландмарок для левой брови
            left_eyebrow = face_landmarks[18:23] / (x_scale, y_scale)
            # Группа ландмарок для правой брови
            right_eyebrow = face_landmarks[23:28] / (x_scale, y_scale)
            # Группа ландмарок для левого глаза
            left_eye = face_landmarks[37:43] / (x_scale, y_scale)
            # Группа ландмарок для правого глаза
            right_eye = face_landmarks[43:49] / (x_scale, y_scale)
            # Группа ландмарок для рта
            mouth = face_landmarks[49:69] / (x_scale, y_scale)
            
            face_feature_landmarks = np.vstack([
                left_eyebrow, right_eyebrow, left_eye, right_eye, mouth
            ])
            
            all_feature_landmarks.append(face_feature_landmarks)
                      
            # Поулчаем центроид по ландмаркам лица
            centroid = self._get_centroids([face_feature_landmarks])[0]
                
            # Дистанции между комбинациями точек группы ландмарок левой брови
            left_eyebrow_dictances = calc_norm_for_combinations(combinations(left_eyebrow, 2))
            # Дистанции между комбинациями точек группы ландмарок правой брови
            right_eyebrow_dictances = calc_norm_for_combinations(combinations(right_eyebrow, 2))
            # Дистанции между комбинациями точек группы ландмарок левого глаза
            left_eye_dictances = calc_norm_for_combinations(combinations(left_eye, 2))
            # Дистанции между комбинациями точек группы ландмарок правого глаза
            right_eye_dictances = calc_norm_for_combinations(combinations(right_eye, 2))
            # Дистанции между комбинациями точек группы ландмарок рта
            mouth_dictances = calc_norm_for_combinations(combinations(mouth, 2))
            
            # Дистанции между центроидом лица и ландмарками левой брови
            centroid_left_eyebrow_dictances = calc_norm_for_centroid(left_eyebrow, centroid)
            # Дистанции между центроидом лица и ландмарками правой брови
            centroid_right_eyebrow_dictances = calc_norm_for_centroid(right_eyebrow, centroid)
            # Дистанции между центроидом лица и ландмарками левого глаза
            centroid_left_eye_dictances = calc_norm_for_centroid(left_eye, centroid)
            # Дистанции между центроидом лица и ландмарками правого глаза
            centroid_right_eye_dictances = calc_norm_for_centroid(right_eye, centroid)
            # Дистанции между центроидом лица и ландмарками рта
            centroid_mouth_dictances = calc_norm_for_centroid(mouth, centroid)
            
            # Объединяем все дистанции в единый вектор
            face_landmark_features = np.hstack(
            [
                left_eyebrow_dictances,
                right_eyebrow_dictances,
                left_eye_dictances,
                right_eye_dictances,
                mouth_dictances,
                centroid_left_eyebrow_dictances,
                centroid_right_eyebrow_dictances,
                centroid_left_eye_dictances,
                centroid_right_eye_dictances,
                centroid_mouth_dictances,
            ])
            
            
            result_features.append(face_landmark_features)
            
        result_features = np.vstack(result_features)
        
        return result_features
    
    def visualise_landmarks(self, img, landmarks):
        """ Визуализация ландмарок на изображении
        
        Параметры
        ---------
        img : np.ndarray
          Изображение в формате HxWxC
        landmarks : list
          Список предсказанных ландмарок для каждого лица в формате numpy
        """
        
        img_with_landmarks = np.copy(img)
        
        centroids = self._get_centroids(landmarks)
        
        for idx, face_landmarks in enumerate(landmarks):
            
            imshow_kwargs = {'cmap': 'gray'} if len(img_with_landmarks.shape) == 2 else {}
            cv2_circle_kwargs = {'color': 0} if len(img_with_landmarks.shape) == 2 else {'color': (255, 0, 0)} 
            cv2_centroid_kwargs = {'color': 255} if len(img_with_landmarks.shape) == 2 else {'color': (0, 0, 255)} 

            thickness = int(10 * img.shape[0]*img.shape[1] / (1400 * 2000))
            
            for face_landmark in face_landmarks:
                x, y = int(face_landmark[0]), int(face_landmark[1])
                cv2.circle(img_with_landmarks, (x, y), radius=0, thickness=thickness, **cv2_circle_kwargs)
               
            cv2.circle(img_with_landmarks, (centroids[idx][0], centroids[idx][1]), radius=0, thickness=thickness*2,
                       **cv2_centroid_kwargs)
            
            plt.imshow(img_with_landmarks, **imshow_kwargs)
            plt.show()  


class FaceEmojiDataMining:
    """ Генерация признаков для датасета Face expression recognition dataset на основе landmarks
    
    Метод start создаст объект self.mining_dataframe_, датафрейм с данными для обучения моделей.
    
    Датафрейм будет содержать следующие данные:
        image_path - Путь к изображению
        split - тип разделения фотографии, test или val
        target - целевая метка эмоции человека на изображении
        emoji_name - название эмоции
        поля f_0 ... f_261 - признаки на основе ландмарок (если лицо на изображении не удалось задетектить, то все признаки будут равны -1)
        
    Параметры
    ---------
    train_images_filepath : str
      Путь к папке изображений для обучения.
      По умолчанию face_expression_recognition_dataset/train/
    val_images_filepath : str
      Путь к папке изображений для валидации.
      По умолчанию face_expression_recognition_dataset/train/
    """
    
    def __init__(self, train_images_filepath="face_expression_recognition_dataset/train/",
                 val_images_filepath="face_expression_recognition_dataset/validation/"):
        
        self.train_images_filepath = Path(train_images_filepath)
        self.val_images_filepath = Path(val_images_filepath)
        
        self._face_detector = FaceDetector()
        self._emoji_vocab = Vocabulary()
        
        
    def _make_emoji_vocab(self):
        """ Наполняем словарь эмоций на основе папки для обучения.
        Каждое название папки - название эмоции
        """
        self._emoji_vocab = Vocabulary()
        emoji_names = os.listdir(self.train_images_filepath)
        for emoji_name in emoji_names:
            self._emoji_vocab.add_token(emoji_name)
            
    def _init_dataframe(self):
        """ Создаем начальный датафрейм с полями image_path, split и target"""
        
        self.emoji_df_ = pd.DataFrame()
        print("==============================================")
        print("Подготовка базового датафрейма")
        for split, images_dir in tqdm(zip(['train', 'val'], [self.train_images_filepath, self.val_images_filepath]),
                                     total=2):
        
            for emoji_dir in images_dir.iterdir():

                emoji_name = emoji_dir.name
                emoji_code = self._emoji_vocab.lookup_token(emoji_name)

                for photo_name in emoji_dir.iterdir():
                    self.emoji_df_ = pd.concat([self.emoji_df_, pd.DataFrame({'image_path': [str(photo_name)],
                                                                              'split': [split],
                                                                              'target': [emoji_code],
                                                                              'emoji_name': [emoji_name],
                                                                             })
                                               ])
                    
    def _get_landmark_features(self):
        """ Дополняем датафрейм признаками ландмарок, добавляем поля f_0 ... f_261"""
        
        print("==============================================")
        print("Генерация фичей")
              
        land_f_count = 262
        total_face_features = []
        for _, face_row in tqdm(self.emoji_df_.iterrows(), total=len(self.emoji_df_)):
            
            img = cv2.imread(face_row.image_path, cv2.IMREAD_GRAYSCALE)
            face_images, landmarks = self._face_detector.detect(img, align=False, landmarks=True)
            if landmarks is None:
                face_features = np.array([-1]*land_f_count)
            else:
                face_features = self._face_detector.calculate_landmarks_features(landmarks, image_size=img.shape)[:1]
                
            total_face_features.append(face_features)
            
        total_face_features = np.vstack(total_face_features)
        
        features_names = ['f_' + str(idx) for idx in range(land_f_count)]
        
        self.emoji_df_.loc[:, features_names] = total_face_features
                    
    def start(self):
        """ Запуск майнинга данных
        
        Результат
        ---------
        self.mining_dataframe_ : pd.DataFrame
        """
        self._make_emoji_vocab()
        self._init_dataframe()
        self._get_landmark_features()
        
    
    def __call__(self):
        return self.start()
    
    
class FacesEmojiDataset(Dataset):
    """ Датасет face expression recognition
    
    Параметры
    ---------
    emoji_faces_xlsx : pd.DataFrame
    """
    def __init__(self, emoji_faces_df, image_size=48, only_landmark_features=False, only_detected_landmarks=False,
                 std_for_add_features=False, train_augmentation=True):
        
        self.only_landmark_features = only_landmark_features
        self._only_detected_landmarks = only_detected_landmarks
        self._std_for_add_features = std_for_add_features
        
        self._additive_features_names = ['f_' + str(idx) for idx in range(0, 262)]
        assert_columns = ['image_path', 'split', 'target', 'emoji_name'] + self._additive_features_names
        
        self._emoji_faces_df = emoji_faces_df
        assert set(assert_columns) == set(self._emoji_faces_df.columns)
        
        if self._only_detected_landmarks:
            self._emoji_faces_df = self._emoji_faces_df.loc[self._emoji_faces_df.f_1 >= 0]
            
            
        # Средние значения для IMAGE NET, используются при стандартизации изображений
        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        # СКО для IMAGE NET, используются при стандартизации изображений
        IMAGENET_STD = [0.229, 0.224, 0.225]
        
        # Определяем функцию обработки изображений для валидации и теста
        if train_augmentation:
            self._train_transforms = T.Compose(
                [T.Resize([image_size, image_size]),
                 T.RandomHorizontalFlip(p=0.5),
                 T.RandomVerticalFlip(p=0.5),
                 T.RandomErasing(p=0.5, scale=(0.02,0.1), ratio=(0.3,3.3)),   
                 T.ConvertImageDtype(torch.float32),
                 T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                 ]
            )
        else:
            self._train_transforms = T.Compose(
                [T.Resize([image_size, image_size]),
                 T.ConvertImageDtype(torch.float32),
                 T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                 ]
            )      
            
        self._val_transforms = T.Compose(
                [T.Resize([image_size, image_size]),
                 T.ConvertImageDtype(torch.float32),
                 T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                 ]
            )      
            
        self._train_df = self._emoji_faces_df[self._emoji_faces_df.split == 'train'].copy()
        self._train_size = len(self._train_df)
        self._val_df = self._emoji_faces_df[self._emoji_faces_df.split == 'val'].copy()
        self._val_size = len(self._val_df)

        self._lookup_dict = {
            'train': (self._train_df, self._train_size, self._train_transforms),
            'val': (self._val_df, self._val_size, self._val_transforms),
        }
        
        if self._std_for_add_features:
            self._make_std_features()        

        # По умолчанию включаем режим обучения
        self.set_split('train')

    def _make_std_features(self):
        """ Преобразование дополнительных признаков с помощью стандартизации

        Параметры
        skf_dict : словарь
          Словарь с фолдами обучающего датасета и тренировочного
          Формат: {номер фолда: (датасет обучения, датасет валидации)}

        Результат
        ---------
        std_skf_dict : dict
          Словарь с фолдами со стандартизованными доп. признаками
        """

        features_train_df = self._train_df.loc[:, self._additive_features_names]
        features_val_df = self._val_df.loc[:, self._additive_features_names]

        self._scaler = StandardScaler()
        std_features_train = self._scaler .fit_transform(features_train_df)
        std_features_val = self._scaler.transform(features_val_df)

        for idx, feature_name in enumerate(self._additive_features_names):
            self._train_df.loc[:, feature_name] = std_features_train[:, idx]
            self._val_df.loc[:, feature_name] = std_features_val[:, idx]

    def set_split(self, split='train'):

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
        'landmark_features': дополнительные признаки,
        'image_path': путь к файлу изображения,
        'target': целевая метка эмоции,
        }
        """

        # Получаем строку датафрейма по его индексу
        row = self._target_df.iloc[index, :]

        # Словарь, который будем возвращать, инициализируем его значением хэша изображения
        model_data = {'landmark_features': torch.tensor(row.loc[self._additive_features_names], dtype=torch.float32),
                      'target':  row.target
                     }
        
        if self.only_landmark_features:
            return model_data
        
        image_path = row.image_path
        if not os.path.exists(image_path):
            raise Exception(f"Файл {image_path} не существует")

        # Читаем изображение и применяем функцию обработки изображения
        img_source = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img_source, cv2.COLOR_GRAY2RGB)
        img = torch.tensor(img).permute(2,0,1)
        
        # Добавляем в возвращаемый словарь обработанное изображение
        model_data['x_data'] = self._target_transforms(img)
        # Добавляем в возвращаемый словарь путь к исходному изображению
        model_data['image_path'] = image_path

        return model_data
    

class FaceEmojiDataModule(pl.LightningDataModule):

    def __init__(self, emoji_faces_df, train_loader_params=None, val_loader_params=None, 
                 dataset_params=None, seed=None):

        super().__init__()

        if seed is not None:
            set_seed(seed)

        self.emoji_faces_df = emoji_faces_df

        if not train_loader_params:
            train_loader_params = {
                'batch_size': 64,
                'shuffle': True,
                'num_workers': 2,
                'drop_last': True,
            }

        if not val_loader_params:
            val_loader_params = {
                'batch_size': 64,
                'shuffle': False,
                'num_workers': 2,
                'drop_last': False
            }

        if not dataset_params:
            dataset_params = {
                'only_landmark_features': False, 
                'only_detected_landmarks': True,
                'std_for_add_features': True
            }

        self.train_loader_params = train_loader_params
        self.val_loader_params = val_loader_params
        self.dataset_params = dataset_params

        self.make_split_dict()

    def make_split_dict(self):

        self.train_dataset = FacesEmojiDataset(emoji_faces_df=self.emoji_faces_df, **self.dataset_params)
        self.train_dataset.set_split('train')

        self.val_dataset = FacesEmojiDataset(emoji_faces_df=self.emoji_faces_df, **self.dataset_params)
        self.val_dataset.set_split('val')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_loader_params['batch_size'],
                          shuffle=self.train_loader_params['shuffle'], drop_last=self.train_loader_params['drop_last'],
                          num_workers=self.train_loader_params['num_workers'])

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_loader_params['batch_size'],
                          drop_last=self.val_loader_params['drop_last'], shuffle=self.val_loader_params['shuffle'],
                          num_workers=self.val_loader_params['num_workers'])

    
class FaceEmojiFCModel(pl.LightningModule):
    """ Модель с полносвязной сетью для классификации на признаках ландмарок
    
    Параметры
    ---------
    dropout : float
      Значение вероятности для всех слоев Dropout полносвязной сети.
      По умолчанию 0.5
    learning_rate : float
      Скорость обучения. По умолчанию 0.01
    l2_regularization : float
      Регуляризация весов L2. По умолчанию 0.001
    adam_betas : tuple
      Коэффициенты оптимизатора Adam. По умолчанию (0.9, 0.999)
    plot_epoch_loss : bool
      Флаг печати графиков ошибок в конце каждой эпохи во время обучения. По умолчанию True.
    seed : int
      Фиксация генератора случайных чисел.
    """
    
    def __init__(self, dropout=0.5, learning_rate=0.01, l2_regularization=1e-3, adam_betas=(0.9, 0.999), plot_epoch_loss=True,
                 seed=None, ):

        super().__init__()

        if seed is not None:
            set_seed(seed)

        self.learning_rate = learning_rate
        self.l2_regularization = l2_regularization
        self.adam_betas = adam_betas

        self._input_size = 262

        self._model = nn.Sequential(
            nn.Linear(self._input_size, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 7),
        )

        # Словарь для хранения значения ошибок на стадии обучения и валидации
        # Для значений типа train добавляем значение np.nan, так как при первом запуске модель вначале осуществляет
        # шаг валидации без обучения и добавляет значения в списки типа val. Это будет считаться эпохой №0.
        self.train_history = {
            'train_loss': [np.nan],
            'train_accuracy': [np.nan],
            'val_loss': [],
            'val_accuracy': [],
        }
        self.plot_epoch_loss = plot_epoch_loss

        self.save_hyperparameters()

    def forward(self, x_in):
        x_out = self._model(x_in)
        return x_out

    def configure_optimizers(self):
        """Конфигурация оптимизатора и планировщика скорости обучения"""
        optimizer = optim.AdamW(self.parameters(), betas=self.adam_betas, lr=self.learning_rate,
                                weight_decay=self.l2_regularization)
        sheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                  T_0=20,
                                                                  eta_min=1e-4)

        return [optimizer], [sheduler]

    def _share_step(self, batch, batch_idx, mode='train'):
        """ Общий шаг для обучения, валидации и теста

        Параметры
        ---------
        batch : dict
          Батч-словарь в следующем формате:
          {
          'landmark_features': 262 признака ландмарок,
          'target': метка класса,
          }
        batch_idx : int
          Номер батча
        mode : str
          Режим. Используется только для префикса названий ошибок в логе.
          По умолчанию train
        """

        # Получаем предсказаннst эмоции для батча
        pred_logit_emoji = self(batch['landmark_features'])
        target_emoji = batch['target']

        # Считаем ошибку cross entropy и логируем ее
        loss = F.cross_entropy(pred_logit_emoji, target_emoji)
        self.log(f'{mode}_loss', loss, prog_bar=True)

        pred_target_emoji = torch.argmax(pred_logit_emoji.cpu().detach(), axis=1).numpy()

        accuracy = accuracy_score(pred_target_emoji, target_emoji.cpu().detach().numpy())
        self.log(f'{mode}_accuracy', accuracy, prog_bar=True)

        return {'loss': loss, 'accuracy': accuracy}

    def training_step(self, batch, batch_idx):
        """Шаг обучения"""
        return self._share_step(batch, batch_idx, mode='train')

    def training_epoch_end(self, outputs):
        """Действия после окончания каждой эпохи обучения

        Параметры
        ---------
        outputs : list
          Список словарей. Каждый словарь - результат функции self._share_step для определенного батча на шаге обучения
        """

        # Считаем средние ошибки loss и rmse_loss по эпохе
        avg_train_loss = torch.tensor([x['loss'] for x in outputs]).detach().mean()
        avg_train_accuracy = torch.tensor([x['accuracy'] for x in outputs]).detach().mean()

        # Добавляем средние ошибки в словарь статистики обучения, используется для построение графиков
        self.train_history['train_loss'].append(avg_train_loss.numpy().item())
        self.train_history['train_accuracy'].append(avg_train_accuracy.numpy().item())

        # Если включено отображение графика обучения в конце эпохи, то рисуем графики
        if self.plot_epoch_loss:
            self.plot_history_loss()

    def validation_step(self, batch, batch_idx):
        """ Шаг валидации """
        return self._share_step(batch, batch_idx, mode='val')

    def validation_epoch_end(self, outputs):
        """Действия после окончания каждой эпохи валидации

        Параметры
        ---------
        outputs : list
          Список словарей.
          Каждый словарь - результат функции self._share_step для определенного батча на шаге валидации
        """

        # Считаем средние ошибки loss и rmse_loss по эпохе
        avg_val_loss = torch.tensor([x['loss'] for x in outputs]).detach().mean()
        avg_val_accuracy = torch.tensor([x['accuracy'] for x in outputs]).detach().mean()
        # Логируем ошибку валидации
        self.log(f'val_loss', avg_val_loss, prog_bar=True)

        # Добавляем средние ошибки в словарь статистики обучения, используется для построение графиков
        self.train_history['val_loss'].append(avg_val_loss.numpy().item())
        self.train_history['val_accuracy'].append(avg_val_accuracy.numpy().item())

        # Если включено отображение графика обучения в конце эпохи, то рисуем графики
        if self.plot_epoch_loss:
            self.plot_history_loss()

    def plot_history_loss(self, clear_output=True):
        """ Функция построения графика обучения в конце эпохи
        """

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        axes[0].plot(np.arange(0, len(self.train_history['train_loss'])),
                     self.train_history['train_loss'],
                     label="train_loss")
        axes[0].scatter(np.arange(0, len(self.train_history['train_loss'])),
                        self.train_history['train_loss'])
        axes[0].plot(np.arange(0, len(self.train_history['val_loss'])),
                     self.train_history['val_loss'],
                     label="val_loss")
        axes[0].scatter(np.arange(0, len(self.train_history['val_loss'])),
                        self.train_history['val_loss'])
        axes[0].legend(loc='best')
        axes[0].set_xlabel("epochs")
        axes[0].set_ylabel("loss")

        val_loss_epoch_min = np.argmin(self.train_history['val_loss'])
        val_loss_min = self.train_history['val_loss'][val_loss_epoch_min]
        val_loss_min = round(val_loss_min, 3) if not np.isnan(val_loss_min) else val_loss_min
        title_min_vals = f'\nValidation minimum {val_loss_min} on epoch {val_loss_epoch_min}'

        axes[0].set_title('MODEL LOSS: Cross Entropy'+title_min_vals)
        axes[0].grid()

        axes[1].plot(np.arange(0, len(self.train_history['train_accuracy'])),
                     self.train_history['train_accuracy'], label="train_accuracy")
        axes[1].scatter(np.arange(0, len(self.train_history['train_accuracy'])),
                        self.train_history['train_accuracy'])
        axes[1].plot(np.arange(0, len(self.train_history['val_accuracy'])),
                     self.train_history['val_accuracy'], label="val_accuracy")
        axes[1].scatter(np.arange(0, len(self.train_history['val_accuracy'])),
                        self.train_history['val_accuracy'])
        axes[1].legend(loc='best')
        axes[1].set_xlabel("epochs")
        axes[1].set_ylabel("accuracy")

        val_loss_epoch_max = np.argmax(self.train_history['val_accuracy'])
        val_loss_max = self.train_history['val_accuracy'][val_loss_epoch_max]
        val_loss_max = round(val_loss_max, 3) if not np.isnan(val_loss_min) else val_loss_max
        title_max_vals = f'\nValidation maximum {val_loss_max} on epoch {val_loss_epoch_max}'

        axes[1].set_title('MONITORING LOSS: ACCURACY'+title_max_vals)
        axes[1].grid()

        plt.show()
        if clear_output:
            display.clear_output(wait=True)
            
            
class FaceEmojiTransferModel(pl.LightningModule):
    """ Модель для трансферного обучения с извлечением признаков из изображения

    Параметры
    ---------
    model_name : str
      Название модели. Допустимые значения см. в документации TransferNet
      По умолчанию resnet18.
    output_dims : list
      Структура полносвязной сети на выходе экстратора признаков.
      Длина списка - количество полносвязных слоев.
      Значения списка - кол-во нейронов в каждом полносвязном слое.
      Пример: [1024, 512, 256, 1] - четыре полносвязных слоя.
              Первый слой с 1024 нейронами, второй слой с 512 нейронами,
              третий слой с 256 нейронами и последний слой с 7 нейронами.
      По умолчанию [128, 256, 7].
      !!! Последний слой должен содержать только 7 нейронов.
    full_trainable : bool
      Сделать ли модель полностью обучаемой. По умолчанию False. Если False, то для обучения доступна только
      полносвязная сеть и около 10% окончания экстрактора признаков
    dropout : float
      Значение вероятности для всех слоев Dropout полносвязной сети.
      По умолчанию 0.5
    learning_rate : float
      Скорость обучения. По умолчанию 0.01
    l2_regularization : float
      Регуляризация весов L2. По умолчанию 0.001
    adam_betas : tuple
      Коэффициенты оптимизатора Adam. По умолчанию (0.9, 0.999)
    pretrained : bool
      Если True, то будут загружена предобученная модель на ImageNet.
      По умолчанию True.
      !!! ВАЖНО: Класс не тестировался на не предобученных моделях.
          Структура модели при значении pretrained=False может отличаться, поведение класса может быть некорректным.
    plot_epoch_loss : bool
      Флаг печати графиков ошибок в конце каждой эпохи во время обучения. По умолчанию True.
    seed : int
      Фиксация генератора случайных чисел.
    """

    def __init__(self, model_name='resnet18', output_dims=None, dropout=0.5, learning_rate=0.01, full_trainable=False,
                 l2_regularization=1e-3, adam_betas=(0.9, 0.999), pretrained=True, plot_epoch_loss=True, seed=None):
        super().__init__()

        if seed is not None:
            set_seed(seed)

        if not output_dims:
            output_dims = [128, 256, 7]

        self._check_output_dims(output_dims)

        # Здесь будет находиться модель для извлечения признаков
        self.feature_extractor = None
        # Здесь будет находиться модель полносвязной сети, вход - данные из экстрактора признаков
        self.head_fc = None
        # Определяем feature_extractor и head_fc
        self._make_model_layers(model_name=model_name, output_dims=output_dims, dropout=dropout,
                                full_trainable=full_trainable, pretrained=pretrained)

        self.learning_rate = learning_rate
        self.l2_regularization = l2_regularization
        self.adam_betas = adam_betas

        # Словарь для хранения значения ошибок на стадии обучения и валидации
        # Для значений типа train добавляем значение np.nan, так как при первом запуске модель вначале осуществляет
        # шаг валидации без обучения и добавляет значения в списки типа val. Это будет считаться эпохой №0.
        self.train_history = {
            'train_loss': [np.nan],
            'train_accuracy': [np.nan],
            'val_loss': [],
            'val_accuracy': [],
        }
        self.plot_epoch_loss = plot_epoch_loss

        self.save_hyperparameters()

    @staticmethod
    def _check_output_dims(output_dims):
        """Проверка корректности размера последнего полносвязного слоя"""
        assert output_dims[-1] == 7, f"Кол-во нейронов в последнем слое output_dims должен быть равен 7. Текущее значение: {output_dims}"

    @staticmethod
    def _get_model_fc_layer_name(model):
        """ Определяем название полносвязного слоя у модели

        Параметры
        ---------
        model : TransferNet object

        Результат
        ---------
        fc_name : str
          Название полносвязного слоя модели
        """

        # В зависимости от модели полносвязный слой может называться по-разному
        # Находим имя полносвязной сети
        if 'fc' in model.__dir__():
            fc_name = 'fc'
        elif 'head' in model.__dir__():
            fc_name = 'head'
        elif 'classifier' in model.__dir__():
            fc_name = 'classifier'
        else:
            raise Exception("В модели не найден полносвязный слой")

        return fc_name

    def _make_model_layers(self, model_name, output_dims, dropout, full_trainable, pretrained):
        """ Загрузка модели и настройка переменных self.feature_extractor и self.head_fc

        Параметры
        ----------
        model_name : str
        output_dims : list
        dropout : float
        pretrained : bool

        Более подробное описание параметров в описании класса.
        """

        model = TransferNet(output_dims=output_dims, dropout=dropout, full_trainable=full_trainable,
                            pretrained=pretrained)(name=model_name)

        """ В этом блоке выделим в отдельные переменные экстрактор признаков модели и полносвязную сеть.
        Так как может возникнуть задача добавления к признакам сверточной сети дополнительных признаков из вне.
        Так как меняется размер признаков, необходимо изменить размер входа полносвязной сети.
        В модели self.model размер входа полносвязной сети равен выходу стандартного экстрактора признаков
        """
        # В зависимости от модели полносвязный слой может называться по-разному
        # Находим имя полносвязной сети
        fc_name = self._get_model_fc_layer_name(model)

        # Копируем в отдельную переменную полносвязную сеть
        self.head_fc = getattr(model, fc_name)

        # А в исходной модели в полносвязный слой устанавливаем пустую последовательность.
        # Таким образом теперь self.model работает как экстрактор признаков из изображения
        setattr(model, fc_name, nn.Sequential())

        self.feature_extractor = model

    def forward(self, x_in):
        features = self.feature_extractor(x_in)
        x_out = self.head_fc(features).squeeze()
        return x_out

    def configure_optimizers(self):
        """Конфигурация оптимизатора и планировщика скорости обучения"""
        optimizer = optim.AdamW(self.parameters(), betas=self.adam_betas, lr=self.learning_rate,
                                weight_decay=self.l2_regularization)
        sheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                  T_0=20,
                                                                  eta_min=1e-4)

        return [optimizer], [sheduler]

    def _share_step(self, batch, batch_idx, mode='train'):
        """ Общий шаг для обучения, валидации и теста

        Параметры
        ---------
        batch : dict
          Батч-словарь в следующем формате:
          {
           'x_data': изображение,
           'target': метка класса,
          }
        batch_idx : int
          Номер батча
        mode : str
          Режим. Используется только для префикса названий ошибок в логе.
          По умолчанию train
        """

        # Получаем предсказаннst эмоции для батча
        pred_logit_emoji = self(batch['x_data'])
        target_emoji = batch['target']

        # Считаем ошибку cross entropy и логируем ее
        loss = F.cross_entropy(pred_logit_emoji, target_emoji)
        self.log(f'{mode}_loss', loss, prog_bar=True)

        pred_target_emoji = torch.argmax(pred_logit_emoji.cpu().detach(), axis=1).numpy()

        accuracy = accuracy_score(pred_target_emoji, target_emoji.cpu().detach().numpy())
        self.log(f'{mode}_accuracy', accuracy, prog_bar=True)

        return {'loss': loss, 'accuracy': accuracy}

    def training_step(self, batch, batch_idx):
        """Шаг обучения"""
        return self._share_step(batch, batch_idx, mode='train')

    def training_epoch_end(self, outputs):
        """Действия после окончания каждой эпохи обучения

        Параметры
        ---------
        outputs : list
          Список словарей. Каждый словарь - результат функции self._share_step для определенного батча на шаге обучения
        """

        # Считаем средние ошибки loss и rmse_loss по эпохе
        avg_train_loss = torch.tensor([x['loss'] for x in outputs]).detach().mean()
        avg_train_accuracy = torch.tensor([x['accuracy'] for x in outputs]).detach().mean()

        # Добавляем средние ошибки в словарь статистики обучения, используется для построение графиков
        self.train_history['train_loss'].append(avg_train_loss.numpy().item())
        self.train_history['train_accuracy'].append(avg_train_accuracy.numpy().item())

        # Если включено отображение графика обучения в конце эпохи, то рисуем графики
        if self.plot_epoch_loss:
            self.plot_history_loss()

    def validation_step(self, batch, batch_idx):
        """ Шаг валидации """
        return self._share_step(batch, batch_idx, mode='val')

    def validation_epoch_end(self, outputs):
        """Действия после окончания каждой эпохи валидации

        Параметры
        ---------
        outputs : list
          Список словарей.
          Каждый словарь - результат функции self._share_step для определенного батча на шаге валидации
        """

        # Считаем средние ошибки loss и rmse_loss по эпохе
        avg_val_loss = torch.tensor([x['loss'] for x in outputs]).detach().mean()
        avg_val_accuracy = torch.tensor([x['accuracy'] for x in outputs]).detach().mean()
        # Логируем ошибку валидации
        self.log(f'val_loss', avg_val_loss, prog_bar=True)

        # Добавляем средние ошибки в словарь статистики обучения, используется для построение графиков
        self.train_history['val_loss'].append(avg_val_loss.numpy().item())
        self.train_history['val_accuracy'].append(avg_val_accuracy.numpy().item())

        # Если включено отображение графика обучения в конце эпохи, то рисуем графики
        if self.plot_epoch_loss:
            self.plot_history_loss()

    def plot_history_loss(self, clear_output=True):
        """ Функция построения графика обучения в конце эпохи """

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        axes[0].plot(np.arange(0, len(self.train_history['train_loss'])),
                     self.train_history['train_loss'],
                     label="train_loss")
        axes[0].scatter(np.arange(0, len(self.train_history['train_loss'])),
                        self.train_history['train_loss'])
        axes[0].plot(np.arange(0, len(self.train_history['val_loss'])),
                     self.train_history['val_loss'],
                     label="val_loss")
        axes[0].scatter(np.arange(0, len(self.train_history['val_loss'])),
                        self.train_history['val_loss'])
        axes[0].legend(loc='best')
        axes[0].set_xlabel("epochs")
        axes[0].set_ylabel("loss")

        val_loss_epoch_min = np.argmin(self.train_history['val_loss'])
        val_loss_min = self.train_history['val_loss'][val_loss_epoch_min]
        val_loss_min = round(val_loss_min, 3) if not np.isnan(val_loss_min) else val_loss_min
        title_min_vals = f'\nValidation minimum {val_loss_min} on epoch {val_loss_epoch_min}'

        axes[0].set_title('MODEL LOSS: Cross Entropy'+title_min_vals)
        axes[0].grid()

        axes[1].plot(np.arange(0, len(self.train_history['train_accuracy'])),
                     self.train_history['train_accuracy'], label="train_accuracy")
        axes[1].scatter(np.arange(0, len(self.train_history['train_accuracy'])),
                        self.train_history['train_accuracy'])
        axes[1].plot(np.arange(0, len(self.train_history['val_accuracy'])),
                     self.train_history['val_accuracy'], label="val_accuracy")
        axes[1].scatter(np.arange(0, len(self.train_history['val_accuracy'])),
                        self.train_history['val_accuracy'])
        axes[1].legend(loc='best')
        axes[1].set_xlabel("epochs")
        axes[1].set_ylabel("accuracy")

        val_loss_epoch_max = np.argmax(self.train_history['val_accuracy'])
        val_loss_max = self.train_history['val_accuracy'][val_loss_epoch_max]
        val_loss_max = round(val_loss_max, 3) if not np.isnan(val_loss_min) else val_loss_max
        title_max_vals = f'\nValidation maximum {val_loss_max} on epoch {val_loss_epoch_max}'

        axes[1].set_title('MONITORING LOSS: ACCURACY'+title_max_vals)
        axes[1].grid()

        plt.show()
        if clear_output:
            display.clear_output(wait=True)
            
            
class FaceEmojiLandmarksModel(FaceEmojiTransferModel):
    """ Модель для трансферного обучения с помощью признаков изображений и признаков ландмарок

    Параметры
    ---------
    model_name : str
      Название модели. Допустимые значения см. в документации TransferNet
      По умолчанию resnet18.
    output_dims : list
      Структура полносвязной сети на выходе экстратора признаков.
      Длина списка - количество полносвязных слоев.
      Значения списка - кол-во нейронов в каждом полносвязном слое.
      Пример: [1024, 512, 256, 7] - четыре полносвязных слоя.
              Первый слой с 1024 нейронами, второй слой с 512 нейронами,
              третий слой с 256 нейронами и последний слой с 7 нейронами.
      По умолчанию [128, 256, 7].
      !!! Последний слой должен содержать только 7 нейронов.
    full_trainable : bool
      Сделать ли модель полностью обучаемой. По умолчанию False. Если False, то для обучения доступна только
      полносвязная сеть и около 10% окончания экстрактора признаков
    dropout : float
      Значение вероятности для всех слоев Dropout полносвязной сети.
      По умолчанию 0.5
    learning_rate : float
      Скорость обучения. По умолчанию 0.01
    l2_regularization : float
      Регуляризация весов L2. По умолчанию 0.001
    adam_betas : tuple
      Коэффициенты оптимизатора Adam. По умолчанию (0.9, 0.999)
    pretrained : bool
      Если True, то будут загружена предобученная модель на ImageNet.
      По умолчанию True.
      !!! ВАЖНО: Класс не тестировался на не предобученных моделях.
          Структура модели при значении pretrained=False может отличаться, поведение класса может быть некорректным.
    plot_epoch_loss : bool
      Флаг печати графиков ошибок в конце каждой эпохи во время обучения. По умолчанию True.
    seed : int
      Фиксация генератора случайных чисел.
    """

    def __init__(self, model_name='resnet18', output_dims=None, dropout=0.5, learning_rate=0.01, full_trainable=False,
                 l2_regularization=1e-3, adam_betas=(0.9, 0.999), pretrained=True, plot_epoch_loss=True, seed=None):
        
        super().__init__(model_name=model_name, output_dims=output_dims, dropout=dropout, learning_rate=learning_rate, full_trainable=full_trainable,
                 l2_regularization=l2_regularization, adam_betas=adam_betas, pretrained=pretrained, plot_epoch_loss=plot_epoch_loss, seed=seed)
        
        
    def _make_model_layers(self, model_name, output_dims, dropout, full_trainable, pretrained):

        model = TransferNet(output_dims=output_dims, dropout=dropout, full_trainable=full_trainable,
                            pretrained=pretrained)(name=model_name)
        
        landmarks_feature_count = 262

        # В зависимости от модели полносвязный слой может называться по-разному
        # Находим имя полносвязной сети
        fc_name = self._get_model_fc_layer_name(model)

        # Когда нашли имя полносвязной сети, получаем параметры размера входа и выхода первого линейного слоя
        # Линейный слой может быть либо на первом месте, либо на втором. Перед ним может быть блок Dropout
        first_head_layer = getattr(model, fc_name)[0]
        if getattr(first_head_layer, 'in_features', None) is not None:
            head_first_linear_index = 0
        else:
            head_first_linear_index = 1

        in_features = getattr(model, fc_name)[head_first_linear_index].in_features
        out_features = getattr(model, fc_name)[head_first_linear_index].out_features

        # К текущему размеру входа прибавляем длину дополнительных параметров
        in_features += landmarks_feature_count

        # Копируем в отдельную переменную полносвязную сеть
        self.head_fc = getattr(model, fc_name)
        # И заменяем у нее первый линейный слой на такой же слой с увеличенным размером входа
        self.head_fc[head_first_linear_index] = nn.Linear(in_features=in_features, out_features=out_features)

        # А в исходной модели в полносвязный слой устанавливаем слой Identity.
        # Таким образом теперь self.model работает как экстрактор признаков из изображения
        setattr(model, fc_name, nn.Identity())
        self.feature_extractor = model

    def forward(self, x_in, landmarks):
        # Извлекаем признаки из изображения
        image_features = self.feature_extractor(x_in)
        # Соединяем признаки изображения с дополнительными признаками
        all_features = torch.hstack([image_features, landmarks])
        # Получаем выход полносвязной сети
        x_out = self.head_fc(all_features).squeeze()
        return x_out
    
    def _share_step(self, batch, batch_idx, mode='train'):
        """ Общий шаг для обучения, валидации и теста

        Параметры
        ---------
        batch : dict
          Батч-словарь в следующем формате:
          {
           'x_data': изображение,
           'landmark_features': 262 признака ландмарок,
           'target': метка класса,
          }
        batch_idx : int
          Номер батча
        mode : str
          Режим. Используется только для префикса названий ошибок в логе.
          По умолчанию train
        """

        # Получаем предсказаннst эмоции для батча
        pred_logit_emoji = self(batch['x_data'], batch['landmark_features'])
        target_emoji = batch['target']

        # Считаем ошибку cross entropy и логируем ее
        loss = F.cross_entropy(pred_logit_emoji, target_emoji)
        self.log(f'{mode}_loss', loss, prog_bar=True)

        pred_target_emoji = torch.argmax(pred_logit_emoji.cpu().detach(), axis=1).numpy()

        accuracy = accuracy_score(pred_target_emoji, target_emoji.cpu().detach().numpy())
        self.log(f'{mode}_accuracy', accuracy, prog_bar=True)

        return {'loss': loss, 'accuracy': accuracy}