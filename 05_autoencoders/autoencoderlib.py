import random

from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import pytorch_lightning as pl
from sklearn.manifold import TSNE
import plotly.express as px
import matplotlib.pyplot as plt
from IPython import display


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
        

class FCModel(nn.Module):
    """Класс для генерации полносвязной модели
    
    Параметры
    ---------
    input_size : int
      Размер входного вектора энкодера
    model_dims : list
      Структура полносвязной сети на выходе модели.
      Длина списка - количество полносвязных слоев.
      Значения списка - кол-во нейронов в каждом полносвязном слое.
      Пример: [1024, 512, 256] - три полносвязных слоя.
              Первый слой с 1024 нейронами, второй слой с 512 нейронами,
              третий слой с 256 нейронами
      По умолчанию [256, 128, 64].
    dropout : bool
      Если True, по после каждого полносвязного слоя будеет добавлен Dropout.
      Кроме последнего слоя.
    last_activation : str
      Активация на последнем слое. По умолчанию relu.
      Может принимать значения relu, tanh, sigmoid или None
    """
    
    def __init__(self, input_size, model_dims=None, dropout=False, last_activation='relu'):
        
        super().__init__()
        
        if not model_dims:
            model_dims = [256, 128, 64]
            
        self._fc_model = self._make_model_layers(input_size=input_size, model_dims=model_dims, 
                                                 last_activation=last_activation, dropout=dropout)
        
    @staticmethod  
    def _make_model_layers(input_size, model_dims, last_activation='relu', dropout=False):
        """ Сборка модели
        """
        
        last_activation = str(last_activation).lower().strip() if last_activation is not None else last_activation
        assert last_activation is None or last_activation in ('relu', 'sigmoid', 'tanh')
        
        fc_model = nn.Sequential()
        input_dim = input_size
        
        len_model_dims = len(model_dims)
        
        for idx, output_dim in enumerate(model_dims):
            
            fc_model.append(nn.Linear(input_dim, output_dim))
            
            if idx < len_model_dims-1:
                if dropout:
                    fc_model.append(nn.Dropout())
                fc_model.append(nn.ReLU())

            elif last_activation == 'sigmoid':
                fc_model.append(nn.Sigmoid())
                
            elif last_activation == 'relu':
                fc_model.append(nn.ReLU())
                
            elif last_activation == 'tanh':
                fc_model.append(nn.Tanh())
                
            input_dim = output_dim
            
        return fc_model
    
    def forward(self, x):
        return self._fc_model(x)
        
        
class FCAutoEncoder(pl.LightningModule):
    """ Модель для датасета Cockatiel VS Cockatoo
    
    Параметры
    ---------
    image_size : int
      Размер изображения
    encoder_dims : list
      Структура полносвязной сети кодировщика.
      Длина списка - количество полносвязных слоев.
      Значения списка - кол-во нейронов в каждом полносвязном слое.
      Пример: [1024, 512, 256] - три полносвязных слоя.
              Первый слой с 1024 нейронами, второй слой с 512 нейронами,
              третий слой с 256 нейронами
    decoder_dims : list
      Структура полносвязной сети декодера.
      Параметр аналогичен encoder_dims
    latent_size : int
      Размер латентного слоя. Латентный слой будет включен
      в последний слой кодировщика.
    channel_count : int
      Количество каналов в изображении
    loss_fn : str
      Функция ошибок для обучения автокодировщика.
      Может принимать значение 'mse'.
      В будущем будут реализованы другие функции ошибок.
    last_activation : str
      Какую активацию использовать на выходе декодера.
      Подбирается исходя из типа выходных данных.
      Может принимаеть значение sigmoid, relu, tanh или None/
      По умолчанию None.
    noised : True
      Если True, то модель будет обучаться на 
      аугментированных данных. Но проверять ошибку
      на изображении без аугментации.
      Опция доступна только для кастомного датасета.
      Изображение с аугментацией будет браться из словаря
      батча по ключу 'img_transform'.
      А ошибка будет считаться для img_source.
      !!! Требуется корректная настройка загрузчика данных
      для обучения
    learning_rate : float
      Скорость обучения модели. По умолчанию 0.001.
    l2_regularization : float
      Размер L2-регуляризации, по умолчанию 0.01
    adam_betas : tuple
      Коэффициенты для оптимизатора Adam.
      По умолчанию (0.99, 0.999).
    plot_epoch_loss : bool
      Если True, то после каждой эпохи обучения и валидации будет выводиться график с результатами обучения
    seed :
    """
    
    def __init__(self, image_size, encoder_dims, decoder_dims, latent_size, channel_count=3, loss_fn='mse',
                 last_activation=None, noised=False, learning_rate=0.001, l2_regularization=None,
                 adam_betas=(0.99, 0.999), plot_epoch_loss=True, seed=None):
        
        super().__init__()
        
        loss_fn = str(loss_fn).lower().strip()
        
        # Поддерживаемые функции ошибки
        assert loss_fn in ('mse')
        
        self._input_size = image_size*image_size*channel_count
        
        # Латентный слой будет частью энкодера
        encoder_dims = list(encoder_dims) + [latent_size]
        self.encoder = FCModel(input_size=self._input_size, model_dims=encoder_dims, last_activation='relu')
        self.decoder = FCModel(input_size=latent_size, model_dims=decoder_dims, last_activation=last_activation)
        
        self.plot_epoch_loss = plot_epoch_loss
        self.learning_rate = learning_rate
        self.l2_regularization = l2_regularization
        self.adam_betas = adam_betas
        self.loss_fn = loss_fn
        self.noised = noised
        
        # Словарь для хранения значения ошибок на стадии обучения и валидации
        # Для значений типа train добавляем значение np.nan, так как при первом запуске модель вначале осуществляет
        # шаг валидации без обучения и добавляет значения в списки типа val. Это будет считаться эпохой №0.
        self.train_history = {
            'train_loss': [np.nan],
            'val_loss': [],
        }
        
    def forward(self, x):
        x = x.view(x.shape[0], self._input_size)
        latent = self.encoder(x)
        output = self.decoder(latent)
        return output
    
    def encode(self, x):
        x = x.view(x.shape[0], self._input_size)
        with torch.no_grad():
            return self.encoder(x)
        
    def decode(self, x):
        x = x.view(x.shape[0], self._input_size)
        with torch.no_grad():
            return self.decoder(x)
        
    def configure_optimizers(self):
        """Конфигурация оптимизатора и планировщика скорости обучения"""
        optimizer = optim.AdamW(self.parameters(), betas=self.adam_betas, lr=self.learning_rate,
                                weight_decay=self.l2_regularization)
        sheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, eta_min=1e-4)

        return [optimizer], [sheduler]

    def _share_step(self, batch, batch_idx, mode='train'):
        """ Общий шаг для обучения, валидации и теста
        Параметры
        ---------
        batch : dict
          Батч-словарь в следующем формате:
          {
          'img_source': список тензоров изображения c базовой трансформацией размером N x C x H x W,
          'img_transform': список тензоров изображения c аугментацией размером N x C x H x W,
          'target': список целевых значений
          }
        batch_idx : int
          Номер батча
        mode : str
          Режим. Используется только для префикса названий ошибок в логе.
          По умолчанию train
        """

        if self.noised and mode == 'train':
            img_batch = batch['img_transform']
        else:
            img_batch = batch['img_source']
        
        y_true = img_batch.view(img_batch.shape[0], -1)
        y_pred = self(img_batch)
        
        if self.loss_fn == 'mse':
            loss = F.mse_loss(y_pred, y_true)
            
        self.log(f'{mode}_loss', loss, prog_bar=True)

        return {'loss': loss}

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

        # Считаем средние ошибки lossпо эпохе
        avg_train_loss = torch.tensor([x['loss'] for x in outputs]).detach().mean()

        # Добавляем средние ошибки в словарь статистики обучения, используется для построение графиков
        self.train_history['train_loss'].append(avg_train_loss.numpy().item())

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

        # Считаем средние ошибки loss по эпохе
        avg_val_loss = torch.tensor([x['loss'] for x in outputs]).detach().mean()
        
        # Логируем ошибку валидации
        self.log(f'val_loss', avg_val_loss, prog_bar=True)

        # Добавляем средние ошибки в словарь статистики обучения, используется для построение графиков
        self.train_history['val_loss'].append(avg_val_loss.numpy().item())

        # Если включено отображение графика обучения в конце эпохи, то рисуем графики
        if self.plot_epoch_loss:
            self.plot_history_loss()

    def plot_history_loss(self, clear_output=True):
        """ Функция построения графика обучения в конце эпохи
        
        Параметры
        clear_output : bool
          Если True, то после каждой эпохи график будет обновляться,
          а не печататься новый.
        """

        plt.figure(figsize=(8, 5))

        plt.plot(np.arange(0, len(self.train_history['train_loss'])),
                     self.train_history['train_loss'],
                     label="train_loss")
        plt.scatter(np.arange(0, len(self.train_history['train_loss'])),
                     self.train_history['train_loss'])
        plt.plot(np.arange(0, len(self.train_history['val_loss'])),
                     self.train_history['val_loss'],
                     label="val_loss")
        plt.scatter(np.arange(0, len(self.train_history['val_loss'])),
                     self.train_history['val_loss'])
        plt.legend(loc='best')
        plt.xlabel("epochs")
        plt.ylabel("loss")
        if len(self.train_history['val_loss'])> 1:
            val_loss_epoch_min = np.argmin(self.train_history['val_loss'][1:]) + 1
            val_loss_min = self.train_history['val_loss'][val_loss_epoch_min]
            val_loss_min = round(val_loss_min, 3) if not np.isnan(val_loss_min) else val_loss_min
            title_min_vals = f'\nValidation minimum {val_loss_min} on epoch {val_loss_epoch_min}'
        else:
            title_min_vals = ""
        plt.title(f'MODEL LOSS: {self.loss_fn.upper()}'+title_min_vals)

        plt.grid()
        plt.show()
        
        if clear_output:
                display.clear_output(wait=True)
                
        
class CNNEncoder(nn.Module):
    """ Класс энкодера на основе CNN
    
    Класс расчитан на входное ихображение размером Cx48x48, где С - число каналов. 
    Число каналов можно задать при инициализации класса.
    
    Параметры
    ---------
    channel_count : int
      Число каналов изображения
    latent_size : int
      Размер латентного слоя
    """
    
    def __init__(self, channel_count, latent_size):
        super().__init__()
        
        self.encoder_cnn = nn.Sequential(
            
            # Сюда приходит размер изображения channel_countx48x48
            nn.Conv2d(channel_count, 64, kernel_size=(3,3), padding=1, stride=1),
            nn.ReLU(),  
            nn.Conv2d(64, 64, kernel_size=(3,3), padding=1, stride=1),
            nn.ReLU(),
            
            # Сюда приходит размер изображения 64x48x48
            nn.Conv2d(64, 128, kernel_size=(3,3), padding=1, stride=2),
            nn.ReLU(),    
            nn.Conv2d(128, 128, kernel_size=(3,3), padding=1, stride=1),
            nn.ReLU(), 
            nn.Conv2d(128, 128, kernel_size=(3,3), padding=1, stride=1),
            nn.ReLU(),
            
            # Сюда приходит размер изображения 128x24x24
            nn.Conv2d(128, 256, kernel_size=(3,3), padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3,3), padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3,3), padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3,3), padding=1, stride=1),
            nn.ReLU(),            
            nn.Conv2d(256, 256, kernel_size=(3,3), padding=1, stride=1),
            nn.ReLU(),
            
            # Сюда приходит размер изображения 256x12x12
            nn.Conv2d(256, 512, kernel_size=(3,3), padding=1, stride=2),
            nn.ReLU(),         
            nn.Conv2d(512, 512, kernel_size=(3,3), padding=1, stride=1),
            nn.ReLU(),             
            nn.Conv2d(512, 512, kernel_size=(3,3), padding=1, stride=1),
            nn.ReLU(),
            
            # Сюда приходит размер изображения 512x6x6
            nn.Conv2d(512, 1024, kernel_size=(3,3), padding=1, stride=2),
            nn.ReLU(),         
            nn.Conv2d(1024, 1024, kernel_size=(3,3), padding=1, stride=1),
            nn.ReLU(),             
            nn.Conv2d(1024, 1024, kernel_size=(3,3), padding=1, stride=1),
            nn.ReLU(),
            # На выходе размер изображения 1024x3x3
        )

        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        
        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(1024*3*3, 2048),
            nn.ReLU(),
            nn.Linear(2048, latent_size)
        )
        
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x
    
class CNNDecoder(nn.Module):
    """ Класс энкодера на основе CNN
    
    Декодер выдает изображение размера Cx48x48, где С - число каналов. 
    Число каналов можно задать при инициализации класса.
        
    Параметры
    ---------
    channel_count : int
      Число каналов изображения
    latent_size : int
      Размер латентного слоя
    """
    
    def __init__(self, channel_count, latent_size):
        
        super().__init__()
        
        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_size, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 1024*3*3),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(1024, 3, 3))

        self.decoder_conv = nn.Sequential(
            
            # Сюда приходит размер изображения 1024x3x3
            nn.ConvTranspose2d(1024, 1024, 1, stride=1, padding=0, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 1024, 1, stride=1, padding=0, output_padding=0),
            nn.ReLU(),            
            nn.ConvTranspose2d(1024, 512, 3, stride=1, padding=0, output_padding=0),
            nn.ReLU(),            
            
            # Сюда приходит размер изображения 512x5x5
            nn.ConvTranspose2d(512, 512, 1, stride=1, padding=0, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, 1, stride=1, padding=0, output_padding=0),
            nn.ReLU(),            
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=0, output_padding=0),
            nn.ReLU(),
            
            # Сюда приходит размер изображения 256x11x11
            nn.ConvTranspose2d(256, 256, 1, stride=1, padding=0, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 1, stride=1, padding=0, output_padding=0),
            nn.ReLU(),   
            nn.ConvTranspose2d(256, 256, 1, stride=1, padding=0, output_padding=0),
            nn.ReLU(),   
            nn.ConvTranspose2d(256, 256, 1, stride=1, padding=0, output_padding=0),
            nn.ReLU(),             
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=0, output_padding=0),
            nn.ReLU(),
            
            # Сюда приходит размер изображения 128x23x23           
            nn.ConvTranspose2d(128, 128, 1, stride=1, padding=0, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(128, channel_count, 3, stride=2, padding=0, output_padding=1),
            # На выходе размер изображения channel_countx48x48
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        return x
        

class CNNAutoEncoder(pl.LightningModule):
    """ Модель для датасета Cockatiel VS Cockatoo
    
    Параметры
    ---------
    latent_size : int
      Размер латентного слоя. Латентный слой будет включен
      в последний слой кодировщика.
    channel_count : int
      Количество каналов в изображении
    loss_fn : str
      Функция ошибок для обучения автокодировщика.
      Может принимать значение 'mse'.
      В будущем будут реализованы другие функции ошибок.
    noised : True
      Если True, то модель будет обучаться на 
      аугментированных данных. Но проверять ошибку
      на изображении без аугментации.
      Опция доступна только для кастомного датасета.
      Изображение с аугментацией будет браться из словаря
      батча по ключу 'img_transform'.
      А ошибка будет считаться для img_source.
      !!! Требуется корректная настройка загрузчика данных
      для обучения
    learning_rate : float
      Скорость обучения модели. По умолчанию 0.001.
    l2_regularization : float
      Размер L2-регуляризации, по умолчанию 0.01
    adam_betas : tuple
      Коэффициенты для оптимизатора Adam.
      По умолчанию (0.99, 0.999).
    plot_epoch_loss : bool
      Если True, то после каждой эпохи обучения и валидации будет выводиться график с результатами обучения
    seed :
    """
    
    def __init__(self, latent_size, channel_count=3, loss_fn='mse', noised=False, learning_rate=0.001, 
                 l2_regularization=None, adam_betas=(0.99, 0.999), plot_epoch_loss=True, seed=None):
        
        super().__init__()
        
        loss_fn = str(loss_fn).lower().strip()
        
        # Поддерживаемые функции ошибки
        assert loss_fn in ('mse')
        
        self.encoder = CNNEncoder(latent_size=latent_size, channel_count=channel_count)
        self.decoder = CNNDecoder(latent_size=latent_size, channel_count=channel_count)
        
        self.plot_epoch_loss = plot_epoch_loss
        self.learning_rate = learning_rate
        self.l2_regularization = l2_regularization
        self.adam_betas = adam_betas
        self.loss_fn = loss_fn
        self.noised = noised
        
        # Словарь для хранения значения ошибок на стадии обучения и валидации
        # Для значений типа train добавляем значение np.nan, так как при первом запуске модель вначале осуществляет
        # шаг валидации без обучения и добавляет значения в списки типа val. Это будет считаться эпохой №0.
        self.train_history = {
            'train_loss': [np.nan],
            'val_loss': [],
        }
        
    def forward(self, x):
        latent = self.encoder(x)
        output = self.decoder(latent)
        return output
    
    def encode(self, x):
        with torch.no_grad():
            return self.encoder(x)
        
    def configure_optimizers(self):
        """Конфигурация оптимизатора и планировщика скорости обучения"""
        optimizer = optim.AdamW(self.parameters(), betas=self.adam_betas, lr=self.learning_rate,
                                weight_decay=self.l2_regularization)
        sheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, eta_min=1e-4)

        return [optimizer], [sheduler]

    def _share_step(self, batch, batch_idx, mode='train'):
        """ Общий шаг для обучения, валидации и теста
        Параметры
        ---------
        batch : dict
          Батч-словарь в следующем формате:
          {
          'img_source': список тензоров изображения c базовой трансформацией размером N x C x H x W,
          'img_transform': список тензоров изображения c аугментацией размером N x C x H x W,
          'target': список целевых значений
          }
        batch_idx : int
          Номер батча
        mode : str
          Режим. Используется только для префикса названий ошибок в логе.
          По умолчанию train
        """

        if self.noised and mode == 'train':
            img_batch = batch['img_transform']
        else:
            img_batch = batch['img_source']
        
        y_true = img_batch.view(img_batch.shape[0], -1)
        y_pred = self(img_batch).view(img_batch.shape[0], -1)
        
        if self.loss_fn == 'mse':
            loss = F.mse_loss(y_pred, y_true)
            
        self.log(f'{mode}_loss', loss, prog_bar=True)

        return {'loss': loss}

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

        # Считаем средние ошибки lossпо эпохе
        avg_train_loss = torch.tensor([x['loss'] for x in outputs]).detach().mean()

        # Добавляем средние ошибки в словарь статистики обучения, используется для построение графиков
        self.train_history['train_loss'].append(avg_train_loss.numpy().item())

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

        # Считаем средние ошибки loss по эпохе
        avg_val_loss = torch.tensor([x['loss'] for x in outputs]).detach().mean()
        
        # Логируем ошибку валидации
        self.log(f'val_loss', avg_val_loss, prog_bar=True)

        # Добавляем средние ошибки в словарь статистики обучения, используется для построение графиков
        self.train_history['val_loss'].append(avg_val_loss.numpy().item())

        # Если включено отображение графика обучения в конце эпохи, то рисуем графики
        if self.plot_epoch_loss:
            self.plot_history_loss()

    def plot_history_loss(self, clear_output=True):
        """ Функция построения графика обучения в конце эпохи
        
        Параметры
        clear_output : bool
          Если True, то после каждой эпохи график будет обновляться,
          а не печататься новый.
        """

        plt.figure(figsize=(8, 5))

        plt.plot(np.arange(0, len(self.train_history['train_loss'])),
                     self.train_history['train_loss'],
                     label="train_loss")
        plt.scatter(np.arange(0, len(self.train_history['train_loss'])),
                     self.train_history['train_loss'])
        plt.plot(np.arange(0, len(self.train_history['val_loss'])),
                     self.train_history['val_loss'],
                     label="val_loss")
        plt.scatter(np.arange(0, len(self.train_history['val_loss'])),
                     self.train_history['val_loss'])
        plt.legend(loc='best')
        plt.xlabel("epochs")
        plt.ylabel("loss")
        if len(self.train_history['val_loss'])> 1:
            val_loss_epoch_min = np.argmin(self.train_history['val_loss'][1:]) + 1
            val_loss_min = self.train_history['val_loss'][val_loss_epoch_min]
            val_loss_min = round(val_loss_min, 3) if not np.isnan(val_loss_min) else val_loss_min
            title_min_vals = f'\nValidation minimum {val_loss_min} on epoch {val_loss_epoch_min}'
        else:
            title_min_vals = ""
        plt.title(f'MODEL LOSS: {self.loss_fn.upper()}'+title_min_vals)

        plt.grid()
        plt.show()
        
        if clear_output:
            display.clear_output(wait=True)
  
  
def calc_conv2d_output_size(image_size=(224, 224), kernel_size=(3,3), padding=0, stride=1):
    
    w_output = np.floor((image_size[0] - kernel_size[1] + 2*padding) / stride + 1)
    h_output = np.floor((image_size[1] - kernel_size[0] + 2*padding) / stride + 1)
    
    return w_output, h_output


def calc_transpose_conv2d_output_size(image_size=(224, 224), kernel_size=(3,3), padding=0, padding_out=0, stride=2):
    
    assert padding_out < stride
    assert stride > 0
    assert kernel_size[0] == kernel_size[1]
    
    w_output = (image_size[0] - 1)*stride + kernel_size[0] + 2*padding + padding_out
    h_output = (image_size[1] - 1)*stride + kernel_size[0] + 2*padding + padding_out
    
    return w_output, h_output
    
    
def plot_latent_tsne(model, datamodule, perplexity=10, learning_rate=200, 
                     n_iter=1000, n_jobs=-1, verbose=0):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    datamodule.setup()
    dataloader = datamodule.val_dataloader()
    
    model.eval()
    model = model.to(device)
    
    X = None
    y = None
    
    print("Подготовка фичей изображений")
    for batch in tqdm(dataloader):
        
        img_source = batch['img_source'].to(device)
        label = batch['target']
            
        latent_code = model.encode(img_source)
        latent_code = latent_code.detach().cpu().numpy()
        
        if X is None:
            X = latent_code
        else:
            X = np.vstack([X, latent_code])
            
        if y is None:
            y = label
        else:
            y = np.hstack([y, label])
            
    assert len(y) == X.shape[0]
    
    embed = TSNE(
        n_components=2,
        perplexity=perplexity, 
        learning_rate=learning_rate,
        n_iter=n_iter,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    
    display.clear_output()

    # Преобразование X
    X_embedded = embed.fit_transform(X)

    # Вывод результатов
    print('New Shape of X: ', X_embedded.shape)
    print('Kullback-Leibler divergence after optimization: ', embed.kl_divergence_)
    print('No. of iterations: ', embed.n_iter_)
    #вывод('Embedding vectors: ', embed.embedding_)

        # Создание диаграммы разброса
    fig = px.scatter(None, x=X_embedded[:,0], y=X_embedded[:,1], 
                     labels={
                         "x": "Dimension 1",
                         "y": "Dimension 2",
                     },
                     opacity=1, color=y.astype(str))

    # Изменение цвета фона графика
    fig.update_layout(dict(plot_bgcolor = 'white'))

    # Обновление линий осей
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', 
                     zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey', 
                     showline=True, linewidth=1, linecolor='black')

    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', 
                     zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey', 
                     showline=True, linewidth=1, linecolor='black')

    # Установка названия рисунка
    fig.update_layout(title_text="t-SNE")

    # Обновление размера маркера
    fig.update_traces(marker=dict(size=3))

    fig.show()
    
    
def print_autoencoder_result(model, datamodule, image_size, channels_count=3, split='val', 
                             denormalize=True, std=None, mean=None):
    
    split = str(split).lower().strip()
    assert split in ('val', 'train')
    
    number_images = 6
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if split == 'train':
        backup_batch_size = datamodule.train_loader_params['batch_size']
        datamodule.train_loader_params['batch_size'] = 1
        datamodule.setup()
        dataloader = datamodule.train_dataloader()
    else:
        backup_batch_size = datamodule.val_loader_params['batch_size']
        datamodule.val_loader_params['batch_size'] = 1
        datamodule.setup()
        dataloader = datamodule.val_dataloader()
    
    if denormalize:
        assert std is not None
        assert mean is not None
        assert channels_count == len(std) and channels_count == len(mean)
        
    model.eval()
    model = model.to(device)
    
    if channels_count == 1:
        plt_params = {
            'cmap': 'gray', 
            'vmin': 0, 
            'vmax': 1
        }
    else:
        plt_params = {}
        
    for idx, batch in enumerate(dataloader):
        
        if idx > number_images:
            break

        img_source = batch['img_source'].to(device)
        img_transform = batch['img_transform'].to(device)
        img_source_pred = model(img_source)
        img_transform_pred = model(img_transform)

        if denormalize:

            if len(std) == 1:
                std = np.tile(std, img_source.shape[0])
            if len(mean) == 1:
                mean = np.tile(mean, img_source.shape[0])

            std = np.array(std).reshape((channels_count, -1))
            mean = np.array(mean).reshape((channels_count, -1))
            img_source = img_source.detach().cpu().view(channels_count, -1) * std + mean
            img_source_pred = img_source_pred.detach().cpu().view(channels_count, -1) * std + mean
            img_transform = img_transform.detach().cpu().view(channels_count, -1) * std + mean
            img_transform_pred = img_transform_pred.detach().cpu().view(channels_count, -1) * std + mean

        img_source = img_source.detach().cpu().view(channels_count, image_size, image_size).permute(1,2,0).numpy()
        img_source_pred = img_source_pred.detach().cpu().view(channels_count, image_size, image_size).permute(1,2,0).numpy()
        img_transform = img_transform.detach().cpu().view(channels_count, image_size, image_size).permute(1,2,0).numpy()
        img_transform_pred = img_transform_pred.detach().cpu().view(channels_count, 
                                                                    image_size, image_size).permute(1,2,0).numpy()
                                                                    
        img_source = np.clip(img_source, 0, 255)
        img_source_pred = np.clip(img_source_pred, 0, 255)  
        img_transform = np.clip(img_transform, 0, 255)  
        img_transform_pred = np.clip(img_transform_pred, 0, 255)          
        
        fig, axes = plt.subplots(1, 4, figsize=(7, 3))
        axes = axes.flatten()
        axes[0].set_title('Src')
        axes[0].imshow(img_source, **plt_params)
        axes[1].set_title('Rec Src')
        axes[1].imshow(img_source_pred, **plt_params)
        axes[2].set_title('Aug')
        axes[2].imshow(img_transform, **plt_params)
        axes[3].set_title('Rec Aug')
        axes[3].imshow(img_transform_pred, **plt_params)
        
        plt.show()
            
    if split == 'train':
        datamodule.train_loader_params['batch_size'] = backup_batch_size
    else:
        datamodule.val_loader_params['batch_size'] = backup_batch_size
        
    datamodule.setup()