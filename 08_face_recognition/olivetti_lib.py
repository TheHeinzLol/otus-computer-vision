import random

import matplotlib.pyplot as plt
from IPython import display
import numpy as np
import torchvision.transforms as T
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


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


class OlivettiDataSet(Dataset):
    """ Датасет Olivetti
    
    Параметры
    ---------
    faces_npy : str
      Путь к файлу с лицами Olivetti в формате npy
    target_npy : str
      путь к целевым меткам лиц Olivetti в формате npy
    test_size : float
      Размер тестовой части, от 0.0 до 1.0
    embedding_size : int
      Размер эмбеддинга изображений (число главных компонент для преобразования PCA)
    seed : int
      Фиксация генератора случайных чисел.
    """

    def __init__(self, faces_npy, target_npy, test_size=0.2, embedding_size=60, seed=None):
        
        super().__init__()
            
        self._data = np.load(faces_npy)
        self._target = np.load(target_npy)
        self._test_size = test_size
        self._embedding_size = embedding_size
        self._seed = seed
        
        X_train, X_val, y_train, y_val = train_test_split(self._data, self._target, test_size=self._test_size, 
                                                          stratify=self._target, random_state=seed)
        
        X_train_pca, X_val_pca = self._make_pca(X_train, X_val, n_components=embedding_size)
        
        train_size, val_size = len(X_train), len(X_val)
        
        self._lookup_dict = {
            'train': (X_train, X_train_pca, y_train, train_size),
            'val': (X_val, X_val_pca, y_val, val_size),
        }
        
        self._transforms = T.Compose([
            T.ToTensor(),
            T.ConvertImageDtype(torch.float32),
        ])
        
        # По умолчанию включаем режим обучения
        self.set_split('train')
        
    def _make_pca(self, X_train, X_val, n_components):
        """ Выполняет преобразование PCA
        
        Параметры
        ---------
        X_train : np.ndarray
        X_test : np.ndarray
        n_components : int
          Число главных компонент
          
        Результат
        ---------
        X_train_pca, X_test_pca
        """
        self._pca = PCA(n_components=n_components, whiten=True, random_state=self._seed)
        
        train_size = X_train.shape[0]
        val_size = X_val.shape[0]
        
        X_train_pca = self._pca.fit_transform(X_train.reshape([train_size, -1]))
        X_val_pca = self._pca.transform(X_val.reshape([val_size, -1]))
        
        return X_train_pca, X_val_pca

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
        self._target_x, self._target_x_pca, self._target_y, self._target_size = self._lookup_dict[split]

    def __len__(self):
        return self._target_size

    def __getitem__(self, index):
        """ Возвращает элемент датасета в формате:
        {
        'img': тензор исходного изображения размером H x W,
        'img_pca': тензор изображения с преобразованием PCA размером H x W,
        'label': метка изображения
        }
        """

        # Получаем строку датафрейма по его индексу
        image = self._target_x[index]
        image_pca = self._target_x_pca[index]
        label = self._target_y[index]

        # Словарь, который будем возвращать
        model_data = {
            'img': self._transforms(image),
            'embedding': torch.tensor(image_pca, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.int64),
        }
        
        return model_data


class OlivettiDataModule(pl.LightningDataModule):
    """ Загрузчик PyTorch Lighting для датасета Olivetti
    
    
    Параметры
    ---------
    faces_npy : str
      Путь к файлу с лицами Olivetti в формате npy
    target_npy : str
      путь к целевым меткам лиц Olivetti в формате npy
    test_size : float
      Размер тестовой части, от 0.0 до 1.0
    embedding_size : int
      Размер эмбеддинга изображений (число главных компонент для преобразования PCA)
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

    def __init__(self, faces_npy, target_npy, test_size=0.2, embedding_size=60, train_loader_params=None, 
                 val_loader_params=None, seed=2022):

        super().__init__()
        
        assert isinstance(seed, int)

        self._faces_npy = faces_npy
        self._target_npy = target_npy
        self._test_size = test_size
        self._embedding_size = embedding_size
        self._seed = seed

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

    def setup(self, stage=None):

        self.train_dataset = OlivettiDataSet(faces_npy=self._faces_npy, target_npy=self._target_npy, 
                                             test_size=self._test_size, embedding_size=self._embedding_size, seed=self._seed)
        self.train_dataset.set_split('train')

        self.val_dataset = OlivettiDataSet(faces_npy=self._faces_npy, target_npy=self._target_npy, 
                                           test_size=self._test_size, embedding_size=self._embedding_size, seed=self._seed)
        self.val_dataset.set_split('val')


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_loader_params['batch_size'],
                          shuffle=self.train_loader_params['shuffle'], drop_last=self.train_loader_params['drop_last'],
                          num_workers=self.train_loader_params['num_workers'])

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_loader_params['batch_size'],
                          drop_last=self.val_loader_params['drop_last'], shuffle=self.val_loader_params['shuffle'],
                          num_workers=self.val_loader_params['num_workers'])


class OlivettiModel(pl.LightningModule):
    """ Реализация сети классификатора эмбеддингов изображений Olivetti 
    
    Параметры
    ---------
    input_size : int
      Размер входа
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
    def __init__(self, input_size, dropout=0.5, learning_rate=0.01, l2_regularization=1e-3, adam_betas=(0.9, 0.999), 
                 plot_epoch_loss=True, seed=None):
        
        super().__init__()
        
        if seed is not None:
            set_seed(seed)        
        
        self.learning_rate = learning_rate
        self.l2_regularization = l2_regularization
        self.adam_betas = adam_betas
        
        self._model = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 40),
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
        
    def configure_optimizers(self):
        """Конфигурация оптимизатора и планировщика скорости обучения"""
        optimizer = optim.AdamW(self.parameters(), betas=self.adam_betas, lr=self.learning_rate,
                                    weight_decay=self.l2_regularization)
        sheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, eta_min=1e-4)
    
        return [optimizer], [sheduler]

    def forward(self, input_data):
        """ Прямой проход модели
        
        Параметры
        ---------
        input_data : torch.tensor
          Тензор размером N x W, где N - размер батча, W - размер эмбеддинга
        
        Результат
        ---------
        forward_result : torch.tensor
          Тензор размером N x 40, где N - размер батча  
        """
        forward_result = self._model(input_data)
        
        return forward_result
    
    def _share_step(self, batch, batch_idx, mode='train'):
        """ Общий шаг для обучения, валидации и теста

        Параметры
        ---------
        batch : dict
          Батч-словарь в следующем формате:
          {
          'pointcloud': облака точек,
          'target': метки класса,
          }
        batch_idx : int
          Номер батча
        mode : str
          Режим. Используется только для префикса названий ошибок в логе.
          По умолчанию train
        """


        forward_result = self(batch['embedding'])
        target_labels = batch['label']
        
        loss = F.cross_entropy(forward_result, target_labels)
        self.log(f'{mode}_loss', loss, prog_bar=True)

        predict_labels = torch.argmax(forward_result.cpu().detach(), axis=1).numpy()

        accuracy = accuracy_score(predict_labels, target_labels.cpu().detach().numpy())
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