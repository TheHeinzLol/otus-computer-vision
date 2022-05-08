import json
import os.path
import os

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy import stats

class TeamDataframeMaker:
    """ Формирование датафрейма для обучения
    
    Параметры
    ---------
    bboxes_json : str
      Путь к файлу bboxes.json
    images_path : str
      Путь к папке с фотографиями
    """
    
    def __init__(self, bboxes_json: str, image_path: str):
        
        self.bboxes_json = bboxes_json
        self.images_path = image_path
        if self.images_path[-1] != os.path.sep:
            self.images_path += os.path.sep
        
    @staticmethod
    def _read_image(image_path:str, rgb: bool=False) -> np.ndarray:
        """ Чтение изображения
        
        Параметры
        ---------
        image_path : str
          Путь к изображению на диске
        rgb : bool
          Если True, то результат будет в формате RGB.
          Иначе BGR.
          
        Результат
        ---------
        img : np.ndarray
        """
        img = cv2.imread(image_path)
        if rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    @staticmethod
    def _draw_bboxes(img: np.ndarray, bboxes: list, targets: list) -> np.ndarray:
        """ Печать боксов на изображении
        
        Параметры
        ---------
        img : np.ndarray
          Изображение
        bboxes : list
          Список координат боксов в формате
          [(x1, y1, x2, y2),(),(),...]
        targets : list
          Целевая метка для каждого бокса, 0 или 1.
          
        Результат
        ---------
        bbox_img : np.ndarray
        """
        
        for bbox, target in zip(bboxes, targets):
            bbox = [int(val) for val in bbox]
            if target == 0:
                color = (255,0,0)
            else:
                color = (0,0,255)
            img = cv2.rectangle(img, bbox[:2], bbox[2:], color, thickness=5)
        
        return img
    
    def _get_dataframe_form_bbox_file(self) -> pd.DataFrame:
        """ Получаем базовый датафрейм из файла self.bboxes_json
        
        Результат
        ---------
        df_sport_teams : pd.DataFrame
          Датафрейм имеет следующие колонки:
          - photo (номер фотографии)
          - bbox_x_norm (нормированная координата x бокса на изображении)
          - bbox_y_norm (нормированная координата y бокса на изображении)
          - bbox_w_norm (нормированная ширина бокса на изображении)
          - bbox_h_norm (нормированная высота бокса на изображении)
          - target (метка номера команды для бокса на изображении)
        """
        
        df_sport_teams = pd.DataFrame()
        with open(self.bboxes_json, 'r', encoding='utf-8') as f:
            bbox_dict = json.load(f)
            
        for photo_number, photo_data in bbox_dict.items():
            for player_number, bbox_data in photo_data.items():
                bbox_x_norm, bbox_y_norm, bbox_w_norm, bbox_h_norm = bbox_data['box']
                target = bbox_data.get('team')
                df_sport_teams = pd.concat([df_sport_teams, pd.DataFrame({'photo_number': [int(photo_number)],
                                                                          'player_number': [int(player_number)],
                                                                          'bbox_x_norm': [bbox_x_norm],
                                                                          'bbox_y_norm': [bbox_y_norm],
                                                                          'bbox_w_norm': [bbox_w_norm],
                                                                          'bbox_h_norm': [bbox_h_norm],
                                                                          'target': [target],
                                                                         })])
                df_sport_teams.index = np.arange(len(df_sport_teams))
                
        return df_sport_teams
    
    def _add_photo_path_to_df(self, df_sport_teams: pd.DataFrame) -> pd.DataFrame:
        """ Добавляем путь к изображению в датафрейм и проверяем наличие изображения
        
        Функция меняет входной датафрейм.
        
        Параметры
        ---------
        df_sport_teams : pd.DataFrame
          Датафрейм, который имеет колонку photo с именем изображения
          
        Результат
        ---------
        df_sport_teams : pd.DataFrame
          В колонке photo будет полный путь к изображению
        """
        
        def get_photo_path(photo_name, images_path):
            photo_path = str(images_path) + str(photo_name) + '.jpeg'
            if not os.path.exists(photo_path):
                raise Exception(f"Изображение {photo_path} не существует")
            return photo_path
                
        df_sport_teams.loc[:, 'photo'] = df_sport_teams.loc[:, 'photo_number'].apply(get_photo_path, images_path=self.images_path)
        
        return df_sport_teams
    
    @staticmethod
    def _bbox_denormalize(image_size: tuple, bboxes: tuple) -> tuple:
        """ Денормализация координат боксов
        
        Параметры
        ---------
        image_size : tuple
          Размер изображения бокса (width, height)
        bbox : tuple
          Кортеж нормированных координат бокса
          (bbox_x_norm, bbox_y_norm, bbox_w_norm, bbox_h_norm)
          
        Результат
        ---------
        bbox_denormalized : tuple
          Кортеж денормированных координат бокса
          (bbox_x, bbox_y, bbox_w, bbox_h)
        """
        
        image_height, image_width = image_size
        
        bbox_denormalized = []
        for bbox in bboxes:
        
            bbox_x_norm, bbox_y_norm, bbox_w_norm, bbox_h_norm = bbox

            bbox_x1 = int(image_width * bbox_x_norm)
            bbox_y1 = int(image_height * bbox_y_norm)
            
            bbox_w = image_width * bbox_w_norm
            bbox_h = image_height * bbox_h_norm
            
            bbox_x2 = int(bbox_x1 + bbox_w)
            bbox_y2 = int(bbox_y1 + bbox_h)
            
            bbox_denormalized.append((bbox_x1, bbox_y1, bbox_x2, bbox_y2))

        return bbox_denormalized
    
    @staticmethod
    def _get_bboxes_channels_mean(img: np.ndarray, bboxes: list, mean_type='rgb'):
        """ Получаем средние значения для бокса в разрезе каналов
        
        Параметры
        ---------
        img : np.ndarray
          Массив NumPy с изображением в формате BGR
          (W, H, C)
        bboxes : list
          Список боксов изображения, для которых 
          нужно выполнить расчет среднего в формате:
          [(bbox1_x1, bbox1_y1, bbox1_x2, bbox1_y2), (), (), ..]
        mean_type : str
          Тип каналов для расчета среднего: rgb или hsv. 
          По умолчанию rgb.
          
        Результат
        ---------
        bbox_mean_result : np.ndarray
          [[bbox1_ch1_mean, bbox1_ch2_mean, bbox1_ch3_mean], [], [], ...]
        """
        
        mean_type = str(mean_type).lower().strip()
        assert mean_type in ('rgb', 'hsv')
        
        image_cvt = np.zeros_like(img)
        
        if mean_type == 'hsv':
            cv2.cvtColor(img, cv2.COLOR_BGR2HSV, image_cvt)
        else:
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, image_cvt)
        
        bbox_mean_result = []
        for bbox in bboxes:
            bbox = [int(val) for val in bbox]
            mask = np.zeros(shape=image_cvt.shape[:2], dtype="uint8")
            mask = cv2.rectangle(mask, bbox[:2], bbox[2:], 1, thickness=-1)
            masked_image = cv2.bitwise_and(image_cvt, image_cvt, mask=mask)
            # Находим сумму значений по каналам с маской
            # Делим на сумму единиц в маске, не учитываем в знаменателе нулевые элементы
            means_per_channel = np.sum(np.sum(masked_image, axis=0), axis=0) / np.sum(mask)
            bbox_mean_result.append(means_per_channel)
            
        return np.array(bbox_mean_result)
    
    @staticmethod
    def _get_bboxes_histogram(img: np.ndarray, bboxes: list, bins=16, hist_type='rgb'):
        """ Получаем значения бинов гистограммы для бокса в разрезе каналов
        
        Параметры
        ---------
        img : np.ndarray
          Массив NumPy с изображением в формате BGR
          (W, H, C)
        bboxes : list
          Список боксов изображения, для которых 
          нужно выполнить расчет среднего в формате:
          [(bbox1_x1, bbox1_y1, bbox1_x2, bbox1_y2), (), (), ..]
        bins : int
          Количество бинов для гистограммы
        hist_type : str
          Тип каналов для расчета гистограммы: rgb или hsv. 
          По умолчанию rgb.
          
        Результат
        ---------
        bbox_hist_result : np.ndarray
          [[bbox1_ch1_bin1, bbox1_ch1_bin2, ..., bbox1_ch2_bin1, bbox1_ch2_bin2, ..., bbox1_ch3_bin1, ...], [], [], ...]
        """
        
        hist_type = str(hist_type).lower().strip()
        assert hist_type in ('rgb', 'hsv')
        
        image_cvt = np.zeros_like(img)
        
        if hist_type == 'hsv':
            cv2.cvtColor(img, cv2.COLOR_BGR2HSV, image_cvt)
        else:
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, image_cvt)
        
        bbox_hist_result = []
        for bbox in bboxes:
            bbox = [int(val) for val in bbox]
            mask = np.zeros(shape=image_cvt.shape[:2], dtype="uint8")
            mask = cv2.rectangle(mask, bbox[:2], bbox[2:], 1, thickness=-1)
            res = []
            for idx_channel in range(3):
                bbox_hist = cv2.calcHist([image_cvt], [idx_channel], mask, [bins], [0, 256]).flatten()
                res.append(bbox_hist)
            res = np.array(res).flatten()
            bbox_hist_result.append(res)
            
        return np.array(bbox_hist_result)
        
    def print_random_image(self):
        """Печать случайной фотографии с боксами"""
        
        uniq_images = self.df_sport_teams_.photo.unique()
        choosed_photo = uniq_images[np.random.randint(len(uniq_images))]
        
        img = self._read_image(choosed_photo, rgb=True)
        
        bboxes = self.df_sport_teams_.loc[self.df_sport_teams_.photo == choosed_photo, 
                                          ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']].to_numpy()
        targets = self.df_sport_teams_.loc[self.df_sport_teams_.photo == choosed_photo, 'target'].to_numpy()
        
        img = self._draw_bboxes(img, bboxes=bboxes, targets=targets)
        
        plt.figure(figsize=(12, 7))
        plt.title(choosed_photo)
        plt.imshow(img)
        plt.show()
        
    def _make_df_features(self, df_sport_teams):
        """ Получаем значения всех признаков для датафрейма df_sport_teams
        
        Параметры
        ---------
        df_sport_teams : pd.DataFrame
          Датафрейм, который имеет колонки photo, bbox_x_norm, bbox_y_norm, bbox_w_norm, bbox_h_norm.
          photo должен содержать полный путь к изображению.
          
        Результат
        ---------
        df_sport_teams_upd : pd.DataFrame
          В датафрейм будут добавлены следующие признаки:
          - bbox_x1, bbox_y1, bbox_x2, bbox_y2 (денормализованные координаты бокса)
          - HSV_mean_h, HSV_mean_s, HSV_mean_v (средние значения по каналам HSV для каждого бокса)
          - RGB_mean_r, RGB_mean_g, RGB_mean_b (средние значения по каналам RGB для каждого бокса)
          - HSV_hist_КОДКАНАЛА_НОМЕРБИНА (значения гистограммы бокса для каждого канала HSV)
          - RGB_hist_КОДКАНАЛА_НОМЕРБИНА (значения гистограммы бокса для каждого канала RGB)
        """
        
        uniq_images = df_sport_teams.photo.unique()
        
        for image_name in uniq_images:
            
            img = self._read_image(image_name, rgb=False)
            image_size = img.shape[:2]
            
            bboxes_norm = df_sport_teams.loc[df_sport_teams.photo == image_name, 
                                             ['bbox_x_norm', 'bbox_y_norm', 'bbox_w_norm', 'bbox_h_norm']].to_numpy()
            
            df_sport_teams.loc[df_sport_teams.photo == image_name, ['bbox_x1',
                                                                    'bbox_y1',
                                                                    'bbox_x2',
                                                                    'bbox_y2']
                              ] = self._bbox_denormalize(image_size, bboxes=bboxes_norm)
            
            bboxes = df_sport_teams.loc[df_sport_teams.photo == image_name, 
                                        ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']].to_numpy()
            
            
            
            df_sport_teams.loc[df_sport_teams.photo == image_name, ['HSV_mean_h',
                                                                    'HSV_mean_s',
                                                                    'HSV_mean_v']
                              ] = self._get_bboxes_channels_mean(img, bboxes=bboxes, mean_type='hsv')
            
            df_sport_teams.loc[df_sport_teams.photo == image_name, ['RGB_mean_r',
                                                                    'RGB_mean_g',
                                                                    'RGB_mean_b']
                              ] = self._get_bboxes_channels_mean(img, bboxes=bboxes, mean_type='rgb')
            
            hist_bins_count = 16
            bgr_hist_names_col = []
            for idx in range(hist_bins_count*3):
                val = idx // hist_bins_count
                if val == 0:
                    col_name = f'RGB_hist_r_bin_{idx}'
                elif val == 1:
                    col_name = f'RGB_hist_g_bin_{idx-hist_bins_count}'
                else:
                    col_name = f'RGB_hist_b_bin_{idx-hist_bins_count*2}'
                bgr_hist_names_col.append(col_name)
                    
            assert len(bgr_hist_names_col) == hist_bins_count*3
            
            df_sport_teams.loc[df_sport_teams.photo == image_name, bgr_hist_names_col
                              ] = self._get_bboxes_histogram(img, bboxes=bboxes, bins=hist_bins_count, hist_type='rgb')
            
            hsv_hist_names_col = []
            for idx in range(hist_bins_count*3):
                val = idx // hist_bins_count
                if val == 0:
                    col_name = f'HSV_hist_h_bin_{idx}'
                elif val == 1:
                    col_name = f'HSV_hist_s_bin_{idx-hist_bins_count}'
                else:
                    col_name = f'HSV_hist_v_tbin_{idx-hist_bins_count*2}'
                hsv_hist_names_col.append(col_name)
                    
            assert len(bgr_hist_names_col) == hist_bins_count*3
            
            df_sport_teams.loc[df_sport_teams.photo == image_name, hsv_hist_names_col
                              ] = self._get_bboxes_histogram(img, bboxes=bboxes, bins=hist_bins_count, hist_type='hsv')
            
        df_sport_teams = df_sport_teams.drop(columns=['bbox_x_norm', 'bbox_y_norm',
                                                      'bbox_w_norm', 'bbox_h_norm'])
            
        return df_sport_teams
    
    def prepare_dataframe(self):
        """ Запуск подготовки датафрейма"""
        self.df_sport_teams_ = self._get_dataframe_form_bbox_file()
        self.df_sport_teams_ = self._add_photo_path_to_df(self.df_sport_teams_)
        self.df_sport_teams_ = self._make_df_features(self.df_sport_teams_)
        
   
class TeamClassifier:
    """ Классификатор для скортивных команд на базе моделей sklearn
    
    С помощью model_class можно указать какой-либо класс модели из библиотеки sklearn.
    Модель обучается с помощью стратифицированной кросс-валидации. Указываем необходимое количество
    фолдов, для каждого фолда будет создан экземпляр модели выбранного класса.
    И каждая модель обучится на своем фолде. Для предсказаний будет использована
    каждая модель и результат будет выбираться по наибольшему количеству голосов для класса.
    
    Параметры
    ---------
    team_df_maker : TeamDataframeMaker
      Экземпляр класса TeamDataframeMaker
    fit_features_names : list
      Список названия признаков, используемыз для обучения
    model_class : sklearn model
      Класс модели из sklearn без инициализации
    model_params : dict
      Словарь параметров для модели. 
      Параметры следует брать из описания конкретной модели.
    folds : int
      Количество фолдов для разбиения датасета.
    model_type : str
      Определяет тип модели. Может принимать значения
      training или clustering.
      Для training в параметре model_class необходимо
      передавать обучаемую на целевых метках модель. 
      А для clustering модель кластеризации без обучения
      на целевых метках.
    random_state : int
    debug : bool
    """
    
    def __init__(self, team_df_maker, fit_features_names, model_class, model_params=None, folds=5,
                 model_type='training', random_state=42, debug=False):
        
        model_type = str(model_type).lower().strip()
        assert model_type in ('clustering', 'training')
        
        self._team_df_maker = team_df_maker
        self.__fit_features_names = fit_features_names
        self.__model_class = model_class
        self.__model_params = model_params
        self.__folds = folds
        self._model_type = model_type
        self.random_state = random_state
        self.__target_feature_name = 'target'
        self.__model_is_fitted = False
        self.debug = debug
        
    def __init_models(self):
        """ Инициализация модели"""
        
        # Инициализируем модель для стандартизации
        self.__scaler = StandardScaler()
        # В зависимости от размера фолда создаем нужное кол-во одинаковых моделей
        self.__models = [self.__model_class(**self.__model_params) for _ in range(self.__folds)]
        # Инициализируем модель для разбиения на фолды
        self.__skf = StratifiedKFold(n_splits=self.__folds, shuffle=True, random_state=self.random_state)
        n_splits = self.__skf.get_n_splits(self._team_df_maker.df_sport_teams_[self.__fit_features_names], 
                                           self._team_df_maker.df_sport_teams_[self.__target_feature_name])
        assert n_splits == len(self.__models)
        # Здесь после обучения будут храниться accuracy для каждой модели
        self._train_kfold_acc = []
        
        # Если модель не обучена, то ряд методов класса будет нельзя выполнить
        self.__model_is_fitted = False
        
    @property
    def team_df_maker(self):
        return self._team_df_maker
    @team_df_maker.setter
    def team_df_maker(self, team_df_maker):
        self._team_df_maker = team_df_maker
        self.__model_is_fitted = False
    
    @property
    def fit_features_names(self):
        return self.__fit_features_names
    @fit_features_names.setter
    def fit_features_names(self, fit_features_names):
        self.__fit_features_names = fit_features_names
        self.__model_is_fitted = False
        
    @property
    def model_class(self):
        return self.__model_class
    @model_class.setter
    def model_class(self, model_class):
        self.__model_class = model_class
        self.__model_is_fitted = False
        
    @property
    def model_params(self):
        return self.__model_params
    @model_params.setter
    def model_params(self, model_params):
        self.__model_params = model_params
        self.__model_is_fitted = False
        
    @property
    def folds(self):
        return self.__folds
    @folds.setter
    def folds(self, folds):
        self.__folds = folds
        self.__model_is_fitted = False
        
    @property
    def train_kfold_accuracy(self):
        if hasattr(self, '_train_kfold_acc'):
            return np.mean(self._train_kfold_acc)
        
    @classmethod
    def load_from_json(cls, bboxes_json, image_path, model_type='training', fit_features_names=None, model_class=None, model_params=None, 
                       folds=5, random_state=42):
        """ Инициализация TeamClassifier с помощью файла bboxes.json
        
        Параметры
        ---------
        bboxes_json : str
          Путь к файлу bboxes.json
        images_path : str
          Путь к папке с фотографиями
        fit_features_names : list
          Список названия признаков, используемыз для обучения
        model_class : sklearn model
          Класс модели из sklearn без инициализации.
          ПО умолчанию RandomForestClassifier.
        model_params : dict
          Словарь параметров для модели. 
          Параметры следует брать из описания конкретной модели.
        folds : int
          Количество фолдов для разбиения датасета.
        random_state : int
        """
        
        if model_class is None:
            model_class = RandomForestClassifier
            model_params = {'n_estimators': 300, 'max_depth': 20, 'random_state': random_state, 'n_jobs': -1}
        
        team_df_maker = TeamDataframeMaker(bboxes_json=bboxes_json, image_path=image_path)
        team_df_maker.prepare_dataframe()
        
        if fit_features_names is None:
            filter_f_names = ('photo', 'photo_number', 'player_number', 'target', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2')
            fit_features_names = [f_name for f_name in team_df_maker.df_sport_teams_.columns if f_name not in filter_f_names]
        
        return cls(team_df_maker=team_df_maker, model_type=model_type, fit_features_names=fit_features_names, model_class=model_class, 
                   model_params=model_params, folds=folds)
    
    def fit(self):
        """Обучение модели"""
        
        self.__init_models()
            
        # Для каждого фолда обучаем отдельную модель
        for idx, (train_index, 
                  test_index) in enumerate(self.__skf.split(self._team_df_maker.df_sport_teams_[self.__fit_features_names], 
                                                            self._team_df_maker.df_sport_teams_[self.__target_feature_name])):
            
            # Поулчаекм модель для выбранного фолда
            model = self.__models[idx]
            
            assert 'predict' in model.__dir__(), "У модели выбранного класса отсутствует метод predict, используйте метод fit_predict"
            
            # Получаем данные для обучения выбранного фолда
            train_fold_df = self._team_df_maker.df_sport_teams_.iloc[train_index, :]
            test_fold_df = self._team_df_maker.df_sport_teams_.iloc[test_index, :]

            # В данных оставляем только признаки из self.__fit_features_names
            X_train = train_fold_df.loc[:, self.__fit_features_names]
            X_val = test_fold_df.loc[:, self.__fit_features_names]
            
            assert len(X_train) > 0
            
            y_train = train_fold_df[self.__target_feature_name]
            y_val = test_fold_df[self.__target_feature_name]
            
            # Обучаем и применяем модель стандартизации
            if idx == 0:
                X_train_std = self.__scaler.fit_transform(X_train)
                X_val_std = self.__scaler.transform(X_val)
            else:
                X_train_std = self.__scaler.transform(X_train)
                X_val_std = self.__scaler.transform(X_val)
              
            # Обучаем модель на стандартизированных данных фолда
            
            if self._model_type == 'training':
                model.fit(X_train_std, y_train)
            else:
                y_pred = model.fit(X_train_std)
               
            y_pred = model.predict(X_val_std) 
            if self.debug:
                print(f'Предсказания модели на фолде {idx}:')
                print(y_pred)
            self._train_kfold_acc.append(accuracy_score(y_pred, y_val))
                
        # Ставим флаг обученности модели
        self.__model_is_fitted = True
        
    def fit_predict(self):
        """ Используется для моделей кластеризации, у которых для предсказания используется только метод fit_ppredict
        
        Метод использовать только для моделей кластеризации, у которых нет метода predict.
        В этом случае не используются фолды, а проверка accuracy происходит на всем датасете.
        
        Результат
        ---------
        
        """
        assert self._model_type == 'clustering'
        
        self.__init_models()
        
        # Поулчаекм модель для выбранного фолда
        model = self.__models[0]
        
        assert 'predict' not in model.__dir__(), "У модели выбранного класса есть метод predict, используется predict вместо fit_predict"
        assert 'fit_predict' in model.__dir__(), "У модели выбранного класса отсутствует метод fit_predict"

        # В данных оставляем только признаки из self.__fit_features_names
        X_train = self._team_df_maker.df_sport_teams_.loc[:, self.__fit_features_names]
        y_val = self._team_df_maker.df_sport_teams_.loc[:, self.__target_feature_name]

        assert len(X_train) > 0

        X_train_std = self.__scaler.fit_transform(X_train)

        # Обучаем модель на стандартизированных данных фолда
        
        y_pred = model.fit_predict(X_train_std)

        self._train_kfold_acc.append(accuracy_score(y_pred, y_val))  
        
        return y_pred
            
    def predict(self, df_sport_teams: pd.DataFrame, kfold_predict_format: bool=False) -> np.ndarray:
        """Предсказание модели
        
        Только для классов моделей, у которых есть метод predict.
        
        Параметры
        ---------
        df_sport_teams : pd.DataFrame
          Датафрейм, получаемый с помощью класса TeamDataframeMaker
        kfold_predict_format : bool
          Если True, то результат будет в формате:
            (N, C), где N - количество боксов, С - предсказание каждой модели
          Иначе формат (N,) - считается значение по большинству голосов
          
        Результат
        ---------
        y_pred : np.ndarray
        """
        
        if not self.__model_is_fitted:
            raise Exception("Модель не обучена, запустите метод fit")
        
        # В данных оставляем только признаки из self.__fit_features_names
        X = df_sport_teams[self.__fit_features_names]
        X_std = self.__scaler.transform(X)
        
        kfold_predict = []
        for idx, model in enumerate(self.__models):
            
            assert 'predict' in model.__dir__(), "У модели выбранного класса отсутствует метод predict, используйте метод fit_predict"
            
            fold_y_pred = model.predict(X_std)
            kfold_predict.append(fold_y_pred)
            
            if self.debug:
                print('===================================')
                print(f'Предсказания модели на фолде {idx}:')
                print(fold_y_pred)
            
        kfold_predict = np.array(kfold_predict).T
        
        if kfold_predict_format:
            return kfold_predict
        
        kfold_mode = stats.mode(kfold_predict, axis=1)
        
        return kfold_mode.mode.flatten()
        
    def predict_from_json(self, bboxes_json: str, image_path: str, kfold_predict_format: bool=False,
                          return_json=False) -> np.ndarray:
        """Предсказание модели для файлов json с разметкой боксов
        
        Только для классов моделей, у которых есть метод prediсt
        
        Параметры
        ---------
        bboxes_json : str
          Путь к файлу bboxes.json
        images_path : str
          Путь к папке с фотографиями
        kfold_predict_format : bool
          Если True, то результат будет в формате:
            (N, C), где N - количество боксов, С - предсказание каждой модели
          Иначе формат (N,) - считается значение по большинству голосов
        return_json : bool
          Если True, то результат предсказания будет выдан в формате JSON.
          Можно использовать только с kfold_predict_format=False.
          
        Результат
        ---------
        y_pred : np.ndarray или [dict, dict, ... ]
        """
        
        if not self.__model_is_fitted:
            raise Exception("Модель не обучена, запустите метод fit")
        
        if kfold_predict_format == True and return_json == True:
            raise Exception("Для поулчения результата в json запустите предсказание с параметром kfold_predict_format=False")
        
        team_df_maker = TeamDataframeMaker(bboxes_json=bboxes_json, image_path=image_path)
        team_df_maker.prepare_dataframe()
        
        model_predict = self.predict(team_df_maker.df_sport_teams_, kfold_predict_format=kfold_predict_format)
        
        if not return_json:
            return model_predict
        
        df_sport_teams_predicted = team_df_maker.df_sport_teams_.copy()
        df_sport_teams_predicted.loc[:, 'predicted_target'] = model_predict
        
        json_predict_data = {}
        
        for frame_number in team_df_maker.df_sport_teams_.photo_number.unique():
            
            frame_dict = {}
            frame_df = df_sport_teams_predicted[df_sport_teams_predicted.photo_number == frame_number]
            
            for _, row in frame_df.iterrows():
                frame_dict[int(row.player_number)] = int(row.predicted_target)
                
            json_predict_data[int(frame_number)] = frame_dict
            
        return json_predict_data