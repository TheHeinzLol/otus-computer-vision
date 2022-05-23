import os
import json
import random
import shutil

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from detectron2.structures import BoxMode
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.logger import setup_logger
from detectron2.evaluation import COCOEvaluator
setup_logger()


class SportTeamsDatasetMaker:
    """ Создание датасета спортивных командв в формате Detectron 2
    
    Параметры
    ---------
    bboxes_json : str
      Путь к файлу bboxes.json
    images_path : str
      Путь к папке с фотографиями
    seed : int
      Фиксация генератору случайных чисел
    """
    
    def __init__(self, bboxes_json: str, image_path: str, seed=42):
        
        self.bboxes_json = bboxes_json
        self.images_path = image_path
        
        if self.images_path[-1] != os.path.sep:
            self.images_path += os.path.sep  
            
        self.seed = seed
        
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
            
            df_sport_teams.loc[df_sport_teams.photo == image_name, ['height',
                                                                    'width']
                              ] = image_size
            
        df_sport_teams = df_sport_teams.drop(columns=['bbox_x_norm', 'bbox_y_norm',
                                                      'bbox_w_norm', 'bbox_h_norm'])
        
        return df_sport_teams
        
    def prepare_dataframe(self):
        """ Запуск подготовки датафрейма с данными"""
        
        self.df_sport_teams_ = self._get_dataframe_form_bbox_file()
        assert len(self.df_sport_teams_) > 0
        self.df_sport_teams_ = self._add_photo_path_to_df(self.df_sport_teams_)
        self.df_sport_teams_ = self._make_df_features(self.df_sport_teams_)
        
    def _make_detectron_dict(self, photo_indexes):
        """ Создание словаря в формате Detectron
        
        Параметры
        ---------
        photo_indexes : list
          Список индексов изображений из датафрейма для подготовки словаря
        """
        
        if 'df_sport_teams_' not in self.__dir__():
            raise Exception("Для подготовки данных для детектрона зарустите метод prepare_dataframe")
        
        detectron_data = []
        
        for photo_idx in photo_indexes:
            
            detectron_photo_dict = {}
            
            photo_df = self.df_sport_teams_[self.df_sport_teams_.photo_number == photo_idx]
            assert len(photo_df) > 0, f'Не найдено изображение с идентификатором {photo_idx}'
            
            detectron_photo_dict['image_id'] = int(photo_idx)
            detectron_photo_dict['file_name'] = photo_df.photo.iloc[0]
            detectron_photo_dict['height'] = int(photo_df.height.iloc[0])
            detectron_photo_dict['width'] = int(photo_df.width.iloc[0])
            detectron_photo_dict['annotations'] = []
            
            for _, photo_row in photo_df.iterrows():
                annotation_dict = {
                    'bbox': [int(photo_row.bbox_x1), int(photo_row.bbox_y1), int(photo_row.bbox_x2), int(photo_row.bbox_y2)],
                    'bbox_mode': BoxMode.XYXY_ABS,
                    'category_id': int(photo_row.target),
                }
                detectron_photo_dict['annotations'].append(annotation_dict)
                
            detectron_data.append(detectron_photo_dict)
                
        return detectron_data
            
    def detectron_train_test_split(self, train_size=0.8):
        """ Получение словаря детектрона для обучения и валидации
        
        Параметры
        ---------
        train_size : float
          Доля части для обучения. По умолчанию 0.8
          
        Результат
        ---------
        train_detectron_dict, val_detectron_dict : (dict, dict)
        """
        
        assert 0 < train_size < 1
        
        uniq_photo_numbers = list(self.df_sport_teams_.photo_number.unique())
        
        np.random.seed(self.seed)
        np.random.shuffle(uniq_photo_numbers)
        
        photo_count = len(uniq_photo_numbers)
        
        train_photo_count = int(photo_count * train_size)
        
        train_photo_indexes = uniq_photo_numbers[:train_photo_count]
        test_photo_indexes = uniq_photo_numbers[train_photo_count:]
        
        train_detectron_dict = self._make_detectron_dict(photo_indexes=train_photo_indexes)
        val_detectron_dict = self._make_detectron_dict(photo_indexes=test_photo_indexes)
        
        return train_detectron_dict, val_detectron_dict

        
class TeamDetector:
    """ Детектор для датасета спортивных команд
    
    Иниицализирует детектор из библиотект Detectron2 и данные для его обучения
    
    Параметры
    ---------
    bboxes_json : str
      Путь к файлу bboxes.json
    images_path : str
      Путь к папке с фотографиями
    train_size : float
      Доля части для обучения. По умолчанию 0.8
    max_iter : int
      Число эпох обучения детектора. По умолчанию 1500
    lr : float
      Скорость обучения. По умолчанию 0.0001
    seed : int
      Фиксация генератору случайных чисел
    """
    
    def __init__(self, bboxes_json, image_path, train_size=0.8, max_iter=1500, lr=0.0001, seed=2022):
        
        
        self._max_iter = max_iter
        self._lr = lr
        
        maker = SportTeamsDatasetMaker(bboxes_json=bboxes_json, image_path=image_path, seed=seed)
        maker.prepare_dataframe()
        
        train_detectron_dict, val_detectron_dict = maker.detectron_train_test_split()
        
        self.detector_dataset_dict = {
            'train': train_detectron_dict,
            'val': val_detectron_dict,
        }
        
        DatasetCatalog.register('train', lambda x='train': self.detector_dataset_dict[x])
        DatasetCatalog.register('val', lambda x='val': self.detector_dataset_dict[x])
        
        MetadataCatalog.get("train").thing_classes = ['white', 'black']
        MetadataCatalog.get("val").thing_classes = ['white', 'black']
        
        self._init_config()
        self._init_trainer()
        
    def _init_config(self):
        """Инициализация детектора"""
        
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml"))
        self.cfg.DATASETS.TRAIN = ("train",)
        self.cfg.DATASETS.TEST = ("val",)
        self.cfg.TEST.EVAL_PERIOD = 0
        
        self.cfg.DATALOADER.NUM_WORKERS = 2
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_101_FPN_3x.yaml")

        self.cfg.INPUT.MAX_SIZE_TRAIN = 1280
        self.cfg.INPUT.MIN_SIZE_TRAIN = (570, 600, 630, 660, 690, 720)

        self.cfg.INPUT.MAX_SIZE_TEST = 1280
        self.cfg.INPUT.MIN_SIZE_TEST = 720

        self.cfg.SOLVER.IMS_PER_BATCH = 2
        self.cfg.SOLVER.BASE_LR = self._lr
        self.cfg.SOLVER.MAX_ITER = self._max_iter  
        self.cfg.SOLVER.STEPS = []    
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
        self.cfg.MODEL.RETINANET.NUM_CLASSES = 2
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        
    def _init_trainer(self):
        """Инициализация тренера"""
        self.trainer = DefaultTrainer(self.cfg) 
        self.trainer.resume_or_load(resume=False)
        
    def _init_predictor(self, score_thresh=0.4):
        """Инициализация предиктора
        
        Параметры
        ---------
        score_thresh : float
          Порог фильтрации предсказанных боксов по уровню доверия.
          По умолчанию 0.4
        """
        
        assert 0 < score_thresh < 1
        
        self.predictor_cfg = self.cfg.clone()
        self.predictor_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
        self.predictor_cfg.MODEL.RETINANET.SCORE_THRESH_TEST = score_thresh
        self.predictor = DefaultPredictor(self.predictor_cfg)
        
    def visualize_train_dataset(self):
        """Визуализация 3 случайных изоборажений из датасета обучения с GT-боксами"""
    
        for image_dict in random.sample(self.detector_dataset_dict['train'], 3):
            
            img = cv2.imread(image_dict["file_name"])
            visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get("train"), scale=0.5)
            out = visualizer.draw_dataset_dict(image_dict)
            plt.figure(figsize=(14,7))
            plt.imshow(out.get_image())
            plt.show()
            
    def visualize_val_dataset(self, score_thresh=0.4):
        """Визуализация 3 случайных изоборажений из датасета обучения с предсказанными боксами
        
        Параметры
        ---------
        score_thresh : float
          Порог фильтрации предсказанных боксов по уровню доверия.
          По умолчанию 0.4
        """
        
        self._init_predictor(score_thresh=score_thresh)
    
        for image_dict in random.sample(self.detector_dataset_dict['train'], 3):
                
            im = cv2.imread(image_dict["file_name"])
            outputs = self.predictor(im)
            v = Visualizer(im[:, :, ::-1],
                           metadata=MetadataCatalog.get("val"), 
                           scale=0.5
            )
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            plt.figure(figsize=(14,7))
            plt.imshow(out.get_image())
            plt.show()
            
    def fit(self):
        """Обучение детектора на датасете обучения"""
        self.trainer.train()
        self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, "model_final.pth")
        
        
    def eval(self, score_thresh=0.4):
        """
        Расчет метрики COCO mAP на валидационном части датасета
        Параметры
        ---------
        score_thresh : float
          Порог фильтрации предсказанных боксов по уровню доверия.
          По умолчанию 0.4
        """
        
        cfg = self.cfg.clone()
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = score_thresh
        
        output_dir = 'coco_eval_output'
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        
        evaluator = COCOEvaluator("val", cfg, False, output_dir=output_dir)
        self.trainer.test(self.cfg, self.trainer.model, evaluator)        
        
    def predict(self, score_thresh=0.4):
        """Обучение детектора для всех изображений валидационного датасета
        
        Параметры
        ---------
        score_thresh : float
          Порог фильтрации предсказанных боксов по уровню доверия.
          По умолчанию 0.4
        
        Результат
        ---------
        predict : [dict]
          Список словарей предсказаний в формате:
          {
            'bboxes': [(x1, y1, x2, y2), (), ..],
            'class': [class1, class2, ...],
            'conf': [conf1, conf2, ...],
          }
        """
        
        self._init_predictor(score_thresh=score_thresh)
        val_perdicts = []
        for image_dict in self.detector_dataset_dict['val']:
            
            im = cv2.imread(image_dict["file_name"])
            outputs = self.predictor(im)
            
            image_pred_dict = {
                'bboxes': outputs['instances'].get('pred_boxes').tensor.cpu().numpy().astype(int).tolist(),
                'class': outputs['instances'].get('pred_classes').cpu().numpy().tolist(),
                'conf': outputs['instances'].get('scores').cpu().numpy().tolist(),
            }               
            
            val_perdicts.append(image_pred_dict)
            
        return val_perdicts