import random
import os
import pickle
from pathlib import Path
import shutil
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import mmcv
from mmdet.apis import init_detector, inference_detector
import ffmpeg


FRAME_NUMBER_TEMPLATE = "04d"

class FFMpegOperations:
    @staticmethod
    def extract_frames_from_one_video(video_path, output_video_frames_path_template, fps):
        (
        ffmpeg
        .input(str(video_path), r=fps)
        .output(str(output_video_frames_path_template), **{'q:v': 1})
        .run()
        )
        
        
    @staticmethod
    def calculate_frame_number_from_seconds(seconds, fps):
        """ Рассчет номера фрейма по его времени в видео
        
        Параметры
        ---------
        seconds : float
          Время кадра
        fps : int
          Кол-во кадров в секунду
        """
        int_seconds_value = int(seconds)
        miliseconds = (seconds - int_seconds_value) * 100
        miliseconds_in_minute = 100
        
        assert miliseconds < miliseconds_in_minute
        
        return fps * int_seconds_value + round(fps / miliseconds_in_minute * miliseconds)
    
    
    @staticmethod
    def calculate_seconds_from_frame_number(frame_number, fps):
        """ Рассчет времени видео по номеру кадра
        
        Параметры
        ---------
        seconds : float
          Время кадра
        fps : int
          Кол-во кадров в секунду
        """
        return round(frame_number / fps, 3)
    
    
    @staticmethod
    def make_image_sequence(input_template, start_number, output_clip, fps=30, clip_size=60):
        (
        ffmpeg
        .input(str(input_template), pattern_type="sequence", start_number=start_number, framerate=fps)
        .output(str(output_clip), **{'frames:v': clip_size})
        .run()
        )
        
    @staticmethod      
    def get_frame_by_time(video_path, frame_time):
        """ Получаем кадр видео по таймингу
        
        Параметры
        ---------
        video_path : str
          Путь к видеофайлу
        frame_time : float
          Время на видео
          
        Результат
        ---------
        image : np.array
          Массив размером HxWxC
        """
        
        vcap = cv2.VideoCapture(str(video_path))
        image_width  = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        image_height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        out, _ = (
            ffmpeg
            .input(str(video_path), ss=frame_time)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24', vframes=1)
            .run(capture_stdout=True)
        )
        
        image = (
            np
            .frombuffer(out, np.uint8)
            .reshape([-1, image_height, image_width, 3])
        )
        
        assert image.shape[0] == 1, "Не найден кадр"
        image = image[0]
        
        return image
    
    @staticmethod
    def concat_videos(videos_paths, output_file):
        
        videos_inputs = [ffmpeg.input(str(path)) for path in videos_paths]
        (
            ffmpeg
            .concat(*videos_inputs)
            .output(str(output_file))
            .run()
        )
        

class CustomAvaVideoHelper:
    """Помощь в подготовке кастомного датасета AVA
    
    Параметры
    ---------
    source_videos_dir : str
      Папка с исходными видео. У каждого файла должно быть уникальное
      имя без пробелов. На основ имени файла будут создаваться папки
      с кадрами и имя будет использоваться в датасете.
    output_dir : str
      Папка для выходных результатов. По умолчанию "ava"
    fps : int
      Целевое значение FPS видео. Все видео будут сконвертированы
      в FPS с указанным значением. По умолчанию 30.
    """
    
    def __init__(self, source_videos_dir, output_dir="ava", fps=30, clip_size=60):
    
        self._source_videos_dir = Path(source_videos_dir)
        
        if not self._source_videos_dir.exists():
            raise Exception(f"Папка {source_videos_dir} не существует")
            
        self._output_dir = Path(output_dir)
        
        if not self._output_dir.exists():
            self._output_dir.mkdir()
            
        self._fps = fps
        self._ffmpgeg_frame_template = f"img_%{FRAME_NUMBER_TEMPLATE}.jpg"
        self._python_frame_template = "img_{:" + str(FRAME_NUMBER_TEMPLATE) + "}.jpg"
        self._frames_output_dir = self._output_dir / "frames"
        self._clips_output_dir = self._output_dir / "clips"
        self._clip_size = clip_size
        self._frames_video_ids = []
        
        
    @property
    def video_ids(self):
        return self._frames_video_ids
  
    def extract_frames(self):
        """ Распаковка всех доступных видео в последовательность изображений
        """
        if not self._frames_output_dir.exists():
            self._frames_output_dir.mkdir()
            
        for current_video_path in self._source_videos_dir.glob(f"*.[am][vp][i4]"):
            output_video_id = current_video_path.stem
            self._frames_video_ids.append(output_video_id)
            output_video_frames_path = self._frames_output_dir / output_video_id
            
            if not output_video_frames_path.exists():
                output_video_frames_path.mkdir()
                
            output_video_frames_path_template =  output_video_frames_path / self._ffmpgeg_frame_template
            FFMpegOperations.extract_frames_from_one_video(video_path=current_video_path, 
                                                           output_video_frames_path_template=output_video_frames_path_template,
                                                           fps=self._fps)
        
            
    def make_clips_from_dataframe(self, dataframe: pd.DataFrame, save_frame_image=False):
        """ Создание клипов на основе данных датафрейма
        
        Параметры
        ---------
        dataframe : pd.DataFrame
          Датафрейм, который должен содержать колонки video_id и frame_time
        save_frame_image : bool
          Дополнительно сохранить фрейм из видео
        """
        
        assert "video_id" in dataframe.columns
        assert "frame_time" in dataframe.columns or "frame_number" in dataframe.columns
        
        if "frame_time" in dataframe.columns:
            frame_id_column_name = "frame_time"
        else:
            frame_id_column_name = "frame_number"
        
        processed_videos = set()
        
        for _, row in dataframe.iterrows():
            
            frame_id = getattr(row, frame_id_column_name)
            frame_signature = (row.video_id, frame_id)
            
            if frame_signature in processed_videos:
                continue

            self.make_clip(video_id=row.video_id, frame_id=frame_id, save_frame_image=save_frame_image) 
            
            processed_videos.add(frame_signature)
            
            
    def make_clip(self, video_id, frame_id, save_frame_image=False):
        """ Создание клипа из выбранного кадра.
        Выбранный кадр становится центральным, дальше берутся границы
        
        video_id : string
          Идентификатор видео. Это имя видео без его расширения.
        frame_id : int или float
          Если int - номер центрального кадра клипа. Если float - то время центрального кадра.
        save_frame_image : bool
          Дополнительно сохранить фрейм из видео
        """
        
        if not video_id in self._frames_video_ids:
            raise Exception(f"Видео с идентификатором {video_id} отстутствует в папке с фреймами. Запустите extract_frames()")
            
        if not self._clips_output_dir.exists():
            self._clips_output_dir.mkdir()
            
        if isinstance(frame_id, float):
            frame_id = FFMpegOperations.calculate_frame_number_from_seconds(frame_id, self._fps)
           
        string_frame_time = str(FFMpegOperations.calculate_seconds_from_frame_number(frame_id, self._fps))
        string_frame_time = "_".join(string_frame_time.split('.'))
        
        output_clip = self._clips_output_dir / video_id / f"frame_{frame_id}_time_{string_frame_time}.mp4"
        
        if not output_clip.parent.exists():
            output_clip.parent.mkdir()
            
        if output_clip.exists():
            os.remove(output_clip)
        
        input_template = self._frames_output_dir / video_id / Path(self._ffmpgeg_frame_template)
        start_number = frame_id - self._clip_size // 2
        
        FFMpegOperations.make_image_sequence(input_template=input_template, start_number=start_number, 
                                             output_clip=output_clip, fps=self._fps, clip_size=self._clip_size)
        
        if save_frame_image:
            input_image = self._frames_output_dir / video_id / Path(self._python_frame_template.format(frame_id))
            output_image = self._clips_output_dir / video_id / f"frame_{frame_id}_time_{string_frame_time}.jpeg"
            shutil.copyfile(input_image, output_image)
            
                        
    @classmethod
    def load_from_extracted_frame_folders(cls, source_videos_dir, output_dir, fps=30, clip_size=60):
        """Инициализация класса для случаев, когда уже все видео были преобразованы в фреймы"""
        
        class_object = cls(source_videos_dir=source_videos_dir, output_dir=output_dir, fps=fps, clip_size=clip_size)
        
        for frame_dir in class_object._frames_output_dir.glob("*"):
            if not frame_dir.is_dir():
                continue
                
            class_object._frames_video_ids.append(frame_dir.name)
        
        return class_object
        

class MarkupsConverterType1:
    
    def __init__(self, input_dir, fps=30):
        """
        Разметка Type1 - разметка в формате txt, которая
        черех запятую содержит название действия и время в секундах.
        При этом имя файла должно быть в виде идентификатора видео.
        Например, видео HBVSMCGRG01.mp4 и разметка HBVSMCGRG01.txt
        
        Параметры
        ---------
        input_dir : str
          Папка с файлами разметки
        fps : int
          Количество кадров в секунду у видеороликов
        """
        
        self._input_dir = Path(input_dir)
        self._fps = fps
        
        self.__init_params()
        
        
    def __init_params(self):
        self._markup_dataframe = None
        self._action_types_to_code = {}
        self._code_to_action_types = {}
        
        if not self._input_dir.exists():
            raise Exception(f"Папка {self._input_dir} не существует")

        if not self._input_dir.is_dir():
            raise Exception(f"Файл {self._input_dir} не является папкой")

    @property
    def markup_dataframe(self):
        return self._markup_dataframe
    
    
    @property
    def action_types_to_code(self):
        return self._action_types_to_code
    
    
    @property
    def code_to_action_types(self):
        return self._code_to_action_types
    

    def convert_markups(self):
        """Конвертация разметки видео в датафрейм
        
        Результат
        ---------
        markup_dataframe : pd.DataFrame
          Датафрейм со столбцами video_id, frame_number и action_code
        """
        
        self.__init_params()
        
        self._markup_dataframe = pd.DataFrame()

        for markup_file in self._input_dir.glob("*.txt"): 
            video_id = markup_file.stem

            with open(markup_file, 'r') as file:
                for line in file:
                    action_type, frame_time = line.split(",")
                    action_type = action_type.lower().strip()
                    frame_time = float(frame_time.strip())
                    frame_number = FFMpegOperations.calculate_frame_number_from_seconds(frame_time, self._fps)

                    self._action_types_to_code.setdefault(action_type, len(self._action_types_to_code))
                    action_code = self._action_types_to_code[action_type]

                    self._markup_dataframe = pd.concat([self._markup_dataframe, 
                                                        pd.DataFrame({"video_id": [video_id],
                                                                      "frame_number": [frame_number], 
                                                                      "action_code": [action_code]})
                                          ])
                    
        self._code_to_action_types = {value: key for key, value in self._action_types_to_code.items()}
        self._markup_dataframe.index = np.arange(len(self._markup_dataframe))

        return self._markup_dataframe
        
       
class MarkupsConverterType2(MarkupsConverterType1):
    
    def __init__(self, input_dir, fps=30):
        """
        Разметка Type2 - разметка в формате xlsx.
        
        Содержит следующие столбцы:
            video_id - идентификатор видео
            label_name - имя бойца
            bbox_x - ненормированная координата x левого верхнего угла бокса
            bbox_y - ненормированная координата y левого верхнего угла бокса 
            bbox_width - ненормированная ширина бокса
            bbox_height - ненормированная высота бокса
            action_type - тип действия в текстовом виде
            frame_time - время кадра в секундах
            image_width - ширина кадра
            image_height - высота кадра
        
        Параметры
        ---------
        input_dir : str
          Папка с файлами разметки
        fps : int
          Количество кадров в секунду у видеороликов
        """
        super().__init__(input_dir=input_dir, fps=fps)
        
        
    def __init_params(self):
        self._markup_dataframe = None
        self._action_types_to_code = {}
        self._code_to_action_types = {}
        self._label_to_code = {}
        self._code_to_label = {}
        
        if not self._input_dir.exists():
            raise Exception(f"Папка {self._input_dir} не существует")

        if not self._input_dir.is_dir():
            raise Exception(f"Файл {self._input_dir} не является папкой")  
            
    @property
    def label_to_code(self):
        return self._label_to_code
    
    
    @property
    def code_to_label(self):
        return self._code_to_label
        
    @staticmethod
    def convert_coordinates_from_yolo_to_ava(x1, y1, box_width, box_height, image_width, image_height):
        """
        Параметры
        ---------
        x1 : int
          Ненормированная координата x левого верхнего угла бокса 
        y1 : int
          Ненормированная координата y левого верхнего угла бокса 
        box_width : int
          Ширина бокса
        box_height : int
          Высота бокса
        image_width : int
          Ширина изображения
        image_height : int
          Высота изображения
          
        Результат
        ---------
        x1_norm, y1_norm, x2_norm, y2_norm : tuple(float, float, float, float)
        """
        
        x1_norm = x1 / image_width
        y1_norm = y1 / image_height
        x2_norm = (x1 + box_width) / image_width
        y2_norm = (y1 + box_height) / image_height
        
        return x1_norm, y1_norm, x2_norm, y2_norm
    
    
    @staticmethod
    def convert_coordinates_from_ava_to_cv2(x1_norm, y1_norm, x2_norm, y2_norm, image_width, image_height):
        
        x1 = int(x1_norm * image_width)
        y1 = int(y1_norm * image_height)
        x2 = int(x2_norm * image_width)
        y2 = int(y2_norm * image_height)
        
        return x1, y1, x2, y2
    
          
    def convert_markups(self):
        """Конвертация разметки видео в датафрейм
        
        Результат
        ---------
        markup_dataframe : pd.DataFrame
        """
        
        self.__init_params()
        self._markup_dataframe = pd.DataFrame()
       
        for markup_file in self._input_dir.glob("*.xlsx"): 
            original_dataframe = pd.read_excel(markup_file)
            
            for _, row in original_dataframe.iterrows():
                
                x1_norm, y1_norm, x2_norm, y2_norm = self.convert_coordinates_from_yolo_to_ava(x1=row.bbox_x, 
                                                                                               y1=row.bbox_y, 
                                                                                               box_width=row.bbox_width, 
                                                                                               box_height=row.bbox_height,
                                                                                               image_width=row.image_width, 
                                                                                               image_height=row.image_height)
            
                frame_number = FFMpegOperations.calculate_frame_number_from_seconds(row.frame_time, self._fps)
                
                self._action_types_to_code.setdefault(row.action_type, len(self._action_types_to_code) + 1)
                action_code = self._action_types_to_code[row.action_type]
                
                label = row.label_name.strip().lower()
                self._label_to_code.setdefault(label, len(self._label_to_code))
                label_code = self._label_to_code[label]

                self._markup_dataframe = pd.concat([self._markup_dataframe, 
                                                    pd.DataFrame({"video_id": [row.video_id],
                                                                  "frame_time": [row.frame_time], 
                                                                  "x1_norm": [x1_norm], 
                                                                  "y1_norm": [y1_norm], 
                                                                  "x2_norm": [x2_norm], 
                                                                  "y2_norm": [y2_norm], 
                                                                  "action_code": [action_code],
                                                                  "action_name": [row.action_type],
                                                                  "label": [label_code],
                                                                  "image_width": [row.image_width],
                                                                  "image_height": [row.image_height]})
                                      ])
                    
        self._code_to_action_types = {value: key for key, value in self._action_types_to_code.items()}
        self._code_to_label = {value: key for key, value in self._label_to_code.items()}
        self._markup_dataframe.index = np.arange(len(self._markup_dataframe))

        return self._markup_dataframe
    

    def train_test_split(self, test_size=0.2):
        """ Разбиение датасета на части для обучения и валидации"""
        
        def get_filtered_dataframe(markup_dataframe, train_val_dataframe):
            filtered_data = []
            markup_dataframe = markup_dataframe.copy()
            
            for _, row in train_val_dataframe.iterrows():
                frame_time = row.frame_time
                label = row.label
                video_id = row.video_id

                filtered_dataframe = markup_dataframe[(markup_dataframe.frame_time == frame_time) &
                                                      (markup_dataframe.label == label) &
                                                      (markup_dataframe.video_id == video_id)]

                filtered_data.append(filtered_dataframe)
                
            return pd.concat(filtered_data)

        if len(self._markup_dataframe) == 0:
            return None, None
        
        prepared_df_groups = []

        for (video_id, label, frame_time), group in self._markup_dataframe.groupby(["video_id", "label", "frame_time"]):
            new_action_group_type = set(map(str, group.action_name.values.tolist()))
            new_action_group_type = tuple(sorted(list(new_action_group_type)))
            group = group.iloc[[0], :].copy()
            group["action_group_name"] = str(new_action_group_type)
            prepared_df_groups.append(group)

        modified_dataframe = pd.concat(prepared_df_groups)
        action_group_type_to_code = {}

        for index, row in modified_dataframe.iterrows():
            action_group_type_to_code.setdefault(row.action_group_name, len(action_group_type_to_code))
            action_group_code = action_group_type_to_code[row.action_group_name]
            modified_dataframe.loc[index, "action_group_code"] = action_group_code

        modified_dataframe.action_group_code = modified_dataframe.action_group_code.astype(int)
        
        action_group_statistics = modified_dataframe.action_group_code.value_counts()
        action_group_one_count = action_group_statistics[action_group_statistics <= 1]
        
        modified_dataframe = modified_dataframe[~modified_dataframe.action_group_code.isin(
            action_group_one_count.keys().to_list())]
        
        
        X_train_mod, X_val_mod = train_test_split(modified_dataframe, test_size=test_size, 
                                                  stratify=modified_dataframe["action_group_code"])
            
        X_train = get_filtered_dataframe(self._markup_dataframe, X_train_mod)
        X_val = get_filtered_dataframe(self._markup_dataframe, X_val_mod)
        
        dataframe_columns = ["video_id", "frame_time", "x1_norm", "y1_norm", "x2_norm", "y2_norm", 
                             "action_code", "label", "image_width", "image_height"]
        
        X_train = X_train.reindex(columns=dataframe_columns)
        X_val = X_val.reindex(columns=dataframe_columns)
        
        return X_train, X_val
        
def get_iou(pred_box, gt_box):
    """
    pred_box : the coordinate for predict bounding box
    gt_box :   the coordinate for ground truth bounding box
    return :   the iou score
    the  left-down coordinate of  pred_box:(pred_box[0], pred_box[1])
    the  right-up coordinate of  pred_box:(pred_box[2], pred_box[3])
    """
    # 1.get the coordinate of inters
    ixmin = max(pred_box[0], gt_box[0])
    ixmax = min(pred_box[2], gt_box[2])
    iymin = max(pred_box[1], gt_box[1])
    iymax = min(pred_box[3], gt_box[3])

    iw = np.maximum(ixmax-ixmin+1., 0.)
    ih = np.maximum(iymax-iymin+1., 0.)

    # 2. calculate the area of inters
    inters = iw*ih

    # 3. calculate the area of union
    uni = ((pred_box[2]-pred_box[0]+1.) * (pred_box[3]-pred_box[1]+1.) +
           (gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) -
           inters)

    # 4. calculate the overlaps between pred_box and gt_box
    iou = inters / uni

    return iou
    
    
class MMADatasetMaker:
    
    def __init__(self, train_df, val_df, clips_dir, output_dir, code_to_action_types, code_to_label,
                 det_config, det_checkpoint=None, det_score_thr=0.9, proposals_iou_thresh=0.5, 
                 clip_size=60, fps=30, device="cuda"):
        
        assert clip_size % fps == 0
        assert clip_size % 2 == 0
        assert  (clip_size / 2) % 2 == 0
        
        self._train_df = train_df
        self._val_df = val_df
        self._clips_dir = Path(clips_dir)
        self._output_dir = Path(output_dir)
        self._clip_size = clip_size
        self._fps = fps
        self._code_to_action_types = code_to_action_types
        self._action_type_to_code = {value: key for key, value in code_to_action_types.items()}
        self._code_to_label = code_to_label
        self._label_to_code = {value: key for key, value in code_to_label.items()}
        
        self._det_config = det_config
        self._device = device
        self._det_checkpoint = det_checkpoint
        self._det_score_thr = det_score_thr
        self._proposals_iou_thresh = proposals_iou_thresh
        
        self._clip_name_template = "frame_{}_time_{}_{}.mp4"
        self._frame_number_template = "{:" + str(FRAME_NUMBER_TEMPLATE) + "}"
        self._python_frame_template = "img_{:" + str(FRAME_NUMBER_TEMPLATE) + "}.jpg"
        
        if not self._output_dir.exists():
            os.makedirs(self._output_dir)
            
        self._final_dataframe_columns = ["video_id", "frame_time", "x1_norm", "y1_norm", "x2_norm", "y2_norm", 
                                         "action_code", "label"]
            
            
    def __prepare_folders(self):
        if self._output_dir.exists():
            shutil.rmtree(self._output_dir)
            
        self._output_videos_dir = self._output_dir / "videos"
        if not self._output_videos_dir.exists():
            os.makedirs(self._output_videos_dir)
        
        self._output_frames_dir = self._output_dir
        if not self._output_frames_dir.exists():
            os.makedirs(self._output_frames_dir)
            
        self._output_dataset_dir = self._output_dir / "dataset" 
        if not self._output_dataset_dir.exists():
            os.makedirs(self._output_dataset_dir)
            
        self._output_bboxes_dir = self._output_dir / "bboxes_samples" 
        if not self._output_bboxes_dir.exists():
            os.makedirs(self._output_bboxes_dir)
        
            
    def prepare_dataset(self):
        self.__prepare_folders()
        self._make_action_list()
        self._make_train_video_and_dataset()
        self._make_val_video_and_dataset()
        self._train_proposal_dictionary = self._make_dense_proposals(mode="train")
        self._val_proposal_dictionary = self._make_dense_proposals(mode="val")
        
        
    def _make_action_list(self):
        
        action_list_path = self._output_dataset_dir / "mma_action_list.pbtxt"
        
        person_movements = ["punch", "overturn", "take_attempt", "takedown", "takedown_attempt", "submission_attempt"]
        write_text = ""
        
        for activity_code, activity_name in self._code_to_action_types.items():
            write_text += "label {\n"
            write_text += f'  name: "{activity_name}"\n'
            write_text += f'  label_id: {activity_code}\n'
            label_type = "PERSON_MOVEMENT" if activity_name in person_movements else "PERSON_INTERACTION"
            write_text += f'  label_type: "{label_type}"\n'
            write_text += "}\n"
        
        with open(action_list_path, 'w') as file:
            file.write(write_text)
            
        action_list_path = self._output_dataset_dir / "mma_action_list.txt"
        write_text = ""
        
        for activity_code, activity_name in self._code_to_action_types.items():
            write_text += f"{activity_code}: {activity_name}\n"
        
        with open(action_list_path, 'w') as file:
            file.write(write_text)
            
        
    def _make_train_video_and_dataset(self):
        output_video = self._output_videos_dir / "HBVSMCGRG_TRAIN.mp4"
        self._final_train_df = self._make_video_and_dataset(self._train_df, output_video)
        self._extract_frames(self._output_dataset_dir, self._output_frames_dir)
        self._final_train_df.to_csv(self._output_dataset_dir / "HBVSMCGRG_TRAIN.csv", columns=self._final_dataframe_columns,
                                    sep=",", index=False, header=False)
        
    
    def _make_val_video_and_dataset(self):
        output_video = self._output_videos_dir / "HBVSMCGRG_VAL.mp4"
        self._final_val_df = self._make_video_and_dataset(self._val_df, output_video)
        self._extract_frames(self._output_videos_dir, self._output_frames_dir)
        self._final_val_df.to_csv(self._output_dataset_dir / "HBVSMCGRG_VAL.csv", columns=self._final_dataframe_columns,
                                  sep=",", index=False, header=False)

    
    def _make_video_and_dataset(self, dataframe: pd.DataFrame, output_file):
        all_clips_paths = []
        dataframe = dataframe.copy()
        dataframe_groups = []
        output_file = Path(output_file)
        
        if output_file.exists():
            os.remove(output_file)
            
        current_seconds = int((self._clip_size / 2) / self._fps)
        
        for (video_id, frame_time, label), group_df in dataframe.groupby(["video_id", "frame_time", "label"]):
            
            frame_number = FFMpegOperations.calculate_frame_number_from_seconds(frame_time, self._fps)
            frame_time_new = FFMpegOperations.calculate_seconds_from_frame_number(frame_number, self._fps)
            clip_name =  self._clip_name_template.format(frame_number, *str(frame_time_new).split("."))
            clip_path = self._clips_dir / str(video_id) / clip_name
            
            assert clip_path.exists()
            all_clips_paths.append(clip_path)
            
            new_frame_number = current_seconds 
            
            group_df.frame_time = self._frame_number_template.format(new_frame_number)
            
            group_df.video_id = output_file.stem
            dataframe_groups.append(group_df)
            
            current_seconds += int(self._clip_size / self._fps)
            
        FFMpegOperations.concat_videos(all_clips_paths, output_file)
        
        return pd.concat(dataframe_groups)
    
    
    def _extract_frames(self, output_videos_dir, output_frames_dir):
        helper = CustomAvaVideoHelper(output_videos_dir, output_frames_dir, clip_size=self._clip_size)
        helper.extract_frames()
        
        
    def print_samples(self, mode="train", samples_count=5, save_images=False):
        
        bboxes_dir = self._output_bboxes_dir
        frames_output_dir = self._output_frames_dir / "frames"
        
        if mode == "train":
            markup_dataframe = self._final_train_df
            video_id = "HBVSMCGRG_TRAIN"
        else:
            markup_dataframe = self._final_val_df
            video_id = "HBVSMCGRG_VAL"
        
        frame_groups = [(params, df) for params, df in markup_dataframe.groupby(["video_id", "frame_time", "label"])]
        random.shuffle(frame_groups)
        
        for (video_id, frame_time, label), df_group in frame_groups:
            if samples_count <= 0:
                break
                
            player_name = self._code_to_label[label]
            
            frame_time = int(frame_time) * self._fps + 1
            
            frame_path = frames_output_dir / str(video_id) / self._python_frame_template.format(int(frame_time))
            frame = cv2.imread(str(frame_path))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            frame_text_lines = [player_name]
            x1_norm = df_group.x1_norm.iloc[0]
            y1_norm = df_group.y1_norm.iloc[0]
            x2_norm = df_group.x2_norm.iloc[0]
            y2_norm = df_group.y2_norm.iloc[0]
            image_width = df_group.image_width.iloc[0]
            image_height = df_group.image_height.iloc[0]
            
            x1, y1, x2, y2 = MarkupsConverterType2.convert_coordinates_from_ava_to_cv2(x1_norm, y1_norm, x2_norm, y2_norm,
                                                                                       image_width, image_height)
            color = (255, 0, 0) if label == 0 else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
            for _, row in df_group.iterrows():
                action_type = self._code_to_action_types[row.action_code]
                frame_text_lines.append(action_type)
                
            text_y_coordinate = y1 + 10
            for text_part in frame_text_lines:
                cv2.putText(frame, text_part, (x2 + 10, text_y_coordinate), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                text_y_coordinate += 40
                
            frame_signature = f"{video_id}_{frame_time}.jpeg"
                
            if save_images:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                output_image_path = bboxes_dir / frame_signature
                cv2.imwrite(str(output_image_path), frame)
            else:
                print(video_id, frame_time)
                plt.figure(figsize=(8, 6))
                plt.imshow(frame)
                plt.show()
            
            samples_count -= 1
    
    
    def _make_dense_proposals(self, mode="train"):
        if mode == "train":
            dataframe = self._final_train_df
            pickle_file_path = self._output_dataset_dir / "ava_proposals_train.pkl"
        else:
            dataframe = self._final_val_df
            pickle_file_path = self._output_dataset_dir / "ava_proposals_val.pkl"
            
        frame_paths = []
        frame_times = []
        frame_sizes = []
        frame_coordinates = []
        video_ids = []
        frame_groups = [(params, df) for params, df in dataframe.groupby(["video_id", "frame_time", "label"])]
        
        for (video_id, frame_time, label), df_group in frame_groups:
            frame_times.append(int(frame_time))
            video_ids.append(video_id)
            frame_sizes.append((df_group.image_width.values[0], df_group.image_height.values[0]))
            central_frame_number = int(frame_time) * self._fps + 1
            central_frame_name = self._python_frame_template.format(central_frame_number)
            central_frame_path = self._output_frames_dir / "frames" / video_id / central_frame_name
            
            x1_norm = df_group.x1_norm.iloc[0]
            y1_norm = df_group.y1_norm.iloc[0]
            x2_norm = df_group.x2_norm.iloc[0]
            y2_norm = df_group.y2_norm.iloc[0]
            frame_coordinates.append((x1_norm, y1_norm, x2_norm, y2_norm))
            
            assert central_frame_path.exists(), f"Кадр {central_frame_path} не существует"
            frame_paths.append(central_frame_path)
            
        detector_results = self._detection_inference(frame_paths)
        assert sum([len(bboxes) > 0 
                    for bboxes in detector_results]) == len(frame_paths), "Детектор не смог найти людей на всех фото"
        
        proposal_dictionary = {}
        
        for idx in range(len(detector_results)):
            frame_time = frame_times[idx]
            video_id = video_ids[idx]
            frame_detector_results = detector_results[idx]
            image_width, image_height = frame_sizes[idx]
            gt_box = frame_coordinates[idx]
            frame_signature = f"{video_id},{self._frame_number_template.format(frame_time)}"
            
            for idx, detector_result in enumerate(frame_detector_results, 1):
                x1, y1, x2, y2, conf = detector_result
                x1, x2 = x1 / image_width, x2 / image_width
                y1, y2 = y1 / image_height, y2 / image_height
                pred_box = (x1, y1, x2, y2)
                iou = get_iou(pred_box, gt_box)
                
                if iou < self._proposals_iou_thresh:
                    if idx == len(frame_detector_results) and proposal_dictionary.get(frame_signature) is None:
                        proposal_dictionary.setdefault(frame_signature, []).append([*gt_box, 1.0]) 
                        
                    continue
                    
                proposal_dictionary.setdefault(frame_signature, []).append([x1,y1,x2,y2,conf])
         
        assert len(proposal_dictionary) == len(frame_paths), "Для некоторых кадров отсутствуют proposals"

        for key in proposal_dictionary.keys():
            proposal_dictionary[key] = np.array(proposal_dictionary[key])
            
        with open(pickle_file_path, 'wb') as handle:
            pickle.dump(proposal_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        return proposal_dictionary
    
    def _detection_inference(self, frame_paths):
        """Получение боксов людей на изображениях 
        
        Параметры
        ---------
        frame_paths : (list[str])
          Пути к фреймам
    
        Результат
        ---------
        list[np.ndarray]
          Результаты детеккции
        """
        model = init_detector(self._det_config, self._det_checkpoint, self._device)
        assert model.CLASSES[0] == 'person', ("Нужен детектор, обученный на COCO")
        results = []

        prog_bar = mmcv.ProgressBar(len(frame_paths))
        for frame_path in frame_paths:
            result = inference_detector(model, frame_path)
            result = result[0][result[0][:, 4] >= self._det_score_thr]
            results.append(result)
            prog_bar.update()
            
        return results