import numpy as np
import math
from path import Path
import random
import os
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import plotly.graph_objects as go
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


def read_off(file):
    if 'OFF' != file.readline().strip():
        raise('Not a valid OFF header')
    n_verts, n_faces, __ = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return verts, faces


def visualize_rotate(data):
    x_eye, y_eye, z_eye = 1.25, 1.25, 0.8
    frames=[]

    def rotate_z(x, y, z, theta):
        w = x+1j*y
        return np.real(np.exp(1j*theta)*w), np.imag(np.exp(1j*theta)*w), z

    for t in np.arange(0, 10.26, 0.1):
        xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
        frames.append(dict(layout=dict(scene=dict(camera=dict(eye=dict(x=xe, y=ye, z=ze))))))

    fig = go.Figure(data=data,
                    layout=go.Layout(
                        updatemenus=[dict(type='buttons',
                                          showactive=False,
                                    y=1,
                                    x=0.8,
                                    xanchor='left',
                                    yanchor='bottom',
                                    pad=dict(t=45, r=10),
                                    buttons=[dict(label='Play',
                                                  method='animate',
                                                    args=[None, dict(frame=dict(duration=50, redraw=True),
                                                                     transition=dict(duration=0),
                                                                    fromcurrent=True,
                                                                    mode='immediate'
                                                                    )]
                                                    )
                                             ]
                                    )
                                     ]
                        ),
                    frames=frames
                    )

    return fig


def pcshow(xs,ys,zs):
    data=[go.Scatter3d(x=xs, y=ys, z=zs,
                                   mode='markers')]
    fig = visualize_rotate(data)
    fig.update_traces(marker=dict(size=2,
                      line=dict(width=2,
                      color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
    fig.show()
    

class PointSampler(object):
    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size

    def triangle_area(self, pt1, pt2, pt3):
        side_a = np.linalg.norm(pt1 - pt2)
        side_b = np.linalg.norm(pt2 - pt3)
        side_c = np.linalg.norm(pt3 - pt1)
        s = 0.5 * ( side_a + side_b + side_c)
        return max(s * (s - side_a) * (s - side_b) * (s - side_c), 0)**0.5

    def sample_point(self, pt1, pt2, pt3):
        # barycentric coordinates on a triangle
        # https://mathworld.wolfram.com/BarycentricCoordinates.html
        s, t = sorted([random.random(), random.random()])
        f = lambda i: s * pt1[i] + (t-s)*pt2[i] + (1-t)*pt3[i]
        return (f(0), f(1), f(2))


    def __call__(self, mesh):
        verts, faces = mesh
        verts = np.array(verts)
        areas = np.zeros((len(faces)))

        for i in range(len(areas)):
            areas[i] = (self.triangle_area(verts[faces[i][0]],
                                           verts[faces[i][1]],
                                           verts[faces[i][2]]))

        sampled_faces = (random.choices(faces, 
                                        weights=areas,
                                      cum_weights=None,
                                      k=self.output_size))

        sampled_points = np.zeros((self.output_size, 3))

        for i in range(len(sampled_faces)):
            sampled_points[i] = (self.sample_point(verts[sampled_faces[i][0]],
                                                   verts[sampled_faces[i][1]],
                                                   verts[sampled_faces[i][2]]))
            
        return sampled_points


class Normalize(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2
        
        norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0) 
        norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))

        return  norm_pointcloud


class RandRotation_z(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        theta = random.random() * 2. * math.pi
        rot_matrix = np.array([[ math.cos(theta), -math.sin(theta),    0],
                               [ math.sin(theta),  math.cos(theta),    0],
                               [0,                             0,      1]])

        rot_pointcloud = rot_matrix.dot(pointcloud.T).T
        return  rot_pointcloud


class RandomNoise(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        noise = np.random.normal(0, 0.02, (pointcloud.shape))

        noisy_pointcloud = pointcloud + noise
        return  noisy_pointcloud

    
class ToTensor(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        return torch.from_numpy(pointcloud).type(torch.float32)
    
    
def default_transforms():
    return transforms.Compose([
                                PointSampler(1024),
                                Normalize(),
                                ToTensor()
                              ])

    
class PointCloudDataset(Dataset):
    """ Датасет для данных http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip
    
    Структура папок датасета:
    
      класс1
        train
        test
      класс2
        train
        test
      .
      .
      .
      класс10
        train
        test
    
    Параметры
    ---------
    root_dir : str
      Путь к распакованному архиву ModelNet10.zip
    valid : bool
      Если True, будут сформированы данные для валидации. Иначе для обучения
    transform : torchvision.transforms.Compose
      Если указаны преобразования, то в случае valid=False будут использованы
      указанные преобразования. Если valid=True или transform=None,
      то будут выполнены преобразования из функции default_transforms
    """
    def __init__(self, root_dir, valid=False, transform=None):
        
        # В зависимости от типа датасета выбираем папку с данными в рамках класса
        if valid:
            folder = 'test'
        else:
            folder = 'train'
            
        root_dir = Path(root_dir)
        
        # Находим все папки у датасета с именами классов
        folders = [class_dirname for class_dirname in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir/class_dirname)]
        # Формируем словарь соответствий имени класса его номеру
        self._classes = {class_dirname: idx for idx, class_dirname in enumerate(folders)}
        # Определяем преобразования для датасета
        self._transforms = transform if not valid or transform is not None else default_transforms()
        
        self._valid = valid
        
        self._files = []
        
        for category_name in self._classes.keys():
            new_dir = root_dir/Path(category_name)/folder
            for file in os.listdir(new_dir):
                if file.endswith('.off'):
                    sample = {}
                    sample['pcd_path'] = new_dir/file
                    sample['category_name'] = category_name
                    self._files.append(sample)
                    
    @property
    def valid(self):
        return self._valid

    def __len__(self):
        return len(self._files)

    def __preproc__(self, file):
        verts, faces = read_off(file)
        pointcloud = self._transforms((verts, faces))
        return pointcloud

    def __getitem__(self, idx):
        pcd_path = self._files[idx]['pcd_path']
        category_name = self._files[idx]['category_name']
        with open(pcd_path, 'r') as f:
            pointcloud = self.__preproc__(f)
        return {'pointcloud': pointcloud, 
                'target': self._classes[category_name],
                'pcd_path': pcd_path,
                'category_name': category_name}
    
    
class PointCloudDatamodule(pl.LightningDataModule):
    """ Модуль для загрузки данных в модель
    
    Параметры
    ---------
    root_dir : str
      Путь к распакованному архиву ModelNet10.zip
    valid : bool
      Если True, будут сформированы данные для валидации. Иначе для обучения
    train_transforms : torchvision.transforms.Compose
      Преобразования для обучения. 
      Если None, то будут использованы преобразования из функции default_transforms
    train_loader_params : dict
      Словарь с параметрами загрузчика для обучающего датасета
    val_loader_params : dict
      Словарь с параметрами загрузчика для валидационного датасета
    seed : int
      Фиксация генератора случайных чисел
    """
    
    def __init__(self, root_dir, train_transforms=None, train_loader_params=None, val_loader_params=None, seed=None):
    
        super().__init__()
    
        if seed is not None:
            set_seed(seed)
    
        self._root_dir = root_dir
    
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
    
        self.train_loader_params = train_loader_params
        self.val_loader_params = val_loader_params
        self._train_transforms = train_transforms
    
        self.make_split_dict()
    
    def make_split_dict(self):
        self.train_dataset = PointCloudDataset(root_dir=self._root_dir, valid=False, transform=self._train_transforms)
        self.val_dataset = PointCloudDataset(root_dir=self._root_dir, valid=True)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_loader_params['batch_size'],
                          shuffle=self.train_loader_params['shuffle'], drop_last=self.train_loader_params['drop_last'],
                          num_workers=self.train_loader_params['num_workers'])

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_loader_params['batch_size'],
                          drop_last=self.val_loader_params['drop_last'], shuffle=self.val_loader_params['shuffle'],
                          num_workers=self.val_loader_params['num_workers'])
    
    
class TNet(nn.Module):
    """ Класс для реализации блока T-Net модели PointNet
    
    Параметры
    ---------
    k : int
      Размер квадратной матрицы преобразования
    """
    def __init__(self, k=3):
        super().__init__()
        
        self.k=k
        
        # Реализация блока MLP с помощью одномерной свертки
        # Одномерная светрка будет применяться индивидуально к каждой координате в батче
        # За счет использования ядра размером 1 мы получаем аналог полносвязной сети на базе свертки
        self.conv1 = nn.Conv1d(k,64,1)
        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,1024,1)
        
        self.fc1 = nn.Linear(1024,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,k*k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, input_data):
        """ Прямой проход модели
        
        Параметры
        ---------
        input_data : torch.tensor
          Тензор размером (кол-во батчей, кол-во координат, кол-во точек)
        
        Результат
        ---------
        matrix : torch.tensor
          Матрица поворота размером (batch_size, k, k)
        """
        
        # Запоминаем размер батча
        bs = input_data.size(0)
        
        # Выполняем преобразование MLP
        xb = F.relu(self.bn1(self.conv1(input_data)))
        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = F.relu(self.bn3(self.conv3(xb)))
        
        # Максимальное значение по 1024 фильтрам, которые вышли из сверток (3 -> 64 -> 128 -> 1024)
        # Берем размер ядра по количеству точек в батче
        pool = nn.MaxPool1d(xb.size(-1))(xb)
        # Сжимаем даныне в один вектор
        flat = nn.Flatten(1)(pool)
        # Применяем полносвязные слои, кроме последнего
        xb = F.relu(self.bn4(self.fc1(flat)))
        xb = F.relu(self.bn5(self.fc2(xb)))

        # Формируем матрицу поворота
        # Инициализируем матрицу поворота, как единичную, идентичное преобразование
        # И уже к этой идентичной матрице поворота будем прибавлять значения нейросети
        # С помощью repeat дублируем единичную матрицу для каждого батча
        init = torch.eye(self.k, requires_grad=True).repeat(bs,1,1)
        if xb.is_cuda:
            init=init.cuda()
        
        # Получаем матрицы поворота для каждого батча
        # Применяем последний полносвязный слой и приводим его к размеру (batch_size, k, k)
        # И добавляем к нему единичную матрицу
        matrix = self.fc3(xb).view(-1,self.k,self.k) + init
        
        return matrix


class Transform(nn.Module):
    """ Модель трансформации, реализует блок сети классификации PointNet
    начиная от обработки входа до получения эмбеддингов после блока MaxPool"""
    
    def __init__(self):
        super().__init__()
        
        # Первый блок T-Net с матрицей поворота 3x3
        self.input_transform = TNet(k=3)
        # Второй блок T-Net с матрицей поворота 64x64
        self.feature_transform = TNet(k=64)
        
        # Одномерные свертки для реализации второго блока MLP после 
        # преобразований с матрицами поворота 64x64
        self.conv1 = nn.Conv1d(3,64,1)
        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,1024,1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, input_data):
        """ Прямой проход модели
        
        Параметры
        ---------
        input_data : torch.tensor
          Тензор размером (batch_size, кол-во точек, кол-во координат)
        
        Результат
        ---------
        embeddings, matrix3x3, matrix64x64, crit_points : (torch.tensor, torch.tensor, torch.tensor, torch.tensor)
          embeddings - ембединги облака точек размером (batch_size, 1024)
          matrix3x3 - матрица поворота размером (batch_size, 3, 3)
          matrix64x64 - матрица поворота размером (batch_size, 64, 64)
          crit_points - вектор индексов критических точек
        """
        
        # Получаем первую матрицу поворота 3x3
        # Через permute конвертируем размер из (кол-во батчей, кол-во точек, кол-во координат) в (кол-во батчей, кол-во координат, кол-во точек)
        # Это особенность работы Conv1D
        matrix3x3 = self.input_transform(input_data.permute(0,2,1))
        
        # Применяем матрицу поворота к нашим данным
        xb = torch.bmm(input_data, matrix3x3)
        
        # Приводим формат (кол-во батчей, кол-во точек, кол-во координат) к формату (кол-во батчей, кол-во координат, кол-во точек)
        # такой формат требует Conv1d 
        xb = xb.permute(0, 2, 1)
        
        # Здесь реализован первый MLP в схеме (64, 64)
        xb = F.relu(self.bn1(self.conv1(xb)))

        # Получаем вторую матрицу поворота для второго блока T-Net
        # У нас уже формат (кол-во батчей, кол-во координат, кол-во точек), permute не требуется
        matrix64x64 = self.feature_transform(xb)
        
        # Применяем матрицу поворота
        # Возвращаем размер (кол-во батчей, кол-во точек, кол-во фичей)  с помощью permute в операции перемножения матриц
        # и переходим к ((кол-во батчей, кол-во фичей, кол-во точек)  для использования дальше в Conv1D
        xb = torch.bmm(xb.permute(0, 2, 1), matrix64x64).permute(0, 2, 1)

        # Здесь реализован второй MLP в схеме (64, 128, 1024)
        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = self.bn3(self.conv3(xb))
        
        # Находим индексы критических точек
        crit_points = torch.argmax(xb, axis=1).detach().flatten()
        
        # Реализация блока MaxPool сети классификатора
        # Находим максимальную фичу для каждой координаты
        xb = nn.MaxPool1d(xb.size(-1))(xb)
        # Схлопываем данные в один вектор в рамках каждого батча, получаем эмбединги для каждого облака точек
        output = nn.Flatten(1)(xb)
        
        # Возвращаем ембединги, матрицу поворотов 3x3 и 64x64
        # Матрицы поворота потребуются для рассчета ошибки
        return output, matrix3x3, matrix64x64, crit_points
    

class PointNet(pl.LightningModule):
    """ Реализация сети классификатора модели PointNet
    
    Параметры
    ---------
    classes : int
      Кол-во классов
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
    def __init__(self, classes=10, dropout=0.5, learning_rate=0.01, l2_regularization=1e-3, adam_betas=(0.9, 0.999), plot_epoch_loss=True, seed=None):
        
        super().__init__()
        
        if seed is not None:
            set_seed(seed)        
        
        self.learning_rate = learning_rate
        self.l2_regularization = l2_regularization
        self.adam_betas = adam_betas
        
        self.transform = Transform()
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, classes)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=dropout)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
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
          Тензор размером (кол-во батчей, кол-во точек, кол-во координат)
        
        Результат
        ---------
        logits, matrix3x3, matrix64x64, embeddings, crit_points : (torch.tensor,)
          logits - логиты предсказаний классов
          matrix3x3 - матрица поворота размером (batch_size, 3, 3)
          matrix64x64 - матрица поворота размером (batch_size, 64, 64)
          embeddings - ембединги облака точек размером (batch_size, 1024)
          crit_points - вектор индексов критических точек
        """
        
        # Получаем эмбеддинги и матрицы поворота
        embeddings, matrix3x3, matrix64x64, crit_points = self.transform(input_data)
        
        # Полносвязная сеть с выходном размера количества классов
        xb = F.relu(self.bn1(self.fc1(embeddings)))
        xb = F.relu(self.bn2(self.dropout(self.fc2(xb))))
        output = self.fc3(xb)
        
        # Возвращаем предсказания классов и матрицы поворота, которые потребуются для расчета ошибки
        return self.logsoftmax(output), matrix3x3, matrix64x64, embeddings, crit_points
    
    def pointnet_loss(self, outputs, labels, m3x3, m64x64, alpha = 0.0001):
        """ Расчет ошибки для модели PointNet
        
        Ошибка складывается из двух частей. 
        Первая часть - ошибка negative log likelihood loss
        Вторая часть - ошибка насколько получаемые матрицы поворота из сети соответствуют реальной структуре матрицы поворота
        
        Вторая часть ошибки расчитывается исходя из следющих тезисов:
        - если матрицу поворота умножить на ее саму транспонированную, то должна получиться единичная матрица
        - отклонения матрицы поворота от единичной можно посчитать с помощью формулы eye_mat - (net_matrix @ net_matrix.T)
        - чем больше матрицы поворота из сети соответствуют реальной структуре матриц поворота, тем ниже будет норма матрицы отклонений и меньше итоговая ошибка
        
        Параметры
        ---------
        outputs : torch.tensor
          Предсказания классов размером (batch_size, )
        labels : torch.tensor
          Ground-truth метки размером (batch_size, )
        m3x3 : torch.tensor
          Матрица поворота  размером (batch_size, 3, 3)
        m64x64 : torch.tensor
          Матрица поворота размером (batch_size, 64, 64)
        alpha : float
          Коэффициент для учета в общем loss ошибки матриц поворота.
          По умолчанию 0.0001
          
        Результат
        ---------
        loss : torch.variable
        """
        
        # Первая часть ошибки
        criterion = torch.nn.NLLLoss()
        
        # Вторая часть ошибки
        # Получаем образцовые единичные матрицы для каждого батча
        bs=outputs.size(0)
        id3x3 = torch.eye(3, requires_grad=True).repeat(bs,1,1)
        id64x64 = torch.eye(64, requires_grad=True).repeat(bs,1,1)
        if outputs.is_cuda:
            id3x3=id3x3.cuda()
            id64x64=id64x64.cuda()
            
        # Находим разницы по формуле eye_mat - (net_matrix @ net_matrix.T) для каждой матрицы поворота сети
        diff3x3 = id3x3-torch.bmm(m3x3,m3x3.transpose(1,2))
        diff64x64 = id64x64-torch.bmm(m64x64,m64x64.transpose(1,2))
        
        # Считаем общую ошибку
        loss = criterion(outputs, labels) + alpha * (torch.norm(diff3x3) + torch.norm(diff64x64)) / float(bs)    
        
        return loss

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


        outputs, matrix3x3, matrix64x64, *_ = self(batch['pointcloud'])
        target_labels = batch['target']
        loss = self.pointnet_loss(outputs, labels=target_labels, m3x3=matrix3x3, m64x64=matrix64x64)
        self.log(f'{mode}_loss', loss, prog_bar=True)

        predict_labels = torch.argmax(outputs.cpu().detach(), axis=1).numpy()

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