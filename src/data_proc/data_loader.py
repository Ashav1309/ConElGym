import cv2
import numpy as np
from typing import Tuple, List, Generator
import os
import json
from src.data_proc.annotation import VideoAnnotation
from src.config import Config
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor
import threading
from src.utils.network_handler import NetworkErrorHandler, NetworkMonitor
import logging
import gc

logger = logging.getLogger(__name__)

class VideoDataLoader:
    def __init__(self, data_path):
        """
        Инициализация загрузчика данных.
        
        Args:
            data_path (str): Путь к директории с данными
        """
        self.data_path = data_path
        self.video_paths = []
        self.labels = []
        self.network_handler = NetworkErrorHandler()
        self.network_monitor = NetworkMonitor()
        self.cache = {}
        self.cache_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Инициализация параметров из конфигурации
        self.sequence_length = Config.SEQUENCE_LENGTH
        self.max_sequences_per_video = Config.MAX_SEQUENCES_PER_VIDEO
        
        self._load_data()
    
    def _load_data(self, infinite_loop=False):
        """
        Загрузка путей к видео (без классов, без подкаталогов) и соответствующих аннотаций.
        
        Args:
            infinite_loop (bool): Бесконечный цикл загрузки данных
            
        Raises:
            FileNotFoundError: Если директория с данными не найдена
            ValueError: Если нет видео файлов в директории
        """
        try:
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"Директория с данными не найдена: {self.data_path}")
            
            annotation_dir = os.path.join(self.data_path, 'annotations')
            if not os.path.exists(annotation_dir):
                print(f"[DEBUG] Создание директории для аннотаций: {annotation_dir}")
                os.makedirs(annotation_dir, exist_ok=True)
            
            print(f"[DEBUG] Поиск видео в {self.data_path}, аннотаций в {annotation_dir}")
            
            video_count = 0
            while True:
                for file_name in os.listdir(self.data_path):
                    file_path = os.path.join(self.data_path, file_name)
                    if file_name.endswith('.mp4') and os.path.isfile(file_path):
                        video_count += 1
                        self.video_paths.append(file_path)
                        base = os.path.splitext(file_name)[0]
                        ann_path = os.path.join(annotation_dir, base + '.json')
                        if os.path.exists(ann_path):
                            print(f"[DEBUG] Найдена аннотация для {file_name}")
                        else:
                            print(f"[DEBUG] Аннотация для {file_name} не найдена")
                        self.labels.append(ann_path if os.path.exists(ann_path) else None)
                
                if not infinite_loop:
                    break
                
            if video_count == 0:
                raise ValueError(f"В директории {self.data_path} не найдено видео файлов")
            
            print(f"[DEBUG] Загружено {video_count} видео файлов")
            
        except Exception as e:
            print(f"[ERROR] Ошибка при загрузке данных: {str(e)}")
            print("[DEBUG] Stack trace:", flush=True)
            import traceback
            traceback.print_exc()
            raise
    
    def load_video(self, video_path, target_size=None):
        """
        Загрузка видео с обработкой сетевых ошибок
        
        Args:
            video_path (str): Путь к видео файлу
            target_size (tuple): Размер изображения (ширина, высота)
            
        Returns:
            list: Список кадров видео
            
        Raises:
            IOError: Если не удалось открыть видео
            ValueError: Если видео имеет неподдерживаемый формат
        """
        def _load_video_operation():
            frames = []
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise IOError(f"Не удалось открыть видео: {video_path}")
            
            try:
                # Проверка формата видео
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                if width == 0 or height == 0 or fps == 0 or frame_count == 0:
                    raise ValueError(f"Неподдерживаемый формат видео: {video_path}")
                
                print(f"[DEBUG] Загрузка видео: {os.path.basename(video_path)}")
                print(f"  - Размер: {width}x{height}")
                print(f"  - FPS: {fps}")
                print(f"  - Количество кадров: {frame_count}")
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if target_size:
                        frame = cv2.resize(frame, target_size)
                    frame = frame.astype(np.float32) / 255.0
                    frames.append(frame)
                
                if len(frames) == 0:
                    raise ValueError(f"Не удалось прочитать кадры из видео: {video_path}")
                
                print(f"[DEBUG] Загружено {len(frames)} кадров")
                return frames
                
            finally:
                cap.release()
            
        return self.network_handler.handle_network_operation(_load_video_operation)
    
    def create_sequences(self, frames, annotation_path, sequence_length, one_hot=False, max_sequences_per_video=10):
        """
        Создание последовательностей кадров и меток на основе аннотации.
        
        Args:
            frames (list): Список кадров видео
            annotation_path (str): Путь к файлу аннотации
            sequence_length (int): Длина последовательности
            one_hot (bool): Использовать one-hot encoding для меток
            max_sequences_per_video (int): Максимальное количество последовательностей
            
        Returns:
            tuple: (sequences, labels) - массивы последовательностей и меток
            
        Raises:
            ValueError: Если входные данные некорректны
        """
        try:
            # Валидация входных данных
            if not isinstance(frames, (list, np.ndarray)):
                raise ValueError(f"frames должен быть списком или numpy.ndarray, получен {type(frames)}")
            
            if len(frames) == 0:
                raise ValueError("frames не может быть пустым")
            
            if not isinstance(sequence_length, int) or sequence_length <= 0:
                raise ValueError(f"sequence_length должен быть положительным целым числом, получено {sequence_length}")
            
            if not isinstance(max_sequences_per_video, int) or max_sequences_per_video <= 0:
                raise ValueError(f"max_sequences_per_video должен быть положительным целым числом, получено {max_sequences_per_video}")
            
            print(f"[DEBUG] Создание последовательностей: frames={len(frames)}, annotation={annotation_path}")
            
            # Инициализация меток
            labels = [0] * len(frames)
            
            # Загрузка аннотаций
            if annotation_path and os.path.exists(annotation_path):
                try:
                    with open(annotation_path, 'r') as f:
                        ann = json.load(f)
                        if "annotations" in ann and len(ann["annotations"]) > 0:
                            for elem in ann["annotations"]:
                                start = elem.get('start_frame', 0)
                                end = elem.get('end_frame', 0)
                                print(f"[DEBUG] Аннотация: start_frame={start}, end_frame={end}")
                                for i in range(start, end + 1):
                                    if 0 <= i < len(labels):
                                        labels[i] = 1
                        else:
                            print(f"[DEBUG] Нет элементов в annotations для {annotation_path}")
                except Exception as e:
                    print(f"[ERROR] Ошибка чтения аннотации {annotation_path}: {str(e)}")
                    print("[DEBUG] Stack trace:", flush=True)
                    import traceback
                    traceback.print_exc()
            else:
                print(f"[DEBUG] Нет аннотации для видео, все метки = 0")
            
            # Создание последовательностей
            sequences = []
            sequence_labels = []
            max_seq = min(len(frames) - sequence_length + 1, max_sequences_per_video)
            
            for i in range(max_seq):
                sequence = frames[i:i + sequence_length]
                seq_labels = labels[i:i + sequence_length]
                sequences.append(sequence)
                
                if one_hot:
                    # Преобразуем метки в one-hot формат
                    one_hot_labels = np.zeros((sequence_length, 2))
                    for j, label in enumerate(seq_labels):
                        one_hot_labels[j, label] = 1
                    sequence_labels.append(one_hot_labels)
                else:
                    sequence_labels.append(seq_labels)
                
            print(f"[DEBUG] Сформировано {len(sequences)} последовательностей (ограничение: {max_sequences_per_video})")
            return np.array(sequences), np.array(sequence_labels)
            
        except Exception as e:
            print(f"[ERROR] Ошибка при создании последовательностей: {str(e)}")
            print("[DEBUG] Stack trace:", flush=True)
            import traceback
            traceback.print_exc()
            raise
    
    def preload_video(self, video_path, target_size):
        """
        Предварительная загрузка видео в отдельном потоке.
        """
        self.load_video(video_path, target_size)
    
    def data_generator(self, batch_size=32, shuffle=True):
        """
        Создает tf.data.Dataset для обучения модели
        Args:
            batch_size: размер батча
            shuffle: перемешивать ли данные
        """
        print("[DEBUG] Запуск генератора данных...")
        
        # Создаем списки для хранения всех последовательностей и меток
        all_sequences = []
        all_labels = []
        
        # Обрабатываем все видео один раз
        for video_path in self.video_paths:
            try:
                print(f"[DEBUG] Обработка видео: {os.path.basename(video_path)}")
                
                # Загружаем видео
                frames = self.load_video(video_path)
                if frames is None or len(frames) == 0:
                    continue
                
                # Получаем аннотации
                annotation_path = self.labels[self.video_paths.index(video_path)]
                if not os.path.exists(annotation_path):
                    continue
                
                # Создаем последовательности
                sequences, labels = self.create_sequences(
                    frames,
                    annotation_path,
                    sequence_length=self.sequence_length,
                    one_hot=True,
                    max_sequences_per_video=self.max_sequences_per_video
                )
                
                if len(sequences) > 0:
                    all_sequences.extend(sequences)
                    all_labels.extend(labels)
                
            except Exception as e:
                print(f"[ERROR] Ошибка при обработке видео {video_path}: {str(e)}")
                continue
        
        # Преобразуем в numpy массивы
        all_sequences = np.array(all_sequences)
        all_labels = np.array(all_labels)
        
        print(f"[DEBUG] Всего создано {len(all_sequences)} последовательностей")
        
        # Создаем tf.data.Dataset
        dataset = tf.data.Dataset.from_tensor_slices((all_sequences, all_labels))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(all_sequences))
        
        # Разбиваем на батчи и оптимизируем производительность
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def load_data(self, sequence_length, batch_size, target_size=None, one_hot=False, infinite_loop=False, max_sequences_per_video=10):
        """
        Загрузка данных для обучения.
        
        Args:
            sequence_length (int): Длина последовательности
            batch_size (int): Размер батча
            target_size (tuple): Размер изображения (ширина, высота)
            one_hot (bool): Использовать one-hot encoding для меток
            infinite_loop (bool): Бесконечный цикл генерации данных
            
        Returns:
            generator: Генератор данных
        """
        return self.data_generator(batch_size, True) 