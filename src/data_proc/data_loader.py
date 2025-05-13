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
    
    def load_video(self, video_path):
        """Загрузка видео с оптимизацией памяти"""
        try:
            print(f"[DEBUG] Загрузка видео: {os.path.basename(video_path)}")
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Не удалось открыть видео: {video_path}")
            
            # Получаем информацию о видео
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"  - Размер: {width}x{height}")
            print(f"  - FPS: {fps}")
            print(f"  - Количество кадров: {total_frames}")
            
            # Очищаем память перед загрузкой кадров
            gc.collect()
            
            frames = []
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Обрабатываем каждый N-й кадр для экономии памяти
                if frame_count % 2 == 0:  # Берем каждый второй кадр
                    # Изменяем размер кадра сразу при загрузке
                    frame = cv2.resize(frame, (224, 224))
                    frames.append(frame)
                    
                    # Очищаем память каждые 100 кадров
                    if len(frames) % 100 == 0:
                        gc.collect()
                    
                frame_count += 1
            
            cap.release()
            
            # Преобразуем в numpy массив с оптимизированным типом данных
            frames = np.array(frames, dtype=np.float32) / 255.0
            
            print(f"[DEBUG] Загружено {len(frames)} кадров")
            return frames
            
        except Exception as e:
            print(f"[ERROR] Ошибка при загрузке видео: {str(e)}")
            raise
    
    def create_sequences(self, frames, annotations):
        """Создание последовательностей с оптимизацией памяти"""
        try:
            sequences = []
            labels = []
            
            # Очищаем память перед созданием последовательностей
            gc.collect()
            
            # Проверяем, что аннотации существуют
            if annotations is None:
                print("[WARNING] Аннотации не найдены, создаем пустые метки")
                annotations = np.zeros((len(frames), Config.NUM_CLASSES), dtype=np.float32)
            else:
                # Загружаем аннотации из JSON файла
                try:
                    with open(annotations, 'r') as f:
                        ann_data = json.load(f)
                        # Преобразуем аннотации в numpy массив
                        annotations = np.array(ann_data['labels'], dtype=np.float32)
                except Exception as e:
                    print(f"[ERROR] Ошибка при загрузке аннотаций: {str(e)}")
                    print("[WARNING] Создаем пустые метки")
                    annotations = np.zeros((len(frames), Config.NUM_CLASSES), dtype=np.float32)
            
            # Проверяем размерности
            if len(annotations) != len(frames):
                print(f"[WARNING] Несоответствие размерностей: frames={len(frames)}, annotations={len(annotations)}")
                # Обрезаем до минимальной длины
                min_len = min(len(frames), len(annotations))
                frames = frames[:min_len]
                annotations = annotations[:min_len]
            
            for i in range(0, len(frames) - self.sequence_length + 1, self.sequence_length // 2):
                sequence = frames[i:i + self.sequence_length]
                sequence_labels = annotations[i:i + self.sequence_length]
                
                sequences.append(sequence)
                labels.append(sequence_labels)
                
                # Очищаем память каждые 10 последовательностей
                if len(sequences) % 10 == 0:
                    gc.collect()
            
            # Преобразуем в numpy массивы с оптимизированным типом данных
            sequences = np.array(sequences, dtype=np.float32)
            labels = np.array(labels, dtype=np.float32)
            
            print(f"[DEBUG] Создано {len(sequences)} последовательностей")
            print(f"[DEBUG] Форма последовательностей: {sequences.shape}")
            print(f"[DEBUG] Форма меток: {labels.shape}")
            
            return sequences, labels
            
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
        self.load_video(video_path)
    
    def data_generator(self):
        """Генератор данных с оптимизацией памяти"""
        try:
            while True:
                for video_path in self.video_paths:
                    try:
                        # Загружаем видео
                        frames = self.load_video(video_path)
                        
                        # Получаем аннотации
                        annotations = self.labels[self.video_paths.index(video_path)]
                        
                        # Создаем последовательности
                        sequences, labels = self.create_sequences(frames, annotations)
                        
                        # Очищаем память после обработки видео
                        del frames
                        gc.collect()
                        
                        # Возвращаем последовательности батчами
                        for i in range(0, len(sequences), self.batch_size):
                            batch_sequences = sequences[i:i + self.batch_size]
                            batch_labels = labels[i:i + self.batch_size]
                            
                            yield batch_sequences, batch_labels
                            
                            # Очищаем память после каждого батча
                            del batch_sequences
                            del batch_labels
                            gc.collect()
                            
                    except Exception as e:
                        print(f"[ERROR] Ошибка при обработке видео {video_path}: {str(e)}")
                        continue
                    
        except Exception as e:
            print(f"[ERROR] Ошибка в генераторе данных: {str(e)}")
            raise
    
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
        return self.data_generator() 