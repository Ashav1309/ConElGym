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
        self._load_data()
        self.cache = {}
        self.cache_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def _load_data(self, infinite_loop=False):
        """
        Загрузка путей к видео и меток.
        """
        while True:
            for class_name in os.listdir(self.data_path):
                class_path = os.path.join(self.data_path, class_name)
                if os.path.isdir(class_path):
                    for video_name in os.listdir(class_path):
                        if video_name.endswith('.mp4'):
                            self.video_paths.append(os.path.join(class_path, video_name))
                            self.labels.append(1 if class_name == 'correct' else 0)
            if not infinite_loop:
                break
    
    def load_video(self, video_path, target_size=None):
        """
        Загрузка видео и извлечение кадров с кэшированием.
        
        Args:
            video_path (str): Путь к видео файлу
            target_size (tuple): Размер изображения (ширина, высота)
            
        Returns:
            list: Список кадров
        """
        cache_key = f"{video_path}_{target_size}"
        
        with self.cache_lock:
            if cache_key in self.cache:
                return self.cache[cache_key]
        
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Изменение размера изображения, если указано
            if target_size:
                frame = cv2.resize(frame, target_size)
            
            # Нормализация
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)
        
        cap.release()
        
        with self.cache_lock:
            self.cache[cache_key] = frames
        
        return frames
    
    def create_sequences(self, frames, labels, sequence_length, target_size=None, one_hot=False):
        """
        Создание последовательностей кадров.
        
        Args:
            frames (list): Список кадров
            labels (list): Список меток
            sequence_length (int): Длина последовательности
            target_size (tuple): Размер изображения (ширина, высота)
            one_hot (bool): Использовать one-hot encoding для меток
            
        Returns:
            tuple: (последовательности, метки последовательностей)
        """
        sequences = []
        sequence_labels = []
        
        for i in range(len(frames) - sequence_length + 1):
            sequence = frames[i:i + sequence_length]
            sequences.append(sequence)
            
            if one_hot:
                label = np.zeros(Config.NUM_CLASSES)
                label[labels[i]] = 1
                sequence_labels.append(label)
            else:
                sequence_labels.append(labels[i])
        
        return np.array(sequences), np.array(sequence_labels)
    
    def preload_video(self, video_path, target_size):
        """
        Предварительная загрузка видео в отдельном потоке.
        """
        self.load_video(video_path, target_size)
    
    def data_generator(self, sequence_length, batch_size, target_size=None, one_hot=False):
        """
        Генератор данных для обучения с оптимизированной загрузкой.
        
        Args:
            sequence_length (int): Длина последовательности
            batch_size (int): Размер батча
            target_size (tuple): Размер изображения (ширина, высота)
            one_hot (bool): Использовать one-hot encoding для меток
            
        Yields:
            tuple: (X_batch, y_batch)
        """
        while True:
            # Случайный выбор видео
            indices = np.random.permutation(len(self.video_paths))
            
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i + batch_size]
                batch_frames = []
                batch_labels = []
                
                # Загрузка текущего батча
                for idx in batch_indices:
                    frames = self.load_video(self.video_paths[idx], target_size)
                    if len(frames) >= sequence_length:  # Проверяем, что видео достаточно длинное
                        batch_frames.extend(frames)
                        batch_labels.extend([self.labels[idx]] * len(frames))
                
                if len(batch_frames) > 0:  # Проверяем, что есть данные для обработки
                    sequences, sequence_labels = self.create_sequences(
                        batch_frames, batch_labels, sequence_length, target_size, one_hot
                    )
                    
                    if len(sequences) > 0:  # Проверяем, что последовательности созданы
                        yield sequences, sequence_labels
    
    def load_data(self, sequence_length, batch_size, target_size=None, one_hot=False):
        """
        Загрузка данных для обучения.
        
        Args:
            sequence_length (int): Длина последовательности
            batch_size (int): Размер батча
            target_size (tuple): Размер изображения (ширина, высота)
            one_hot (bool): Использовать one-hot encoding для меток
            
        Returns:
            generator: Генератор данных
        """
        return self.data_generator(sequence_length, batch_size, target_size, one_hot) 