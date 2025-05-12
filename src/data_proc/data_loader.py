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
        Загрузка путей к видео (без классов, без подкаталогов) и соответствующих аннотаций.
        """
        annotation_dir = os.path.join(self.data_path, 'annotations')
        print(f"[DEBUG] Поиск видео в {self.data_path}, аннотаций в {annotation_dir}")
        while True:
            for file_name in os.listdir(self.data_path):
                file_path = os.path.join(self.data_path, file_name)
                if file_name.endswith('.mp4') and os.path.isfile(file_path):
                    self.video_paths.append(file_path)
                    base = os.path.splitext(file_name)[0]
                    ann_path = os.path.join(annotation_dir, base + '.json')
                    if os.path.exists(ann_path):
                        print(f"[DEBUG] Найдена аннотация для {file_name}: {ann_path}")
                    else:
                        print(f"[DEBUG] Аннотация для {file_name} не найдена!")
                    self.labels.append(ann_path if os.path.exists(ann_path) else None)
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
                print(f"[DEBUG] Видео из кэша: {video_path}")
                return self.cache[cache_key]
        
        frames = []
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Изменение размера изображения, если указано
            if target_size:
                # cv2.resize ожидает размеры в формате (ширина, высота)
                frame = cv2.resize(frame, (target_size[1], target_size[0]))
                # print(f"Resized frame shape: {frame.shape}")  # Отладочная информация
            
            # Нормализация
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)
            frame_idx += 1
        
        cap.release()
        print(f"[DEBUG] Загружено {len(frames)} кадров из {video_path}")
        
        with self.cache_lock:
            self.cache[cache_key] = frames
        
        return frames
    
    def create_sequences(self, frames, annotation_path, sequence_length, one_hot=False):
        """
        Создание последовательностей кадров и меток на основе аннотации.
        """
        print(f"[DEBUG] Создание последовательностей: frames={len(frames)}, annotation={annotation_path}")
        labels = [0] * len(frames)
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
                print(f"[DEBUG] Ошибка чтения аннотации {annotation_path}: {e}")
        else:
            print(f"[DEBUG] Нет аннотации для видео, все метки = 0")
        sequences = []
        sequence_labels = []
        for i in range(len(frames) - sequence_length + 1):
            sequence = frames[i:i + sequence_length]
            seq_labels = labels[i:i + sequence_length]
            sequences.append(sequence)
            if one_hot:
                sequence_labels.append([[1,0] if l==0 else [0,1] for l in seq_labels])
            else:
                sequence_labels.append(seq_labels)
        print(f"[DEBUG] Сформировано {len(sequences)} последовательностей")
        return np.array(sequences), np.array(sequence_labels)
    
    def preload_video(self, video_path, target_size):
        """
        Предварительная загрузка видео в отдельном потоке.
        """
        self.load_video(video_path, target_size)
    
    def data_generator(self, sequence_length, batch_size, target_size=None, one_hot=False, infinite_loop=False):
        print(f"[DEBUG] Запуск генератора данных: sequence_length={sequence_length}, batch_size={batch_size}")
        while True:
            indices = np.random.permutation(len(self.video_paths))
            # print(f"[DEBUG] Индексы для батча: {indices}")
            batch_frames = []
            batch_labels = []
            for idx in indices:
                frames = self.load_video(self.video_paths[idx], target_size)
                annotation_path = self.labels[idx]
                seqs, seq_labels = self.create_sequences(frames, annotation_path, sequence_length, one_hot)
                print(f"[DEBUG] sequences: {seqs.shape if hasattr(seqs, 'shape') else type(seqs)}")
                for s, l in zip(seqs, seq_labels):
                    batch_frames.append(s)
                    batch_labels.append(l)
                    if len(batch_frames) == batch_size:
                        print(f"[DEBUG] batch_frames: {len(batch_frames)}")
                        print(f"[DEBUG] batch_labels shape: {np.array(batch_labels).shape}")
                        yield np.array(batch_frames), np.array(batch_labels)
                        batch_frames = []
                        batch_labels = []
            # Если остались невыданные последовательности
            if len(batch_frames) > 0:
                print(f"[DEBUG] batch_frames (остаток): {len(batch_frames)}")
                print(f"[DEBUG] batch_labels shape (остаток): {np.array(batch_labels).shape}")
                yield np.array(batch_frames), np.array(batch_labels)
            if not infinite_loop:
                break
    
    def load_data(self, sequence_length, batch_size, target_size=None, one_hot=False, infinite_loop=False):
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
        return self.data_generator(sequence_length, batch_size, target_size, one_hot, infinite_loop) 