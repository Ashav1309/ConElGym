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
    def __init__(self, data_path, max_videos=Config.MAX_VIDEOS):
        """
        Инициализация загрузчика данных
        Args:
            data_path: путь к директории с данными
            max_videos: максимальное количество видео для загрузки (None для загрузки всех видео)
        """
        self.data_path = data_path
        self.max_videos = max_videos
        self.video_paths = []
        self.labels = []
        self.video_count = 0
        self.batch_size = 32
        self.current_video_index = 0
        self.current_frame_index = 0
        self.current_batch = 0
        self.total_batches = 0
        self.network_handler = NetworkErrorHandler()
        self.network_monitor = NetworkMonitor()
        self.cache = {}
        self.cache_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Инициализация параметров из конфигурации
        self.sequence_length = Config.SEQUENCE_LENGTH
        self.max_sequences_per_video = Config.MAX_SEQUENCES_PER_VIDEO
        
        # Загружаем видео
        self._load_videos()
        
        # Рассчитываем общее количество батчей
        self._calculate_total_batches()
        
        print(f"[DEBUG] Загружено {self.video_count} видео")
        if self.max_videos is not None and self.video_count > self.max_videos:
            print(f"[WARNING] Загружено слишком много видео: {self.video_count} > {self.max_videos}")
            self.video_paths = self.video_paths[:self.max_videos]
            self.labels = self.labels[:self.max_videos]
            self.video_count = self.max_videos
            print(f"[DEBUG] Оставлено {self.video_count} видео")
    
    def _load_videos(self):
        """
        Загрузка путей к видео и соответствующих аннотаций.
        
        Raises:
            FileNotFoundError: Если директория с данными не найдена
            ValueError: Если нет видео файлов в директории
        """
        try:
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"Директория с данными не найдена: {self.data_path}")
            
            # Определяем путь к аннотациям в зависимости от типа данных (train/valid)
            if 'train' in self.data_path:
                annotation_dir = Config.TRAIN_ANNOTATION_PATH
            else:
                annotation_dir = Config.VALID_ANNOTATION_PATH
            
            if not os.path.exists(annotation_dir):
                print(f"[DEBUG] Создание директории для аннотаций: {annotation_dir}")
                os.makedirs(annotation_dir, exist_ok=True)
            
            print(f"[DEBUG] Поиск видео в {self.data_path}, аннотаций в {annotation_dir}")
            
            self.video_paths = []
            self.labels = []
            self.video_count = 0  # Добавляем счетчик
            
            for file_name in os.listdir(self.data_path):
                if self.max_videos is not None and self.video_count >= self.max_videos:
                    break
                
                file_path = os.path.join(self.data_path, file_name)
                if file_name.endswith('.mp4') and os.path.isfile(file_path):
                    self.video_paths.append(file_path)
                    base = os.path.splitext(file_name)[0]
                    ann_path = os.path.join(annotation_dir, base + '.json')
                    
                    if os.path.exists(ann_path):
                        pass
                        # print(f"[DEBUG] Найдена аннотация для {file_name}")
                    else:
                        print(f"[DEBUG] Аннотация для {file_name} не найдена")
                    
                    self.labels.append(ann_path if os.path.exists(ann_path) else None)
                    self.video_count += 1  # Увеличиваем счетчик
            
            self.video_count = len(self.video_paths)
            
            print(f"[DEBUG] Загружено {self.video_count} видео файлов")
            
            # Ограничиваем количество видео до Config.MAX_VIDEOS
            if hasattr(Config, "MAX_VIDEOS") and len(self.video_paths) > Config.MAX_VIDEOS:
                print(f"[DEBUG] Ограничиваем количество видео до {Config.MAX_VIDEOS}")
                self.video_paths = self.video_paths[:Config.MAX_VIDEOS]
                self.labels = self.labels[:Config.MAX_VIDEOS]
            
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
            
            return cap, total_frames
            
        except Exception as e:
            print(f"[ERROR] Ошибка при загрузке видео: {str(e)}")
            raise
    
    def get_batch(self, batch_size, sequence_length, target_size, one_hot=True, max_sequences_per_video=None):
        """Получение батча данных"""
        try:
            if self.current_video_index >= len(self.video_paths):
                self.current_video_index = 0
                return None
            
            video_path = self.video_paths[self.current_video_index]
            cap, total_frames = self.load_video(video_path)
            
            # Загружаем аннотации
            annotations = self.labels[self.current_video_index]
            if annotations is not None:
                with open(annotations, 'r') as f:
                    ann_data = json.load(f)
                    frame_labels = np.zeros((total_frames, Config.NUM_CLASSES), dtype=np.float32)
                    
                    for annotation in ann_data['annotations']:
                        start_frame = annotation['start_frame']
                        end_frame = annotation['end_frame']
                        
                        for frame_idx in range(start_frame, end_frame + 1):
                            if frame_idx < len(frame_labels):
                                if frame_idx == start_frame:
                                    frame_labels[frame_idx] = [1, 0]
                                elif frame_idx == end_frame:
                                    frame_labels[frame_idx] = [0, 1]
                                else:
                                    frame_labels[frame_idx] = [0, 0]
            else:
                frame_labels = np.zeros((total_frames, Config.NUM_CLASSES), dtype=np.float32)
            
            # Создаем батч
            batch_sequences = []
            batch_labels = []
            
            while len(batch_sequences) < batch_size:
                # Читаем sequence_length кадров
                frames = []
                labels = []
                
                for _ in range(sequence_length):
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    frame = cv2.resize(frame, target_size)
                    frames.append(frame)
                    labels.append(frame_labels[self.current_frame_index])
                    self.current_frame_index += 1
                
                if len(frames) == sequence_length:
                    batch_sequences.append(frames)
                    batch_labels.append(labels)
                
                # Если достигли конца видео, переходим к следующему
                if self.current_frame_index >= total_frames:
                    cap.release()
                    self.current_video_index += 1
                    self.current_frame_index = 0
                    if self.current_video_index >= len(self.video_paths):
                        break
                    video_path = self.video_paths[self.current_video_index]
                    cap, total_frames = self.load_video(video_path)
            
            if len(batch_sequences) == 0:
                return None
            
            # Преобразуем в numpy массивы
            X = np.array(batch_sequences, dtype=np.float32) / 255.0
            y = np.array(batch_labels, dtype=np.float32)
            
            return X, y
            
        except Exception as e:
            print(f"[ERROR] Ошибка при получении батча: {str(e)}")
            print("[DEBUG] Stack trace:", flush=True)
            import traceback
            traceback.print_exc()
            return None
    
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
                        # Создаем массив меток для каждого кадра
                        frame_labels = np.zeros((len(frames), Config.NUM_CLASSES), dtype=np.float32)
                        
                        # Обрабатываем каждую аннотацию
                        for annotation in ann_data['annotations']:
                            start_frame = annotation['start_frame']
                            end_frame = annotation['end_frame']
                            
                            print(f"[DEBUG] Аннотация: начало кадра {start_frame}, конец кадра {end_frame}")
                            
                            # Устанавливаем метки для кадров в пределах аннотации
                            for frame_idx in range(start_frame, end_frame + 1):
                                if frame_idx < len(frame_labels):
                                    # [1, 0] для начала элемента
                                    if frame_idx == start_frame:
                                        frame_labels[frame_idx] = [1, 0]
                                    # [0, 1] для конца элемента
                                    elif frame_idx == end_frame:
                                        frame_labels[frame_idx] = [0, 1]
                                    # [0, 0] для промежуточных кадров
                                    else:
                                        frame_labels[frame_idx] = [0, 0]
                        
                        annotations = frame_labels
                        print(f"[DEBUG] Загружены аннотации формы: {annotations.shape}")
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
            
            # Создаем последовательности
            for i in range(0, len(frames) - self.sequence_length + 1, self.sequence_length // 2):
                sequence = frames[i:i + self.sequence_length]
                sequence_labels = annotations[i:i + self.sequence_length]
                
                # Проверяем размерности последовательности
                if len(sequence) == self.sequence_length and len(sequence_labels) == self.sequence_length:
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
            print("\n[DEBUG] ===== Запуск генератора данных =====")
            print(f"[DEBUG] Количество видео для обработки: {len(self.video_paths)}")
            
            while True:
                # Получаем батч данных
                batch_data = self.get_batch(
                    batch_size=self.batch_size,
                    sequence_length=self.sequence_length,
                    target_size=Config.INPUT_SIZE,
                    one_hot=True,
                    max_sequences_per_video=self.max_sequences_per_video
                )
                
                if batch_data is None:
                    print("[DEBUG] Достигнут конец эпохи")
                    break
                    
                X, y = batch_data
                
                # Проверяем размерности
                if X.shape[0] == 0 or y.shape[0] == 0:
                    print("[WARNING] Получен пустой батч")
                    continue
                    
                # Преобразуем в тензоры
                x = tf.convert_to_tensor(X, dtype=tf.float32)
                y = tf.convert_to_tensor(y, dtype=tf.float32)
                
                yield (x, y)
                
        except Exception as e:
            print(f"[ERROR] Ошибка в генераторе данных: {str(e)}")
            print("[DEBUG] Stack trace:", flush=True)
            import traceback
            traceback.print_exc()
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
    
    def _calculate_total_batches(self):
        """
        Рассчитывает общее количество батчей для данных.
        """
        self.total_batches = sum(1 for _ in self.data_generator()) 
    
    def get_video_info(self, video_path):
        """
        Получение информации о видео
        
        Args:
            video_path: путь к видео файлу
            
        Returns:
            dict: словарь с информацией о видео (total_frames, fps, width, height)
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Не удалось открыть видео: {video_path}")
            
            # Получаем информацию о видео
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            cap.release()
            
            return {
                'total_frames': total_frames,
                'fps': fps,
                'width': width,
                'height': height
            }
            
        except Exception as e:
            print(f"[ERROR] Ошибка при получении информации о видео {video_path}: {str(e)}")
            raise 