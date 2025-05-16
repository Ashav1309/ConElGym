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
        self.stuck_counter = 0
        self.max_stuck_batches = 10
        self.positive_indices_cache = {}  # Кэш для индексов положительных кадров
        self.video_cache = {}  # Кэш для видео
        self.used_frames_cache = {}  # Кэш для отслеживания использованных кадров
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
        """Загрузка видео с оптимизацией памяти и подробным логированием"""
        try:
            print(f"[DEBUG] Загрузка видео: {os.path.basename(video_path)}")
            
            # Проверяем существование файла
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Видеофайл не найден: {video_path}")
            
            # Проверяем размер файла
            file_size = os.path.getsize(video_path)
            print(f"[DEBUG] Размер файла: {file_size / (1024*1024):.2f} MB")
            
            # Открываем видео с таймаутом
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Не удалось открыть видео: {video_path}")
            
            print("[DEBUG] Видео успешно открыто")
            
            # Получаем информацию о видео с проверкой каждого свойства
            try:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                print(f"[DEBUG] Ширина: {width}")
            except Exception as e:
                print(f"[ERROR] Ошибка при получении ширины: {str(e)}")
                width = 0
            
            try:
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"[DEBUG] Высота: {height}")
            except Exception as e:
                print(f"[ERROR] Ошибка при получении высоты: {str(e)}")
                height = 0
            
            try:
                fps = cap.get(cv2.CAP_PROP_FPS)
                print(f"[DEBUG] FPS: {fps}")
            except Exception as e:
                print(f"[ERROR] Ошибка при получении FPS: {str(e)}")
                fps = 0
            
            try:
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                print(f"[DEBUG] Количество кадров: {total_frames}")
            except Exception as e:
                print(f"[ERROR] Ошибка при получении количества кадров: {str(e)}")
                total_frames = 0
            
            # Проверяем корректность полученных данных
            if width <= 0 or height <= 0 or fps <= 0 or total_frames <= 0:
                raise ValueError(f"Некорректные параметры видео: width={width}, height={height}, fps={fps}, frames={total_frames}")
            
            print(f"[DEBUG] Видео успешно загружено:")
            print(f"  - Размер: {width}x{height}")
            print(f"  - FPS: {fps}")
            print(f"  - Количество кадров: {total_frames}")
            
            return cap, total_frames
            
        except Exception as e:
            print(f"[ERROR] Ошибка при загрузке видео: {str(e)}")
            if 'cap' in locals():
                cap.release()
            raise
    
    def _collect_sequences(self, video_path, start_frame, batch_size, sequence_length, target_size, frame_labels, positive_indices=None, force_positive=False):
        """
        Сбор последовательностей из видео с улучшенной логикой для положительных примеров
        
        Args:
            video_path: путь к видео
            start_frame: начальный кадр
            batch_size: размер батча
            sequence_length: длина последовательности
            target_size: размер кадра
            frame_labels: метки кадров
            positive_indices: индексы положительных кадров
            force_positive: флаг принудительного добавления положительных примеров
        """
        try:
            batch_sequences = []
            batch_labels = []
            
            # Получаем множество использованных кадров для текущего видео
            used_indices = self.used_frames_cache[video_path]
            
            # Получаем видео из кэша
            cap, total_frames = self.video_cache[video_path]
            
            # Сначала добавляем положительные последовательности
            if force_positive and positive_indices is not None and len(positive_indices) > 0:
                num_positive = max(1, batch_size // 4)
                print(f"[DEBUG] _collect_sequences: Добавляем {num_positive} положительных последовательностей")
                
                # Фильтруем положительные индексы, исключая уже использованные
                available_pos_indices = [idx for idx in positive_indices if idx not in used_indices]
                
                if len(available_pos_indices) > 0:
                    selected_pos_indices = np.random.choice(available_pos_indices, 
                                                          size=min(num_positive, len(available_pos_indices)), 
                                                          replace=False)
                    
                    for pos_idx in selected_pos_indices:
                        # Центрируем последовательность вокруг положительного кадра
                        start_idx = max(0, pos_idx - sequence_length // 2)
                        end_idx = min(total_frames, start_idx + sequence_length)
                        
                        # Проверяем, что последовательность не выходит за границы
                        if end_idx - start_idx < sequence_length:
                            continue
                        
                        # Проверяем, что последовательность не пересекается с уже использованными
                        if any(idx in used_indices for idx in range(start_idx, end_idx)):
                            continue
                        
                        sequence = []
                        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
                        
                        # Собираем последовательность
                        for _ in range(sequence_length):
                            ret, frame = cap.read()
                            if not ret:
                                break
                            if target_size:
                                frame = cv2.resize(frame, target_size)
                            sequence.append(frame)
                        
                        if len(sequence) == sequence_length:
                            batch_sequences.append(np.array(sequence))
                            batch_labels.append(frame_labels[pos_idx])
                            # Отмечаем использованные кадры
                            used_indices.update(range(start_idx, end_idx))
                            print(f"[DEBUG] _collect_sequences: Добавлена положительная последовательность с кадра {start_idx} по {end_idx} (pos_idx={pos_idx})")
            
            # Добавляем обычные последовательности
            current_frame = start_frame
            while len(batch_sequences) < batch_size and current_frame < total_frames:
                # Пропускаем уже использованные кадры
                if current_frame in used_indices:
                    current_frame += 1
                    continue
                
                # Проверяем, что последовательность не выходит за границы
                if current_frame + sequence_length > total_frames:
                    break
                
                sequence = []
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                
                # Собираем последовательность
                for _ in range(sequence_length):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if target_size:
                        frame = cv2.resize(frame, target_size)
                    sequence.append(frame)
                
                if len(sequence) == sequence_length:
                    batch_sequences.append(np.array(sequence))
                    batch_labels.append(frame_labels[current_frame])
                    # Отмечаем использованные кадры
                    used_indices.update(range(current_frame, current_frame + sequence_length))
                
                current_frame += 1
            
            return batch_sequences, batch_labels, current_frame
            
        except Exception as e:
            print(f"[ERROR] Ошибка в _collect_sequences: {str(e)}")
            print("[DEBUG] Stack trace:", flush=True)
            import traceback
            traceback.print_exc()
            return [], [], start_frame

    def get_batch(self, batch_size, sequence_length, target_size, one_hot=True, max_sequences_per_video=None, force_positive=False):
        """Получение батча данных с улучшенной логикой переключения между видео"""
        try:
            # Проверяем, нужно ли переключиться на следующее видео
            if self.current_video_index >= len(self.video_paths):
                print(f"[DEBUG] get_batch: Обработаны все видео, начинаем новую эпоху")
                self.current_video_index = 0
                self.current_frame_index = 0
                self.stuck_counter = 0
                self.used_frames_cache.clear()  # Очищаем кэш использованных кадров
                self.positive_indices_cache.clear()  # Очищаем кэш положительных индексов
                return None
            
            video_path = self.video_paths[self.current_video_index]
            print(f"[DEBUG] get_batch: Обработка видео {os.path.basename(video_path)} (индекс {self.current_video_index + 1}/{len(self.video_paths)})")
            
            # Инициализируем множество использованных кадров для текущего видео, если его еще нет
            if video_path not in self.used_frames_cache:
                self.used_frames_cache[video_path] = set()
            
            # Загружаем или получаем из кэша видео
            if video_path not in self.video_cache:
                cap, total_frames = self.load_video(video_path)
                self.video_cache[video_path] = (cap, total_frames)
            else:
                cap, total_frames = self.video_cache[video_path]
            
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
                                frame_labels[frame_idx] = [1, 0]
            else:
                frame_labels = np.zeros((total_frames, Config.NUM_CLASSES), dtype=np.float32)
            
            # Получаем индексы положительных кадров
            if video_path not in self.positive_indices_cache:
                positive_indices = np.where(np.any(frame_labels == 1, axis=1))[0]
                self.positive_indices_cache[video_path] = positive_indices
            else:
                positive_indices = self.positive_indices_cache[video_path]
            
            old_frame_index = self.current_frame_index
            # Собираем последовательности
            batch_sequences, batch_labels, new_frame_index = self._collect_sequences(
                video_path,
                self.current_frame_index,
                batch_size,
                sequence_length,
                target_size,
                frame_labels,
                positive_indices,
                force_positive
            )
            # Обновляем индексы
            self.current_frame_index = new_frame_index

            # Если не удалось продвинуться по кадрам (застряли в конце видео)
            if self.current_frame_index == old_frame_index:
                self.stuck_counter += 1
                print(f"[DEBUG] get_batch: Не удалось продвинуться по кадрам {self.stuck_counter} раз подряд")
                if self.stuck_counter >= self.max_stuck_batches:
                    print(f"[DEBUG] get_batch: Достигнут лимит маленьких батчей, переходим к следующему видео")
                    self.current_video_index += 1
                    self.current_frame_index = 0
                    self.stuck_counter = 0
                    # Очищаем кэш для текущего видео
                    if video_path in self.used_frames_cache:
                        del self.used_frames_cache[video_path]
                    if video_path in self.positive_indices_cache:
                        del self.positive_indices_cache[video_path]
                    return None
                if len(batch_sequences) > 0:
                    # Возвращаем маленький батч
                    num_positive = np.sum([label[0] == 1 for label in batch_labels])
                    print(f"[DEBUG] get_batch: Возвращаем маленький батч. Положительных примеров (label[0] == 1): {num_positive}")
                    print(f"[DEBUG] get_batch: Текущий индекс видео: {self.current_video_index + 1}/{len(self.video_paths)}")
                    print(f"[DEBUG] get_batch: Текущий индекс кадра: {self.current_frame_index}/{total_frames}")
                    return np.array(batch_sequences), np.array(batch_labels)
                else:
                    # Переходим к следующему видео
                    print(f"[DEBUG] get_batch: Батч пустой, переходим к следующему видео")
                    self.current_video_index += 1
                    self.current_frame_index = 0
                    self.stuck_counter = 0
                    # Очищаем кэш для текущего видео
                    if video_path in self.used_frames_cache:
                        del self.used_frames_cache[video_path]
                    if video_path in self.positive_indices_cache:
                        del self.positive_indices_cache[video_path]
                    return None
            else:
                self.stuck_counter = 0
            
            # Проверяем наличие положительных примеров для train
            if force_positive:
                positive_count = np.sum([np.any(label == 1) for label in batch_labels])
                if positive_count == 0:
                    print(f"[WARNING] В батче нет положительных примеров, хотя force_positive=True")
                    return self._resample_batch(video_path, batch_size, sequence_length, target_size, frame_labels, positive_indices)
            
            # Логируем информацию
            num_positive = np.sum([label[0] == 1 for label in batch_labels])
            print(f"[DEBUG] get_batch: Батч собран. Размер: {len(batch_sequences)}")
            print(f"[DEBUG] get_batch: Положительных примеров (label[0] == 1): {num_positive}")
            print(f"[DEBUG] get_batch: Текущий индекс видео: {self.current_video_index + 1}/{len(self.video_paths)}")
            print(f"[DEBUG] get_batch: Текущий индекс кадра: {self.current_frame_index}/{total_frames}")
            
            return np.array(batch_sequences), np.array(batch_labels)
            
        except Exception as e:
            print(f"[ERROR] Ошибка в get_batch: {str(e)}")
            print("[DEBUG] Stack trace:", flush=True)
            import traceback
            traceback.print_exc()
            return None
    
    def _resample_batch(self, video_path, batch_size, sequence_length, target_size, frame_labels, positive_indices):
        """Пересборка батча с гарантированным наличием положительных примеров"""
        max_attempts = 3
        print(f"[DEBUG] Начинаем пересборку батча для видео {os.path.basename(video_path)}")
        print(f"[DEBUG] Доступно положительных кадров: {len(positive_indices)}")
        
        for attempt in range(max_attempts):
            print(f"\n[DEBUG] Попытка {attempt + 1}/{max_attempts} пересборки батча")
            
            # Пробуем взять кадры из другой части видео
            new_start = np.random.randint(0, len(frame_labels) - sequence_length)
            print(f"[DEBUG] Начинаем с кадра {new_start}")
            
            # Собираем новый батч
            batch_sequences = []
            batch_labels = []
            
            # Сначала добавляем положительные последовательности
            if len(positive_indices) > 0:
                num_positive = max(1, batch_size // 4)
                print(f"[DEBUG] Пытаемся добавить {num_positive} положительных последовательностей")
                
                selected_pos_indices = np.random.choice(positive_indices, 
                                                      size=min(num_positive, len(positive_indices)), 
                                                      replace=False)
                
                for pos_idx in selected_pos_indices:
                    start_idx = max(0, pos_idx - sequence_length // 2)
                    end_idx = min(len(frame_labels), start_idx + sequence_length)
                    
                    if end_idx - start_idx < sequence_length:
                        print(f"[DEBUG] Пропускаем последовательность: неполная длина ({end_idx - start_idx} < {sequence_length})")
                        continue
                    
                    cap, _ = self.video_cache[video_path]
                    sequence = []
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
                    
                    for _ in range(sequence_length):
                        ret, frame = cap.read()
                        if not ret:
                            print(f"[DEBUG] Пропускаем последовательность: ошибка чтения кадра")
                            break
                        if target_size:
                            frame = cv2.resize(frame, target_size)
                        sequence.append(frame)
                    
                    if len(sequence) == sequence_length:
                        batch_sequences.append(np.array(sequence))
                        batch_labels.append(frame_labels[pos_idx])
                        print(f"[DEBUG] Добавлена положительная последовательность с кадра {start_idx} по {end_idx}")
            
            # Добавляем обычные последовательности
            print(f"[DEBUG] Добавляем обычные последовательности. Текущий размер батча: {len(batch_sequences)}")
            while len(batch_sequences) < batch_size:
                sequence = []
                cap, _ = self.video_cache[video_path]
                cap.set(cv2.CAP_PROP_POS_FRAMES, new_start)
                
                for _ in range(sequence_length):
                    ret, frame = cap.read()
                    if not ret:
                        print(f"[DEBUG] Пропускаем последовательность: ошибка чтения кадра")
                        break
                    if target_size:
                        frame = cv2.resize(frame, target_size)
                    sequence.append(frame)
                
                if len(sequence) == sequence_length:
                    batch_sequences.append(np.array(sequence))
                    batch_labels.append(frame_labels[new_start])
                    print(f"[DEBUG] Добавлена обычная последовательность с кадра {new_start}")
                
                new_start = (new_start + 1) % (len(frame_labels) - sequence_length)
            
            # Проверяем наличие положительных примеров
            positive_count = np.sum([np.any(label == 1) for label in batch_labels])
            print(f"[DEBUG] В пересобранном батче положительных примеров: {positive_count}")
            
            if positive_count > 0:
                print(f"[DEBUG] Успешно пересобран батч с {positive_count} положительными примерами")
                return np.array(batch_sequences), np.array(batch_labels)
            else:
                print(f"[DEBUG] Не удалось добавить положительные примеры в попытке {attempt + 1}")
        
        print(f"[ERROR] Не удалось собрать батч с положительными примерами после {max_attempts} попыток")
        print(f"[DEBUG] Возвращаем последний собранный батч (размер: {len(batch_sequences)})")
        return np.array(batch_sequences), np.array(batch_labels)
    
    def create_sequences(self, frames, annotations):
        """Создание последовательностей с оптимизацией памяти"""
        sequences = []
        labels = []
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
                    for annotation in ann_data['annotations']:
                        start_frame = annotation['start_frame']
                        end_frame = annotation['end_frame']
                        for frame_idx in range(start_frame, end_frame + 1):
                            if frame_idx < len(frame_labels):
                                frame_labels[frame_idx] = [1, 0]
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
    
    def preload_video(self, video_path, target_size):
        """
        Предварительная загрузка видео в отдельном потоке.
        """
        self.load_video(video_path)
    
    def data_generator(self, force_positive=True):
        """Генератор данных с sampling положительных примеров"""
        try:
            print("\n[DEBUG] ===== Запуск генератора данных =====")
            print(f"[DEBUG] Количество видео для обработки: {len(self.video_paths)}")
            while True:
                batch_data = self.get_batch(
                    batch_size=self.batch_size,
                    sequence_length=self.sequence_length,
                    target_size=Config.INPUT_SIZE,
                    one_hot=True,
                    max_sequences_per_video=self.max_sequences_per_video,
                    force_positive=force_positive
                )
                if batch_data is None:
                    print("[DEBUG] Достигнут конец эпохи")
                    break
                
                X, y = batch_data
                if X is None or y is None or X.shape[0] == 0 or y.shape[0] == 0:
                    print("[WARNING] Получен пустой батч")
                    continue
                
                try:
                    num_positive = int((y[...,1] == 1).sum())
                    # print(f"[DEBUG] В батче положительных примеров (class 1): {num_positive}")
                    
                    # Конвертируем в тензоры с оптимизацией памяти
                    x = tf.convert_to_tensor(X, dtype=tf.float32)
                    y_tensor = tf.convert_to_tensor(y, dtype=tf.float32)
                    
                    # Очищаем память
                    del X
                    del y
                    gc.collect()
                    
                    yield (x, y_tensor)
                    
                except Exception as e:
                    print(f"[ERROR] Ошибка при обработке батча: {str(e)}")
                    print("[DEBUG] Stack trace:", flush=True)
                    import traceback
                    traceback.print_exc()
                    continue
                
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
        try:
            print("[DEBUG] Начало расчета общего количества батчей")
            batch_count = 0
            for _ in self.data_generator():
                batch_count += 1
            self.total_batches = batch_count
            print(f"[DEBUG] Рассчитано батчей: {self.total_batches}")
        except Exception as e:
            print(f"[ERROR] Ошибка при расчете количества батчей: {str(e)}")
            print("[DEBUG] Stack trace:", flush=True)
            import traceback
            traceback.print_exc()
            self.total_batches = 0
    
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
