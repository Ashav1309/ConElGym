import cv2
import numpy as np
from typing import Tuple, List, Generator, Optional, Dict, Set
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
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

class VideoDataLoaderError(Exception):
    """Базовый класс для исключений VideoDataLoader"""
    pass

class InvalidAnnotationError(VideoDataLoaderError):
    """Исключение при некорректном формате аннотаций"""
    pass

class CorruptedVideoError(VideoDataLoaderError):
    """Исключение при поврежденном видео"""
    pass

@dataclass
class VideoInfo:
    """Информация о видео"""
    path: str
    total_frames: int
    fps: float
    width: int
    height: int
    file_size: int
    exists: bool

class VideoDataLoader:
    def __init__(self, data_path: str, max_videos: Optional[int] = None):
        """
        Инициализация загрузчика данных
        
        Args:
            data_path: путь к директории с данными
            max_videos: максимальное количество видео для загрузки (None для загрузки всех видео)
            
        Raises:
            FileNotFoundError: если директория с данными не найдена
            ValueError: если нет видео файлов в директории
        """
        self.stuck_counter = 0
        self.max_stuck_batches = Config.MAX_STUCK_BATCHES
        self.cache_cleanup_threshold = Config.CACHE_CLEANUP_THRESHOLD
        
        # Кэши
        self.positive_indices_cache: Dict[str, np.ndarray] = {}
        self.video_cache: Dict[str, cv2.VideoCapture] = {}
        self.used_frames_cache: Dict[str, Set[int]] = {}
        self.file_info_cache: Dict[str, VideoInfo] = {}
        
        self.processed_videos: Set[str] = set()
        self.data_path = Path(data_path)
        self.max_videos = max_videos or Config.MAX_VIDEOS
        self.video_paths: List[str] = []
        self.labels: List[Optional[str]] = []
        self.video_count = 0
        
        # Параметры из конфигурации
        self.batch_size = Config.BATCH_SIZE
        self.sequence_length = Config.SEQUENCE_LENGTH
        self.max_sequences_per_video = Config.MAX_SEQUENCES_PER_VIDEO
        self.target_size = Config.INPUT_SIZE
        
        # Состояние
        self.current_video_index = 0
        self.current_frame_index = 0
        self.current_batch = 0
        self.total_batches = 0
        
        # Загружаем видео
        self._load_videos()
        
        # Рассчитываем общее количество батчей
        self._calculate_total_batches()
        
        logger.info(f"Загружено {self.video_count} видео")
        if self.max_videos is not None and self.video_count > self.max_videos:
            logger.warning(f"Загружено слишком много видео: {self.video_count} > {self.max_videos}")
            self.video_paths = self.video_paths[:self.max_videos]
            self.labels = self.labels[:self.max_videos]
            self.video_count = self.max_videos
            logger.info(f"Оставлено {self.video_count} видео")

    def clear_cache(self):
        """Очистка всех кэшей и освобождение памяти"""
        logger.debug("Очистка кэшей")
        for cap in self.video_cache.values():
            if cap is not None:
                cap.release()
        self.video_cache.clear()
        self.used_frames_cache.clear()
        self.positive_indices_cache.clear()
        self.file_info_cache.clear()
        gc.collect()

    def _get_video_info(self, video_path: str) -> VideoInfo:
        """
        Получение информации о видео с кэшированием
        
        Args:
            video_path: путь к видео файлу
            
        Returns:
            VideoInfo: информация о видео
            
        Raises:
            CorruptedVideoError: если видео повреждено или имеет некорректные параметры
        """
        if video_path in self.file_info_cache:
            return self.file_info_cache[video_path]
            
        path = Path(video_path)
        exists = path.exists()
        file_size = path.stat().st_size if exists else 0
        
        if not exists:
            return VideoInfo(
                path=video_path,
                total_frames=0,
                fps=0,
                width=0,
                height=0,
                file_size=0,
                exists=False
            )
            
        cap = None
        try:
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise CorruptedVideoError(f"Не удалось открыть видео: {video_path}")
                
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if fps <= 0 or total_frames <= 0 or width <= 0 or height <= 0:
                raise CorruptedVideoError(f"Некорректные параметры видео: fps={fps}, frames={total_frames}, size={width}x{height}")
                
            info = VideoInfo(
                path=video_path,
                total_frames=total_frames,
                fps=fps,
                width=width,
                height=height,
                file_size=file_size,
                exists=True
            )
            self.file_info_cache[video_path] = info
            return info
            
        except Exception as e:
            raise CorruptedVideoError(f"Ошибка при получении информации о видео {video_path}: {str(e)}")
        finally:
            if cap is not None:
                cap.release()

    def load_video(self, video_path: str) -> Tuple[cv2.VideoCapture, int]:
        """
        Загрузка видео с оптимизацией памяти и подробным логированием
        
        Args:
            video_path: путь к видео файлу
            
        Returns:
            Tuple[cv2.VideoCapture, int]: кортеж (объект VideoCapture, количество кадров)
            
        Raises:
            FileNotFoundError: если файл не найден
            CorruptedVideoError: если видео повреждено или имеет некорректные параметры
        """
        cap = None
        try:
            logger.debug(f"Загрузка видео: {os.path.basename(video_path)}")
            
            # Получаем информацию о видео
            info = self._get_video_info(video_path)
            if not info.exists:
                raise FileNotFoundError(f"Видеофайл не найден: {video_path}")
                
            logger.debug(f"Размер файла: {info.file_size / (1024*1024):.2f} MB")
            
            # Открываем видео
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise CorruptedVideoError(f"Не удалось открыть видео: {video_path}")
                
            logger.debug(f"Видео успешно загружено:")
            logger.debug(f"  - Размер: {info.width}x{info.height}")
            logger.debug(f"  - FPS: {info.fps}")
            logger.debug(f"  - Количество кадров: {info.total_frames}")
            
            return cap, info.total_frames
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке видео: {str(e)}")
            if cap is not None:
                cap.release()
            raise

    def _load_videos(self):
        """Загрузка видео из директории"""
        try:
            video_files = []
            for ext in ['.mp4', '.avi', '.mov']:
                video_files.extend(list(self.data_path.glob(f'*{ext}')))
            
            if not video_files:
                raise ValueError(f"Видео не найдены в {self.data_path}")
            
            # Сортируем видео по имени для воспроизводимости
            video_files.sort()
            
            # Ограничиваем количество видео
            if len(video_files) > Config.MAX_VIDEOS:
                print(f"[DEBUG] Ограничение количества видео до {Config.MAX_VIDEOS}")
                video_files = video_files[:Config.MAX_VIDEOS]
            
            # Определяем тип данных (train/valid) на основе пути
            data_type = 'train' if 'train' in str(self.data_path) else 'valid'
            print(f"[DEBUG] Загрузка {data_type} данных из {len(video_files)} видео")
            
            self.video_paths = [str(path) for path in video_files]
            self.labels = [None] * len(video_files)
            self.video_count = len(video_files)
            
            print(f"[DEBUG] Пути к видео:")
            for path in self.video_paths:
                print(f"  - {path}")
            
            return self.video_paths
            
        except Exception as e:
            print(f"[ERROR] Ошибка при загрузке видео: {str(e)}")
            raise

    def _load_annotations(self, video_path: str) -> np.ndarray:
        """
        Загрузка аннотаций для видео
        
        Args:
            video_path: путь к видео файлу
            
        Returns:
            np.ndarray: массив меток для каждого кадра
            
        Raises:
            InvalidAnnotationError: если формат аннотаций некорректен
        """
        try:
            # Получаем путь к файлу аннотаций
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            annotation_path = os.path.join(os.path.dirname(video_path), 'annotations', f'{base_name}.json')
            
            if not os.path.exists(annotation_path):
                logger.warning(f"Аннотации не найдены для {video_path}")
                return np.zeros(self._get_video_info(video_path).total_frames)
            
            with open(annotation_path, 'r') as f:
                annotations = json.load(f)
            
            # Проверяем формат аннотаций
            if not isinstance(annotations, dict) or 'frames' not in annotations:
                raise InvalidAnnotationError(f"Некорректный формат аннотаций в {annotation_path}")
            
            # Создаем массив меток
            total_frames = self._get_video_info(video_path).total_frames
            labels = np.zeros(total_frames)
            
            # Заполняем метки
            for frame_info in annotations['frames']:
                if not isinstance(frame_info, dict) or 'frame' not in frame_info or 'label' not in frame_info:
                    raise InvalidAnnotationError(f"Некорректный формат кадра в {annotation_path}")
                
                frame_idx = frame_info['frame']
                if not isinstance(frame_idx, int) or frame_idx < 0 or frame_idx >= total_frames:
                    raise InvalidAnnotationError(f"Некорректный индекс кадра {frame_idx} в {annotation_path}")
                
                label = frame_info['label']
                if not isinstance(label, (int, float)) or label not in [0, 1]:
                    raise InvalidAnnotationError(f"Некорректная метка {label} в {annotation_path}")
                
                labels[frame_idx] = label
            
            return labels
            
        except json.JSONDecodeError as e:
            raise InvalidAnnotationError(f"Ошибка при чтении JSON файла {annotation_path}: {str(e)}")
        except Exception as e:
            raise InvalidAnnotationError(f"Ошибка при загрузке аннотаций для {video_path}: {str(e)}")

    def create_sequences(self, video_path: str, sequence_length: int, target_size: Optional[Tuple[int, int]] = None) -> Tuple[List[np.ndarray], List[int]]:
        """
        Создание последовательностей из видео
        
        Args:
            video_path: путь к видео файлу
            sequence_length: длина последовательности
            target_size: размер кадра
            
        Returns:
            Tuple[List[np.ndarray], List[int]]: кортеж (список последовательностей, список меток)
            
        Raises:
            CorruptedVideoError: если видео повреждено
            InvalidAnnotationError: если формат аннотаций некорректен
        """
        try:
            # Загружаем видео
            cap, total_frames = self.load_video(video_path)
            
            # Загружаем аннотации
            frame_labels = self._load_annotations(video_path)
            
            # Проверяем соответствие размеров
            if len(frame_labels) != total_frames:
                raise InvalidAnnotationError(f"Количество меток ({len(frame_labels)}) не соответствует количеству кадров ({total_frames})")
            
            sequences = []
            labels = []
            
            # Создаем последовательности
            for i in range(0, total_frames - sequence_length + 1, sequence_length):
                sequence = []
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                
                for _ in range(sequence_length):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if target_size:
                        frame = cv2.resize(frame, target_size)
                    sequence.append(frame)
                
                if len(sequence) == sequence_length:
                    sequences.append(np.array(sequence))
                    labels.append(frame_labels[i])
                
                # Очищаем память каждые 10 последовательностей
                if len(sequences) % 10 == 0:
                    gc.collect()
            
            return sequences, labels
            
        except Exception as e:
            logger.error(f"Ошибка при создании последовательностей: {str(e)}")
            raise
        finally:
            if cap is not None:
                cap.release()

    def _get_sequence(self, cap: cv2.VideoCapture, video_path: str, sequence_length: int, target_size: Optional[Tuple[int, int]] = None,
                     one_hot: bool = True, force_positive: bool = False) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Получение последовательности кадров из видео
        
        Args:
            cap: объект VideoCapture
            video_path: путь к видео файлу
            sequence_length: длина последовательности
            target_size: размер кадра
            one_hot: использовать one-hot encoding для меток
            force_positive: принудительно брать положительные примеры
            
        Returns:
            Tuple[Optional[np.ndarray], Optional[np.ndarray]]: кортеж (последовательность, метка) или (None, None)
        """
        try:
            # Получаем текущий индекс кадра
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.debug(f"Получение последовательности: кадр {current_frame}/{total_frames}")
            
            # Проверяем, что последовательность не выходит за границы
            if current_frame + sequence_length > total_frames:
                logger.debug(f"Последовательность выходит за границы: {current_frame + sequence_length} > {total_frames}")
                return None, None
            
            # Получаем множество использованных кадров для текущего видео
            if video_path not in self.used_frames_cache:
                self.used_frames_cache[video_path] = set()
            
            used_frames = self.used_frames_cache[video_path]
            
            # Проверяем, не пересекается ли последовательность с уже использованными кадрами
            if any(frame in used_frames for frame in range(current_frame, current_frame + sequence_length)):
                logger.debug(f"Последовательность пересекается с использованными кадрами: {current_frame}-{current_frame + sequence_length}")
                return None, None
            
            sequence = []
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            
            # Собираем последовательность
            for i in range(sequence_length):
                ret, frame = cap.read()
                if not ret:
                    logger.error(f"Не удалось прочитать кадр {current_frame + i}")
                    raise CorruptedVideoError(f"Не удалось прочитать кадр {current_frame + i}")
                if target_size:
                    frame = cv2.resize(frame, target_size)
                sequence.append(frame)
            
            # Отмечаем использованные кадры
            used_frames.update(range(current_frame, current_frame + sequence_length))
            
            # Создаем метку (в данном случае просто 0, так как метки загружаются отдельно)
            label = np.zeros(2) if one_hot else 0
            
            logger.debug(f"Последовательность успешно получена: {len(sequence)} кадров")
            return np.array(sequence), label
            
        except Exception as e:
            logger.error(f"Ошибка при получении последовательности: {str(e)}")
            return None, None

    def get_batch(self, batch_size: Optional[int] = None, sequence_length: Optional[int] = None,
                 target_size: Optional[Tuple[int, int]] = None, one_hot: bool = True,
                 max_sequences_per_video: Optional[int] = None, force_positive: bool = False) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Получение батча данных
        
        Args:
            batch_size: размер батча (если None, используется значение из конфига)
            sequence_length: длина последовательности (если None, используется значение из конфига)
            target_size: размер кадра (если None, используется значение из конфига)
            one_hot: использовать one-hot encoding для меток
            max_sequences_per_video: максимальное количество последовательностей на видео
            force_positive: принудительно брать положительные примеры
            
        Returns:
            Optional[Tuple[np.ndarray, np.ndarray]]: кортеж (X_batch, y_batch) или None, если батч не удалось собрать
        """
        try:
            # Используем значения из конфига, если параметры не указаны
            batch_size = batch_size or self.batch_size
            sequence_length = sequence_length or self.sequence_length
            target_size = target_size or self.target_size
            max_sequences_per_video = max_sequences_per_video or self.max_sequences_per_video
            
            logger.debug(f"\nПолучение батча (batch_size={batch_size}, sequence_length={sequence_length})")
            logger.debug(f"Текущее видео: {self.current_video_index}/{len(self.video_paths)}")
            logger.debug(f"Текущий кадр: {self.current_frame_index}")
            logger.debug(f"Обработанные видео: {len(self.processed_videos)}/{len(self.video_paths)}")
            
            # Очищаем кэш только при начале новой эпохи и только если все видео обработаны
            if self.current_batch == 0 and len(self.processed_videos) >= len(self.video_paths):
                logger.debug("Начало новой эпохи - очистка кэшей")
                self.clear_cache()
                self.processed_videos.clear()
                self.current_video_index = 0
                self.current_frame_index = 0
            
            # Проверяем, все ли видео обработаны
            if len(self.processed_videos) >= len(self.video_paths):
                logger.debug("Все видео обработаны - конец эпохи")
                return None
            
            # Счетчик попыток найти необработанное видео
            attempts = 0
            max_attempts = len(self.video_paths) * 2
            
            while attempts < max_attempts:
                attempts += 1
                logger.debug(f"Попытка {attempts}/{max_attempts}")
                
                # Проверяем, что индекс видео не выходит за границы
                if self.current_video_index >= len(self.video_paths):
                    logger.debug("Достигнут конец списка видео - начинаем новую эпоху")
                    self.clear_cache()
                    self.processed_videos.clear()
                    self.current_video_index = 0
                    self.current_frame_index = 0
                    continue
                
                # Получаем текущее видео
                video_path = self.video_paths[self.current_video_index]
                logger.debug(f"Обработка видео: {video_path}")
                
                # Если видео уже обработано, переходим к следующему
                if video_path in self.processed_videos:
                    logger.debug(f"Видео {video_path} уже обработано - переходим к следующему")
                    if self.current_video_index < len(self.video_paths) - 1:
                        self.current_video_index += 1
                    else:
                        self.current_video_index = 0
                    self.current_frame_index = 0
                    continue
                
                # Получаем информацию о видео
                info = self._get_video_info(video_path)
                if not info.exists:
                    logger.error(f"Видеофайл не найден: {video_path}")
                    self.processed_videos.add(video_path)
                    if self.current_video_index < len(self.video_paths) - 1:
                        self.current_video_index += 1
                    else:
                        self.current_video_index = 0
                    self.current_frame_index = 0
                    continue
                
                logger.debug(f"Информация о видео: {info}")
                
                # Проверяем, есть ли видео в кэше
                if video_path in self.video_cache:
                    cap = self.video_cache[video_path]
                    logger.debug("Видео загружено из кэша")
                else:
                    # Очищаем предыдущее видео из кэша если оно есть
                    if hasattr(self, 'current_cap') and self.current_cap is not None:
                        self.current_cap.release()
                    
                    logger.debug("Открываем видео через OpenCV")
                    cap = cv2.VideoCapture(video_path)
                    if not cap.isOpened():
                        logger.error(f"Не удалось открыть видео: {video_path}")
                        self.processed_videos.add(video_path)
                        if self.current_video_index < len(self.video_paths) - 1:
                            self.current_video_index += 1
                        else:
                            self.current_video_index = 0
                        self.current_frame_index = 0
                        continue
                    
                    self.video_cache[video_path] = cap
                    self.current_cap = cap
                
                # Инициализируем кэш использованных кадров для текущего видео
                if video_path not in self.used_frames_cache:
                    self.used_frames_cache[video_path] = set()
                
                # Проверяем, нужно ли перейти к следующему видео
                if self.current_frame_index >= info.total_frames - sequence_length:
                    logger.debug(f"Достигнут конец видео {self.current_video_index}")
                    # Проверяем процент использованных кадров
                    if video_path in self.used_frames_cache:
                        used_frames = self.used_frames_cache[video_path]
                        used_percentage = len(used_frames) / info.total_frames * 100
                        if used_percentage > self.cache_cleanup_threshold:
                            logger.debug(f"Видео использовано на {used_percentage:.1f}% - помечаем как обработанное")
                            self.processed_videos.add(video_path)
                            # Очищаем кэш для текущего видео
                            if video_path in self.video_cache:
                                cap = self.video_cache.pop(video_path)
                                cap.release()
                            if video_path in self.used_frames_cache:
                                del self.used_frames_cache[video_path]
                            if video_path in self.positive_indices_cache:
                                del self.positive_indices_cache[video_path]
                        else:
                            logger.debug(f"Видео использовано на {used_percentage:.1f}% - начинаем сначала")
                            self.current_frame_index = 0
                            continue
                    
                    # Переходим к следующему видео
                    if self.current_video_index < len(self.video_paths) - 1:
                        self.current_video_index += 1
                    else:
                        self.current_video_index = 0
                    self.current_frame_index = 0
                    continue
                
                # Собираем батч
                X_batch = []
                y_batch = []
                
                for i in range(batch_size):
                    logger.debug(f"Получение последовательности {i+1}/{batch_size}")
                    # Получаем последовательность
                    sequence, label = self._get_sequence(
                        cap,
                        video_path,
                        sequence_length,
                        target_size,
                        one_hot,
                        force_positive
                    )
                    
                    if sequence is None:
                        logger.debug("Не удалось получить последовательность")
                        # Проверяем, действительно ли видео полностью использовано
                        if video_path in self.used_frames_cache:
                            used_frames = self.used_frames_cache[video_path]
                            used_percentage = len(used_frames) / info.total_frames * 100
                            if used_percentage > self.cache_cleanup_threshold:
                                logger.debug(f"Видео использовано на {used_percentage:.1f}% - помечаем как обработанное")
                                self.processed_videos.add(video_path)
                                # Очищаем кэш для текущего видео
                                if video_path in self.video_cache:
                                    cap = self.video_cache.pop(video_path)
                                    cap.release()
                                if video_path in self.used_frames_cache:
                                    del self.used_frames_cache[video_path]
                                if video_path in self.positive_indices_cache:
                                    del self.positive_indices_cache[video_path]
                                
                                # Переходим к следующему видео
                                if self.current_video_index < len(self.video_paths) - 1:
                                    self.current_video_index += 1
                                else:
                                    self.current_video_index = 0
                                self.current_frame_index = 0
                                break
                            else:
                                logger.debug(f"Видео использовано на {used_percentage:.1f}% - продолжаем")
                                # Если не можем найти непересекающуюся последовательность,
                                # очищаем кэш использованных кадров и начинаем заново
                                if video_path in self.used_frames_cache:
                                    del self.used_frames_cache[video_path]
                                self.current_frame_index = 0
                                continue
                        else:
                            logger.debug("Нет информации об использованных кадрах - продолжаем")
                            self.current_frame_index = 0
                            continue
                    
                    X_batch.append(sequence)
                    y_batch.append(label)
                
                # Если батч собран успешно
                if len(X_batch) == batch_size:
                    # Увеличиваем счетчик батчей
                    self.current_batch += 1
                    
                    # Конвертируем в numpy массивы
                    X_batch = np.array(X_batch)
                    y_batch = np.array(y_batch)
                    
                    logger.debug(f"Батч успешно собран: {X_batch.shape}, {y_batch.shape}")
                    
                    # Проверяем процент использованных кадров
                    if video_path in self.used_frames_cache:
                        used_frames = self.used_frames_cache[video_path]
                        used_percentage = len(used_frames) / info.total_frames * 100
                        if used_percentage > self.cache_cleanup_threshold:
                            logger.debug(f"Видео использовано на {used_percentage:.1f}% - помечаем как обработанное")
                            self.processed_videos.add(video_path)
                    
                    # Переходим к следующему видео после успешного сбора батча
                    if self.current_video_index < len(self.video_paths) - 1:
                        self.current_video_index += 1
                    else:
                        self.current_video_index = 0
                    self.current_frame_index = 0
                    
                    return X_batch, y_batch
                
                # Если батч не собран полностью, продолжаем с следующим видео
                continue
            
            logger.debug("Не удалось найти необработанное видео после проверки всех видео")
            return None
            
        except Exception as e:
            logger.error(f"Ошибка при получении батча: {str(e)}")
            raise

    def data_generator(self, force_positive: bool = True) -> Generator[Tuple[tf.Tensor, tf.Tensor], None, None]:
        """
        Генератор данных с sampling положительных примеров
        
        Args:
            force_positive: принудительно брать положительные примеры
            
        Yields:
            Tuple[tf.Tensor, tf.Tensor]: кортеж (X_batch, y_batch)
            
        Raises:
            CorruptedVideoError: если видео повреждено
            InvalidAnnotationError: если формат аннотаций некорректен
        """
        try:
            logger.info("\n===== Запуск генератора данных =====")
            logger.info(f"Количество видео для обработки: {len(self.video_paths)}")
            
            # Счетчик попыток найти необработанное видео
            video_attempts = 0
            max_video_attempts = len(self.video_paths)
            
            while True:
                # Проверяем, все ли видео обработаны
                if len(self.processed_videos) >= len(self.video_paths):
                    logger.debug("Все видео обработаны - конец эпохи")
                    break
                
                # Проверяем количество попыток найти необработанное видео
                if video_attempts >= max_video_attempts:
                    logger.debug("Достигнуто максимальное количество попыток найти необработанное видео")
                    break
                
                batch_data = self.get_batch(
                    batch_size=self.batch_size,
                    sequence_length=self.sequence_length,
                    target_size=self.target_size,
                    one_hot=True,
                    max_sequences_per_video=self.max_sequences_per_video,
                    force_positive=force_positive
                )
                
                if batch_data is None:
                    logger.debug("Не удалось получить батч - увеличиваем счетчик попыток")
                    video_attempts += 1
                    continue
                
                # Сбрасываем счетчик попыток при успешном получении батча
                video_attempts = 0
                
                X, y = batch_data
                if X is None or y is None or X.shape[0] == 0 or y.shape[0] == 0:
                    logger.warning("Получен пустой батч")
                    continue
                
                try:
                    num_positive = int((y[...,1] == 1).sum())
                    logger.debug(f"В батче положительных примеров (class 1): {num_positive}")
                    
                    # Конвертируем в тензоры с оптимизацией памяти
                    x = tf.convert_to_tensor(X, dtype=tf.float32)
                    y_tensor = tf.convert_to_tensor(y, dtype=tf.float32)
                    
                    # Очищаем память
                    del X
                    del y
                    gc.collect()
                    
                    yield (x, y_tensor)
                    
                except Exception as e:
                    logger.error(f"Ошибка при обработке батча: {str(e)}")
                    continue
            
            logger.debug("Завершение генератора данных")
            return
                
        except Exception as e:
            logger.error(f"Ошибка в генераторе данных: {str(e)}")
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
