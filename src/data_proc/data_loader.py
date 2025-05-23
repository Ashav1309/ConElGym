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
import time

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
        self.annotations_cache: Dict[str, np.ndarray] = {}  # Кэш для аннотаций
        self.open_videos: Set[str] = set()  # Множество открытых видео
        self.processed_video_paths: Set[str] = set()  # Множество обработанных видео (полные пути)
        self.sequence_counter: Dict[str, int] = {}    # Счетчик последовательностей для каждого видео
        self.used_sequences: Set[str] = set()         # Множество использованных последовательностей
        
        # Счетчики для отслеживания прогресса
        self.total_processed_videos = 0  # Общее количество обработанных видео
        self.total_processed_frames = 0  # Общее количество обработанных кадров
        self.total_processed_sequences = 0  # Общее количество обработанных последовательностей
        
        self.data_path = Path(os.path.abspath(data_path))
        self.max_videos = max_videos or Config.MAX_VIDEOS
        self.video_paths: List[str] = []
        self.labels: List[Optional[str]] = []
        self.video_count = 0
        
        # Параметры из конфигурации
        self.batch_size = Config.BATCH_SIZE
        self.sequence_length = Config.SEQUENCE_LENGTH
        self.max_sequences_per_video = Config.MAX_SEQUENCES_PER_VIDEO
        self.target_size = Config.INPUT_SIZE
        self.frame_size = self.target_size[0]  # Используем первый элемент target_size как размер кадра
        
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
        try:
            # Закрываем все открытые видео
            for cap in self.video_cache.values():
                if cap is not None:
                    cap.release()
            
            # Очищаем временные кэши
            self.video_cache.clear()
            self.used_frames_cache.clear()
            self.positive_indices_cache.clear()
            self.file_info_cache.clear()
            self.open_videos.clear()
            self.sequence_counter.clear()       # Очищаем счетчик последовательностей
            self.used_sequences.clear()         # Очищаем множество использованных последовательностей
            self.annotations_cache.clear()      # Очищаем кэш аннотаций
            
            # НЕ очищаем processed_video_paths, так как это общий список обработанных видео
            
            # Сбрасываем счетчики
            self.total_processed_videos = len(self.processed_video_paths)  # Обновляем из processed_video_paths
            self.total_processed_frames = 0
            self.total_processed_sequences = 0
            self.current_video_index = 0
            self.current_frame_index = 0
            self.current_batch = 0
            
            # Принудительная очистка памяти
            gc.collect()
            
            logger.debug("Временные кэши очищены")
            logger.debug(f"Обработано видео: {len(self.processed_video_paths)}/{self.total_videos} ({len(self.processed_video_paths)/self.total_videos*100:.1f}%)")
        except Exception as e:
            logger.error(f"Ошибка при очистке кэшей: {str(e)}")

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
            
            # Проверяем минимальную длину видео
            if info.total_frames < self.sequence_length:
                raise CorruptedVideoError(f"Видео слишком короткое: {info.total_frames} кадров < {self.sequence_length} (минимальная длина последовательности)")
            
            # Открываем видео
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise CorruptedVideoError(f"Не удалось открыть видео: {video_path}")
            
            # Проверяем, что можем читать кадры
            ret, frame = cap.read()
            if not ret or frame is None:
                raise CorruptedVideoError(f"Не удалось прочитать первый кадр видео: {video_path}")
            
            # Возвращаемся к началу видео
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
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
        """Загрузка видео из директории с поддержкой лимита MAX_VIDEOS на одну сессию"""
        try:
            video_files = []
            logger.debug(f"[DEBUG] Поиск видео в директории: {self.data_path}")
            
            for ext in ['.mp4']:
                pattern = f'*{ext}'
                logger.debug(f"[DEBUG] Поиск по шаблону: {pattern}")
                files = list(self.data_path.glob(pattern))
                logger.debug(f"[DEBUG] Найдено {len(files)} файлов с расширением {ext}")
                video_files.extend(files)
            
            if not video_files:
                logger.error(f"[ERROR] Видео не найдены в {self.data_path}")
                raise ValueError(f"Видео не найдены в {self.data_path}")
            
            # Сортируем видео по имени для воспроизводимости
            video_files.sort()
            
            # Сохраняем все пути к видео
            self.all_video_paths = [str(path) for path in video_files]
            self.total_videos = len(self.all_video_paths)
            
            # Загружаем только MAX_VIDEOS видео за раз
            self.current_video_index = 0
            self._load_video_chunk()
            
            data_type = 'train' if 'train' in str(self.data_path) else 'valid'
            logger.debug(f"[DEBUG] Всего найдено {self.total_videos} видео")
            logger.debug(f"[DEBUG] Загружено {self.video_count} видео из {self.total_videos} для текущей сессии")
            logger.debug(f"[DEBUG] Пути к видео: {self.all_video_paths}")
        except Exception as e:
            logger.error(f"[ERROR] Ошибка при загрузке видео: {str(e)}")
            raise

    def _load_video_chunk(self):
        """Загрузка следующей порции видео"""
        try:
            start_idx = self.current_video_index
            end_idx = min(start_idx + self.max_videos, self.total_videos)
            
            if start_idx >= self.total_videos:
                print("[DEBUG] Достигнут конец списка видео")
                return
                
            print(f"[DEBUG] Загрузка порции видео {start_idx+1}-{end_idx} из {self.total_videos}")
            self.video_paths = self.all_video_paths[start_idx:end_idx]
            self.current_video_index = end_idx
            self.processed_video_paths = set()  # Сбрасываем только для текущей порции
            
        except Exception as e:
            print(f"[ERROR] Ошибка при загрузке порции видео: {str(e)}")
            import traceback
            traceback.print_exc()

    def _load_annotations(self, video_path: str) -> np.ndarray:
        """
        Загружает аннотации для видео и создает метки кадров
        """
        try:
            # Получаем информацию о видео
            video_info = self._get_video_info(video_path)
            total_frames = video_info.total_frames
            
            # Создаем пустые метки (все кадры - фон)
            labels = np.zeros((total_frames, 2))  # 2 класса: фон, действие
            labels[:, 0] = 1  # По умолчанию все кадры - фон [1, 0]
            
            # Получаем путь к файлу аннотаций
            annotation_path = self._get_annotation_path(video_path)
            
            # Проверяем наличие аннотаций в кэше
            if video_path in self.annotations_cache:
                logger.debug(f"[DEBUG] Аннотации для {os.path.basename(video_path)} найдены в кэше")
                return self.annotations_cache[video_path]
            
            # Если файл аннотаций не существует, возвращаем пустые метки
            if not os.path.exists(annotation_path):
                logger.debug(f"[DEBUG] Файл аннотаций не найден: {annotation_path}")
                empty_labels = np.zeros((total_frames, 2))
                empty_labels[:, 0] = 1  # Все кадры - фон [1, 0]
                return empty_labels
            
            # Загружаем аннотации
            with open(annotation_path, 'r') as f:
                annotations = json.load(f)
                
            # Обрабатываем каждую аннотацию
            for annotation in annotations['annotations']:
                start_frame = annotation['start_frame']
                end_frame = annotation['end_frame']
                
                # Отмечаем кадры как действие [0, 1]
                for frame_idx in range(start_frame, end_frame + 1):
                    if frame_idx < total_frames:
                        labels[frame_idx] = [0, 1]  # Действие
            
            logger.debug(f"[DEBUG] Статистика меток для {os.path.basename(video_path)}:")
            logger.debug(f"  - Всего кадров: {total_frames}")
            logger.debug(f"  - Кадры с фоном: {np.sum(labels[:, 0] == 1)}")
            logger.debug(f"  - Кадры с действием: {np.sum(labels[:, 1] == 1)}")
            logger.debug(f"  - Сумма всех меток: {np.sum(labels)}")
            
            # Сохраняем в кэш
            self.annotations_cache[video_path] = labels
            logger.debug(f"[DEBUG] Аннотации для {os.path.basename(video_path)} сохранены в кэш")
            
            return labels
            
        except Exception as e:
            logger.error(f"[ERROR] Ошибка при загрузке аннотаций для {video_path}: {str(e)}")
            # В случае ошибки возвращаем пустые метки
            empty_labels = np.zeros((total_frames, 2))
            empty_labels[:, 0] = 1  # Все кадры - фон [1, 0]
            return empty_labels

    def create_sequences(self, video_path: str, labels: np.ndarray, sequence_length: int = Config.SEQUENCE_LENGTH, 
                        max_sequences: int = 200, step: int = 16, force_positive: bool = False) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Создает последовательности кадров из видео.
        """
        try:
            print(f"\n[DEBUG] Обработка видео: {os.path.basename(video_path)}")
            
            # Проверяем наличие действий в видео
            has_actions = np.any(labels[:, 1] == 1)
            if not has_actions:
                print(f"[DEBUG] В видео нет действий, пропускаем")
                return None, None
            
            # Загружаем видео и получаем количество кадров
            cap, total_frames = self.load_video(video_path)
            
            if cap is None:
                print("[DEBUG] Не удалось загрузить видео")
                return None, None
                
            # Проверяем, что у нас достаточно кадров
            if total_frames < sequence_length:
                print(f"[DEBUG] Видео слишком короткое: {total_frames} кадров < {sequence_length}")
                return None, None
            
            # Находим сегменты с действиями
            action_segments = []
            in_action = False
            start_frame = 0
            processed_frames = 0
            
            for i in range(total_frames):
                if labels[i, 1] == 1 and not in_action:
                    in_action = True
                    start_frame = i
                elif labels[i, 1] == 0 and in_action:
                    in_action = False
                    action_segments.append((start_frame, i - 1))
                processed_frames += 1
            
            if in_action:
                action_segments.append((start_frame, total_frames - 1))
            
            # Создаем последовательности
            action_dominant_sequences = []  # Последовательности с преобладанием действия
            action_dominant_labels = []
            background_dominant_sequences = []  # Последовательности с преобладанием фона
            background_dominant_labels = []
            
            # Ограничиваем количество последовательностей каждого типа
            max_positive = int(max_sequences * 0.6)  # 75% положительных
            max_negative = max_sequences - max_positive  # 25% отрицательных
            
            # Сначала создаем положительные последовательности из сегментов с действиями
            sequence_attempts = 0
            max_sequence_attempts = min(max_positive * 2, 100)
            
            for start_frame, end_frame in action_segments:
                if sequence_attempts >= max_sequence_attempts or len(action_dominant_sequences) >= max_positive:
                    break
                
                # Вычисляем возможные начальные позиции для последовательностей
                possible_starts = []
                for i in range(start_frame, end_frame - sequence_length + 2, step):
                    sequence_label = labels[i:i + sequence_length]
                    if np.any(sequence_label[:, 1] == 1):
                        possible_starts.append(i)
                    processed_frames += sequence_length
                
                # Создаем последовательности из этого сегмента
                for start_idx in possible_starts:
                    if sequence_attempts >= max_sequence_attempts or len(action_dominant_sequences) >= max_positive:
                        break
                    
                    # Читаем кадры для последовательности
                    frames = []
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
                    
                    for _ in range(sequence_length):
                        ret, frame = cap.read()
                        if not ret:
                            break
                        try:
                            frame = cv2.resize(frame, (self.frame_size, self.frame_size))
                            frames.append(frame)
                        except Exception as e:
                            break
                    
                    if len(frames) == sequence_length:
                        frames_array = np.array(frames)
                        sequence_label = labels[start_idx:start_idx + sequence_length]
                        action_ratio = np.mean(sequence_label[:, 1])
                        
                        if action_ratio > 0.3:
                            action_dominant_sequences.append(frames_array)
                            action_dominant_labels.append(sequence_label)
                    
                    sequence_attempts += 1
            
            # Теперь создаем отрицательные последовательности
            sequence_attempts = 0
            max_sequence_attempts = min(max_negative * 2, 100)
            
            # Находим кадры с минимальным действием
            min_action_frames = np.where(labels[:, 1] < 0.1)[0]
            np.random.shuffle(min_action_frames)
            
            # Создаем отрицательные последовательности
            for frame_idx in min_action_frames[::step]:
                if sequence_attempts >= max_sequence_attempts or len(background_dominant_sequences) >= max_negative:
                    break
                    
                if frame_idx + sequence_length > total_frames:
                    continue
                
                # Проверяем, что все кадры в последовательности имеют минимальное действие
                sequence_label = labels[frame_idx:frame_idx + sequence_length]
                if np.any(sequence_label[:, 1] > 0.1):
                    continue
                
                # Читаем кадры
                frames = []
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                
                for _ in range(sequence_length):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    try:
                        frame = cv2.resize(frame, (self.frame_size, self.frame_size))
                        frames.append(frame)
                    except Exception as e:
                        break
                
                if len(frames) == sequence_length:
                    frames_array = np.array(frames)
                    action_ratio = np.mean(sequence_label[:, 1])
                    
                    if action_ratio < 0.1:
                        background_dominant_sequences.append(frames_array)
                        background_dominant_labels.append(sequence_label)
                
                sequence_attempts += 1
            
            cap.release()
            
            print(f"[DEBUG] Результаты обработки видео {os.path.basename(video_path)}:")
            print(f"  - Положительных последовательностей: {len(action_dominant_sequences)}")
            print(f"  - Отрицательных последовательностей: {len(background_dominant_sequences)}")
            
            # Если нет последовательностей, возвращаем None
            if not action_dominant_sequences and not background_dominant_sequences:
                print("[DEBUG] Не удалось создать последовательности")
                return None, None
            
            # Объединяем последовательности
            all_sequences = action_dominant_sequences + background_dominant_sequences
            all_labels = action_dominant_labels + background_dominant_labels
            
            # Перемешиваем
            indices = np.random.permutation(len(all_sequences))
            all_sequences = [all_sequences[i] for i in indices]
            all_labels = [all_labels[i] for i in indices]
            
            # Выбираем последовательность с учетом баланса 75/25
            if force_positive and action_dominant_sequences:
                # Если требуется положительная последовательность и она есть
                idx = np.random.randint(len(action_dominant_sequences))
                X = action_dominant_sequences[idx]
                y = action_dominant_labels[idx]
            else:
                # Иначе выбираем последовательность с учетом баланса
                if len(action_dominant_sequences) > 0 and len(background_dominant_sequences) > 0:
                    # Если есть оба типа последовательностей, выбираем с вероятностью 75/25
                    if np.random.random() < 0.6:
                        idx = np.random.randint(len(action_dominant_sequences))
                        X = action_dominant_sequences[idx]
                        y = action_dominant_labels[idx]
                    else:
                        idx = np.random.randint(len(background_dominant_sequences))
                        X = background_dominant_sequences[idx]
                        y = background_dominant_labels[idx]
                else:
                    # Если есть только один тип, выбираем из него
                    idx = np.random.randint(len(all_sequences))
                    X = all_sequences[idx]
                    y = all_labels[idx]
            
            return X, y
            
        except Exception as e:
            print(f"[ERROR] Ошибка при создании последовательности: {str(e)}")
            if cap is not None:
                cap.release()
            return None, None

    def _get_random_video(self) -> Optional[str]:
        """
        Получение случайного видео для обработки с проверкой на повторное использование
        
        Returns:
            Optional[str]: путь к видео или None, если все видео обработаны
        """
        try:
            max_attempts = len(self.video_paths) * 2
            attempts = 0
            reset_count = 0
            max_resets = 5
            
            is_train = 'train' in str(self.data_path)
            
            print(f"\n[DEBUG] Начало поиска случайного видео")
            print(f"[DEBUG] Тип данных: {'train' if is_train else 'valid'}")
            print(f"[DEBUG] Всего видео в датасете: {self.total_videos}")
            print(f"[DEBUG] Видео в текущей порции: {len(self.video_paths)}")
            print(f"[DEBUG] Обработано видео: {self.total_processed_videos}/{self.total_videos} ({self.total_processed_videos/self.total_videos*100:.1f}%)")
            print(f"[DEBUG] Текущий индекс видео: {self.current_video_index}")
            
            while attempts < max_attempts:
                attempts += 1
                print(f"\n[DEBUG] Попытка {attempts}/{max_attempts}")
                print(f"[DEBUG] processed_video_paths: {len(self.processed_video_paths)}")
                print(f"[DEBUG] video_paths: {len(self.video_paths)}")
                
                if len(self.processed_video_paths) >= len(self.video_paths):
                    print("[DEBUG] Все видео обработаны, загружаем новую порцию")
                    print(f"[DEBUG] До загрузки новой порции:")
                    print(f"  - processed_video_paths: {len(self.processed_video_paths)}")
                    print(f"  - video_paths: {len(self.video_paths)}")
                    print(f"  - current_video_index: {self.current_video_index}")
                    
                    self._load_video_chunk()
                    
                    print(f"[DEBUG] После загрузки новой порции:")
                    print(f"  - processed_video_paths: {len(self.processed_video_paths)}")
                    print(f"  - video_paths: {len(self.video_paths)}")
                    print(f"  - current_video_index: {self.current_video_index}")
                    
                    if self.current_video_index >= self.total_videos:
                        print("[DEBUG] Достигнут конец списка видео")
                        return None
                        
                    continue
                
                available_videos = [v for v in self.video_paths if v not in self.processed_video_paths]
                print(f"[DEBUG] Доступные видео: {len(available_videos)}")
                
                if not available_videos:
                    print("[DEBUG] Нет доступных видео")
                    if reset_count < max_resets:
                        reset_count += 1
                        print(f"[DEBUG] Сброс счетчиков (попытка {reset_count}/{max_resets})")
                        print("[DEBUG] Загружаем следующую порцию видео")
                        self._load_video_chunk()
                        
                        if self.current_video_index >= self.total_videos:
                            print("[DEBUG] Достигнут конец списка видео")
                            return None
                            
                        continue
                    else:
                        print("[DEBUG] Достигнут лимит сбросов")
                        return None
                
                print(f"[DEBUG] Выбираем случайное видео из {len(available_videos)} доступных")
                video_path = np.random.choice(available_videos)
                print(f"[DEBUG] Выбрано видео: {os.path.basename(video_path)}")
                
                if not os.path.exists(video_path):
                    print(f"[DEBUG] Видео не найдено: {video_path}")
                    continue
                
                self.processed_video_paths.add(video_path)
                self.total_processed_videos += 1  # Увеличиваем общий счетчик
                return video_path
                
        except Exception as e:
            print(f"[ERROR] Ошибка при получении случайного видео: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _get_sequence(self, sequence_length, target_size, force_positive=False, is_validation=False):
        """
        Получение последовательности кадров с предотвращением дублирования
        """
        video_path = self._get_random_video()
        if video_path is None:
            return None, None

        # Проверяем, есть ли аннотации в кэше
        if video_path not in self.annotations_cache:
            # Если аннотаций нет в кэше, загружаем их
            try:
                labels = self._load_annotations(video_path)  # Сохраняем результат загрузки
                if labels is None:  # Если загрузка не удалась
                    total_frames = self._get_video_info(video_path).total_frames
                    empty_labels = np.zeros((total_frames, 2))
                    self.annotations_cache[video_path] = empty_labels
                    self.processed_video_paths.add(video_path)  # Добавляем видео в обработанные
                    print(f"[DEBUG] Обработано видео: {len(self.processed_video_paths)}/{self.total_videos} ({len(self.processed_video_paths)/self.total_videos*100:.1f}%)")
                    return None, None
            except Exception as e:
                # Создаем пустые аннотации и добавляем в кэш
                total_frames = self._get_video_info(video_path).total_frames
                empty_labels = np.zeros((total_frames, 2))
                self.annotations_cache[video_path] = empty_labels
                self.processed_video_paths.add(video_path)  # Добавляем видео в обработанные
                print(f"[DEBUG] Обработано видео: {len(self.processed_video_paths)}/{self.total_videos} ({len(self.processed_video_paths)/self.total_videos*100:.1f}%)")
                return None, None

        # Инициализируем счетчик для видео, если его еще нет
        if video_path not in self.sequence_counter:
            self.sequence_counter[video_path] = 0

        # Получаем последовательность
        try:
            X_seq, y_seq = self.create_sequences(
                video_path=video_path,
                labels=self.annotations_cache[video_path],
                sequence_length=sequence_length,
                max_sequences=self.max_sequences_per_video,
                step=16,  # Используем шаг 1 для более точного выбора
                force_positive=force_positive  # Используем переданный параметр
            )

            if X_seq is not None and y_seq is not None:
                # Создаём уникальный идентификатор только для одной последовательности
                seq_id = f"{os.path.basename(video_path)}_{self.sequence_counter[video_path]}"
                if seq_id in self.used_sequences:
                    self.processed_video_paths.add(video_path)  # Добавляем видео в обработанные
                    print(f"[DEBUG] Обработано видео: {len(self.processed_video_paths)}/{self.total_videos} ({len(self.processed_video_paths)/self.total_videos*100:.1f}%)")
                    return None, None

                self.used_sequences.add(seq_id)
                self.sequence_counter[video_path] += 1

                # Проверяем, все ли прочитанные кадры использованы
                video_info = self._get_video_info(video_path)
                if video_info:
                    total_readable_frames = len(self.used_frames_cache.get(video_path, set()))
                    if total_readable_frames >= video_info.total_frames - sequence_length:
                        self.processed_video_paths.add(video_path)  # Добавляем видео в обработанные
                        print(f"[DEBUG] Обработано видео: {len(self.processed_video_paths)}/{self.total_videos} ({len(self.processed_video_paths)/self.total_videos*100:.1f}%)")

                # Помечаем видео как обработанное после успешного создания последовательности
                self.processed_video_paths.add(video_path)
                print(f"[DEBUG] Обработано видео: {len(self.processed_video_paths)}/{self.total_videos} ({len(self.processed_video_paths)/self.total_videos*100:.1f}%)")

                return X_seq, y_seq
            else:
                # Если не удалось создать последовательность, помечаем видео как обработанное
                self.processed_video_paths.add(video_path)
                print(f"[DEBUG] Обработано видео: {len(self.processed_video_paths)}/{self.total_videos} ({len(self.processed_video_paths)/self.total_videos*100:.1f}%)")

        except Exception as e:
            self.processed_video_paths.add(video_path)  # Добавляем видео в обработанные при ошибке
            print(f"[DEBUG] Обработано видео: {len(self.processed_video_paths)}/{self.total_videos} ({len(self.processed_video_paths)/self.total_videos*100:.1f}%)")
            return None, None

        return None, None

    def _save_batch_statistics(self, X_batch: np.ndarray, y_batch: np.ndarray, batch_number: int, positive_count: int, negative_count: int, video_path: str):
        """
        Сохранение статистики по батчу
        
        Args:
            X_batch: батч данных
            y_batch: батч меток
            batch_number: номер батча
            positive_count: количество положительных примеров
            negative_count: количество отрицательных примеров
            video_path: путь к видео
        """
        try:
            # Обновляем счетчики
            self.total_processed_sequences += len(X_batch)
            self.total_processed_frames += len(X_batch) * self.sequence_length
            
            # Если это новое видео, увеличиваем счетчик
            if video_path not in self.processed_video_paths:
                self.processed_video_paths.add(video_path)
                self.total_processed_videos += 1
            
            logger.debug(f"[DEBUG] Статистика батча {batch_number}:")
            logger.debug(f"  - Размер батча: {X_batch.shape}")
            logger.debug(f"  - Положительных примеров: {positive_count}")
            logger.debug(f"  - Отрицательных примеров: {negative_count}")
            logger.debug(f"  - Видео: {video_path}")
            logger.debug(f"[DEBUG] Общая статистика обработки:")
            logger.debug(f"  - Обработано видео: {self.total_processed_videos}/{self.total_videos} ({self.total_processed_videos/self.total_videos*100:.1f}%)")
            logger.debug(f"  - Обработано кадров: {self.total_processed_frames}")
            logger.debug(f"  - Обработано последовательностей: {self.total_processed_sequences}")
        except Exception as e:
            logger.warning(f"[WARNING] Ошибка при сохранении статистики батча: {str(e)}")

    def get_batch(self, batch_size, sequence_length, target_size, one_hot=True, max_sequences_per_video=None, force_positive=False, is_validation=False):
        """
        Получение батча данных с исправлением некорректных форм и балансировкой классов
        """
        # Сбрасываем индексы при переключении между наборами данных
        if not hasattr(self, '_last_is_validation') or self._last_is_validation != is_validation:
            self.current_video_index = 0
            self.current_frame_index = 0
            self.current_batch = 0
            self._last_is_validation = is_validation
            print(f"[DEBUG] Сброс индексов при переключении на {'валидационный' if is_validation else 'обучающий'} набор")
        
        X_batch = []
        y_batch = []
        attempts = 0
        max_attempts = batch_size * 10  # Увеличиваем количество попыток
        max_empty_sequences = 20  # Увеличиваем допустимое количество пустых последовательностей
        empty_sequence_count = 0
        
        # Счетчики для балансировки классов
        positive_count = 0
        negative_count = 0
        max_positive = int(batch_size * 0.6)  # 75% положительных
        max_negative = batch_size - max_positive  # 25% отрицательных
        
        # Ожидаемая форма последовательности
        expected_shape = (sequence_length, *target_size, 3) if target_size else (sequence_length,)
        
        print(f"\n[DEBUG] Начало сбора батча {self.current_batch + 1}/{self.total_batches}")
        print(f"[DEBUG] Параметры:")
        print(f"  - batch_size: {batch_size}")
        print(f"  - sequence_length: {sequence_length}")
        print(f"  - target_size: {target_size}")
        print(f"  - force_positive: {force_positive}")
        print(f"  - is_validation: {is_validation}")
        print(f"[DEBUG] Целевые значения: {max_positive} положительных, {max_negative} отрицательных")
        
        last_positive_count = 0
        last_negative_count = 0
        no_progress_count = 0
        while (len(X_batch) < batch_size and 
               (positive_count < max_positive or negative_count < max_negative) and 
               attempts < max_attempts):
            try:
                X_seq, y_seq = self._get_sequence(
                    sequence_length=sequence_length,
                    target_size=target_size,
                    force_positive=force_positive,
                    is_validation=is_validation
                )
                
                if X_seq is not None and y_seq is not None:
                    # Проверяем форму последовательности
                    if X_seq.shape != expected_shape:
                        print(f"[DEBUG] Некорректная форма последовательности: {X_seq.shape}, ожидалось: {expected_shape}")
                        # Исправляем форму последовательности
                        try:
                            # Если у нас лишняя размерность, убираем её
                            if len(X_seq.shape) == 5 and X_seq.shape[0] == 1:
                                X_seq = X_seq[0]  # Убираем первую размерность
                            elif len(X_seq.shape) == 5 and X_seq.shape[1] == sequence_length:
                                X_seq = X_seq[0]  # Берем первую последовательность
                            elif len(X_seq.shape) == 5 and X_seq.shape[0] == sequence_length:
                                X_seq = X_seq.transpose(1, 0, 2, 3)  # Меняем порядок размерностей
                            
                            # Проверяем, что форма теперь правильная
                            if X_seq.shape != expected_shape:
                                print(f"[DEBUG] Не удалось исправить форму последовательности: {X_seq.shape} -> {expected_shape}")
                                empty_sequence_count += 1
                                attempts += 1
                                continue
                                
                            print(f"[DEBUG] Форма после исправления: {X_seq.shape}")
                        except Exception as e:
                            print(f"[DEBUG] Не удалось исправить форму последовательности: {str(e)}")
                            empty_sequence_count += 1
                            attempts += 1
                            continue
                    
                    # Проверяем тип последовательности и добавляем в батч с учетом балансировки
                    is_positive = np.any(y_seq[:, 1] == 1)
                    if is_positive and positive_count < max_positive:
                        X_batch.append(X_seq)
                        y_batch.append(y_seq)
                        positive_count += 1
                        empty_sequence_count = 0
                        print(f"[DEBUG] Добавлена положительная последовательность {len(X_batch)}/{batch_size}")
                        print(f"[DEBUG] Прогресс балансировки: {positive_count}/{max_positive} положительных, {negative_count}/{max_negative} отрицательных")
                    elif not is_positive and negative_count < max_negative:
                        X_batch.append(X_seq)
                        y_batch.append(y_seq)
                        negative_count += 1
                        empty_sequence_count = 0
                        print(f"[DEBUG] Добавлена отрицательная последовательность {len(X_batch)}/{batch_size}")
                        print(f"[DEBUG] Прогресс балансировки: {positive_count}/{max_positive} положительных, {negative_count}/{max_negative} отрицательных")
                    else:
                        # Пропускаем пример, если достигли лимита для его класса
                        attempts += 1
                        continue
                else:
                    empty_sequence_count += 1
                    attempts += 1
                    print(f"[DEBUG] Пустая последовательность (попытка {attempts}/{max_attempts})")
                    
                    if empty_sequence_count >= max_empty_sequences:
                        print(f"[DEBUG] Слишком много пустых последовательностей подряд ({empty_sequence_count})")
                        if len(X_batch) > 0:
                            print("[DEBUG] Возвращаем неполный батч")
                            break
                        else:
                            print("[DEBUG] Не удалось собрать батч")
                            return None, None
                    continue
                
                if positive_count == last_positive_count and negative_count == last_negative_count:
                    no_progress_count += 1
                    if no_progress_count >= 20:  # Увеличиваем количество попыток без прогресса
                        print("[DEBUG] Нет прогресса в сборе батча")
                        break
                else:
                    no_progress_count = 0
                last_positive_count = positive_count
                last_negative_count = negative_count
                
            except Exception as e:
                print(f"[ERROR] Ошибка при получении последовательности: {str(e)}")
                import traceback
                traceback.print_exc()
                attempts += 1
                empty_sequence_count += 1
                if empty_sequence_count >= max_empty_sequences:
                    print("[DEBUG] Слишком много ошибок подряд")
                    if len(X_batch) > 0:
                        break
                    else:
                        return None, None
                continue
        
        if len(X_batch) == 0:
            print("[DEBUG] Не удалось собрать ни одной последовательности")
            return None, None
        
        try:
            # Преобразуем списки в массивы с явным указанием формы
            X_batch_array = np.stack(X_batch)
            y_batch_array = np.stack(y_batch)
            
            # Сохраняем статистику батча
            self._save_batch_statistics(
                X_batch=X_batch_array,
                y_batch=y_batch_array,
                batch_number=self.current_batch,
                positive_count=positive_count,
                negative_count=negative_count,
                video_path=os.path.basename(self.video_paths[self.current_video_index]) if self.current_video_index < len(self.video_paths) else "unknown"
            )
            
            self.current_batch += 1
            print(f"\n[DEBUG] Батч {self.current_batch}/{self.total_batches} собран успешно")
            print(f"[DEBUG] Итоговая статистика:")
            print(f"  - Всего последовательностей: {len(X_batch)}")
            print(f"  - Положительных: {positive_count}")
            print(f"  - Отрицательных: {negative_count}")
            print(f"  - Попыток: {attempts}")
            return X_batch_array, y_batch_array
            
        except Exception as e:
            print(f"[ERROR] Ошибка при формировании батча: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None

    def data_generator(self, force_positive: bool = True, is_validation: bool = False):
        max_empty_batches = 20  # Увеличьте с 10 до 20
        empty_batch_count = 0
        
        while True:
            try:
                X_batch, y_batch = self.get_batch(
                    batch_size=self.batch_size,
                    sequence_length=self.sequence_length,
                    target_size=self.target_size,
                    one_hot=True,
                    max_sequences_per_video=self.max_sequences_per_video,
                    force_positive=force_positive,
                    is_validation=is_validation
                )
                
                if X_batch is not None and y_batch is not None:
                    empty_batch_count = 0
                    yield X_batch, y_batch
                else:
                    empty_batch_count += 1
                    if empty_batch_count >= max_empty_batches:
                        logger.error("Слишком много пустых батчей подряд")
                        break
            except Exception as e:
                logger.error(f"Ошибка в генераторе данных: {str(e)}")
                break

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
            logger.debug("[DEBUG] Начало расчета общего количества батчей")
            
            # Рассчитываем количество батчей на основе количества видео и ограничений
            total_sequences = 0
            for video_path in self.video_paths:
                info = self._get_video_info(video_path)
                if info.exists:
                    # Берем минимум из:
                    # 1. Максимального количества последовательностей на видео
                    # 2. Количества возможных последовательностей в видео
                    sequences_per_video = min(
                        self.max_sequences_per_video,
                        max(0, info.total_frames - self.sequence_length + 1)
                    )
                    total_sequences += sequences_per_video
            
            # Количество батчей = общее количество последовательностей / размер батча
            self.total_batches = total_sequences // self.batch_size
            
            logger.debug(f"[DEBUG] Рассчитано батчей: {self.total_batches}")
            logger.debug(f"[DEBUG] Общее количество последовательностей: {total_sequences}")
            
        except Exception as e:
            logger.debug(f"[ERROR] Ошибка при расчете количества батчей: {str(e)}")
            logger.debug("[DEBUG] Stack trace:", flush=True)
            import traceback
            traceback.print_exc()
            self.total_batches = 0
    
    def get_video_info(self, video_path: str) -> dict:
        """
        Получение информации о видео в виде словаря
        
        Args:
            video_path: путь к видео файлу
            
        Returns:
            dict: словарь с информацией о видео (total_frames, fps, width, height)
            
        Raises:
            CorruptedVideoError: если видео повреждено или имеет некорректные параметры
        """
        info = self._get_video_info(video_path)
        return {
            'total_frames': info.total_frames,
            'fps': info.fps,
            'width': info.width,
            'height': info.height,
            'file_size': info.file_size,
            'exists': info.exists
        }

    def _load_frame(self, video_path: str, frame_idx: int, target_size: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Загрузка кадра из видео с обработкой ошибок
        
        Args:
            video_path: путь к видео файлу
            frame_idx: индекс кадра
            target_size: целевой размер кадра (height, width)
            
        Returns:
            Optional[np.ndarray]: кадр или None в случае ошибки
        """
        try:
            # Проверяем размер кэша
            if len(self.video_cache) >= self.cache_cleanup_threshold:
                logger.debug("[DEBUG] Очистка кэша видео")
                self.clear_cache()
            
            # Проверяем кэш
            if video_path in self.video_cache:
                cap = self.video_cache[video_path]
                if not cap.isOpened():
                    logger.warning(f"[WARNING] Видео в кэше повреждено: {video_path}")
                    del self.video_cache[video_path]
                    cap = None
            else:
                cap = None
            
            # Открываем видео если нужно
            if cap is None:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    logger.error(f"[ERROR] Не удалось открыть видео: {video_path}")
                    return None
                self.video_cache[video_path] = cap
            
            # Устанавливаем позицию
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            
            # Читаем кадр
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"[WARNING] Не удалось прочитать кадр {frame_idx} из {video_path}")
                return None
            
            # Проверяем размер кадра
            if frame.size == 0:
                logger.warning(f"[WARNING] Пустой кадр {frame_idx} из {video_path}")
                return None
            
            # Изменяем размер
            try:
                frame = cv2.resize(frame, target_size)
            except Exception as e:
                logger.warning(f"[WARNING] Ошибка при изменении размера кадра {frame_idx}: {str(e)}")
                return None
            
            # Нормализуем
            try:
                frame = frame.astype(np.float32) / 255.0
            except Exception as e:
                logger.warning(f"[WARNING] Ошибка при нормализации кадра {frame_idx}: {str(e)}")
                return None
            
            return frame
            
        except Exception as e:
            logger.error(f"[ERROR] Ошибка при загрузке кадра: {str(e)}")
            if video_path in self.video_cache:
                del self.video_cache[video_path]
            return None 

    def _get_annotation_path(self, video_path: str) -> str:
        """
        Получает путь к файлу аннотаций для видео
        
        Args:
            video_path: путь к видео файлу
            
        Returns:
            str: путь к файлу аннотаций
        """
        try:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            
            # Определяем, является ли видео тренировочным или валидационным
            if 'train' in video_path:
                annotation_path = os.path.join(Config.TRAIN_ANNOTATION_PATH, f"{video_name}.json")
            elif 'valid' in video_path:
                annotation_path = os.path.join(Config.VALID_ANNOTATION_PATH, f"{video_name}.json")
            else:
                raise ValueError(f"Неизвестный тип данных в пути: {video_path}")
            
            return annotation_path
            
        except Exception as e:
            logger.error(f"Ошибка при получении пути к аннотациям: {str(e)}")
            raise 
