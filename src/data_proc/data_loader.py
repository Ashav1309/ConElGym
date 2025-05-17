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
        self.open_videos: Set[str] = set()  # Множество открытых видео
        
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
        try:
            # Закрываем все открытые видео
            for cap in self.video_cache.values():
                if cap is not None:
                    cap.release()
            
            # Очищаем все кэши
            self.video_cache.clear()
            self.used_frames_cache.clear()
            self.positive_indices_cache.clear()
            self.file_info_cache.clear()
            self.processed_videos.clear()
            self.open_videos.clear()
            
            # Принудительная очистка памяти
            gc.collect()
            
            print("[DEBUG] Все кэши очищены")
        except Exception as e:
            print(f"[ERROR] Ошибка при очистке кэшей: {str(e)}")

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
        """Загрузка видео из директории с поддержкой лимита MAX_VIDEOS на одну сессию"""
        try:
            video_files = []
            for ext in ['.mp4', '.avi', '.mov']:
                video_files.extend(list(self.data_path.glob(f'*{ext}')))
            
            if not video_files:
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
            print(f"[DEBUG] Всего найдено {self.total_videos} видео")
            print(f"[DEBUG] Загружено {self.video_count} видео из {self.total_videos} для текущей сессии")
        except Exception as e:
            print(f"[ERROR] Ошибка при загрузке видео: {str(e)}")
            raise

    def _load_video_chunk(self):
        """Загружает очередную порцию видео согласно MAX_VIDEOS"""
        start_idx = self.current_video_index
        end_idx = min(start_idx + self.max_videos, self.total_videos)
        self.video_paths = self.all_video_paths[start_idx:end_idx]
        self.video_count = len(self.video_paths)
        self.labels = [None] * self.video_count
        print(f"[DEBUG] Загружена порция видео: {self.video_count} видео (индексы {start_idx}:{end_idx})")

    def _load_annotations(self, video_path: str) -> np.ndarray:
        """
        Загрузка аннотаций для видео
        
        Args:
            video_path: путь к видео файлу
            
        Returns:
            np.ndarray: массив меток для каждого кадра (three-hot encoding)
            [1,0,0] - фон
            [0,1,0] - действие
            [0,0,1] - переход (начало/конец)
            
        Raises:
            InvalidAnnotationError: если формат аннотаций некорректен
        """
        try:
            # Получаем путь к файлу аннотаций
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            annotation_path = os.path.join(os.path.dirname(video_path), 'annotations', f'{base_name}.json')
            
            print(f"[DEBUG] Загрузка аннотаций из: {annotation_path}")
            
            if not os.path.exists(annotation_path):
                logger.warning(f"Аннотации не найдены для {video_path}")
                return np.zeros((self._get_video_info(video_path).total_frames, 3))
            
            with open(annotation_path, 'r') as f:
                annotations = json.load(f)
            
            print(f"[DEBUG] Содержимое файла аннотаций: {json.dumps(annotations, indent=2)}")
            
            # Проверяем формат аннотаций
            if not isinstance(annotations, dict) or 'annotations' not in annotations:
                raise InvalidAnnotationError(f"Некорректный формат аннотаций в {annotation_path}")
            
            # Создаем массив меток
            total_frames = self._get_video_info(video_path).total_frames
            labels = np.zeros((total_frames, 3))  # 3 класса: фон, действие, переход
            
            # Заполняем метки
            print(f"[DEBUG] Обработка аннотаций:")
            for i, annotation in enumerate(annotations['annotations']):
                if not isinstance(annotation, dict) or 'start_frame' not in annotation or 'end_frame' not in annotation:
                    raise InvalidAnnotationError(f"Некорректный формат аннотации {i} в {annotation_path}")
                
                start_frame = annotation['start_frame']
                end_frame = annotation['end_frame']
                
                print(f"[DEBUG] Аннотация {i+1}: кадры {start_frame}-{end_frame}")
                
                if not isinstance(start_frame, int) or not isinstance(end_frame, int):
                    raise InvalidAnnotationError(f"Некорректные индексы кадров в аннотации {i}")
                
                if start_frame < 0 or end_frame >= total_frames:
                    raise InvalidAnnotationError(f"Индексы кадров вне диапазона в аннотации {i}")
                
                # Отмечаем действие
                labels[start_frame:end_frame + 1, 1] = 1  # [0,1,0] - действие
                
                # Отмечаем переходы
                labels[start_frame, 2] = 1  # [0,0,1] - начало
                labels[end_frame, 2] = 1    # [0,0,1] - конец
            
            # Статистика
            background_frames = np.sum(labels[:, 0] == 1)
            action_frames = np.sum(labels[:, 1] == 1)
            transition_frames = np.sum(labels[:, 2] == 1)
            
            print(f"[DEBUG] Статистика аннотаций:")
            print(f"  - Всего кадров: {total_frames}")
            print(f"  - Фоновых кадров: {background_frames}")
            print(f"  - Кадров действия: {action_frames}")
            print(f"  - Кадров перехода: {transition_frames}")
            
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

    def _get_random_video(self) -> Optional[str]:
        """
        Получение случайного видео из текущей порции. Если все видео обработаны — загружается новая порция.
        """
        try:
            if not self.video_paths:
                print("[DEBUG] Список видео пуст, загружаем новую порцию")
                self.current_video_index += self.max_videos
                if self.current_video_index >= self.total_videos:
                    self.current_video_index = 0  # Циклически
                self._load_video_chunk()
                if not self.video_paths:
                    print("[ERROR] Нет доступных видео для загрузки")
                    return None
            # Получаем список необработанных видео
            unprocessed_videos = [v for v in self.video_paths if v not in self.processed_videos]
            if not unprocessed_videos:
                print("[DEBUG] Все видео из текущей порции обработаны, загружаем новую порцию")
                self.current_video_index += self.max_videos
                if self.current_video_index >= self.total_videos:
                    self.current_video_index = 0
                self._load_video_chunk()
                self.processed_videos.clear()
                unprocessed_videos = self.video_paths
                if not unprocessed_videos:
                    print("[ERROR] Нет доступных видео после обновления порции")
                    return None
            video_path = np.random.choice(unprocessed_videos)
            print(f"[DEBUG] Выбрано видео: {video_path}")
            if not os.path.exists(video_path):
                print(f"[ERROR] Видео не найдено: {video_path}")
                self.processed_videos.add(video_path)
                return None
            if video_path in self.used_frames_cache:
                video_info = self._get_video_info(video_path)
                if video_info and len(self.used_frames_cache[video_path]) >= video_info.total_frames - self.sequence_length:
                    print(f"[DEBUG] Все кадры видео {video_path} уже использованы")
                    self.processed_videos.add(video_path)
                    return None
            return video_path
        except Exception as e:
            print(f"[ERROR] Ошибка при выборе случайного видео: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _get_sequence(self, sequence_length, target_size, force_positive=False, is_validation=False):
        try:
            # Получаем случайное видео
            video_path = self._get_random_video()
            if video_path is None:
                print("[ERROR] Не удалось получить видео")
                return None, None
            
            # Загружаем аннотации
            frame_labels = self._load_annotations(video_path)
            if frame_labels is None:
                print(f"[ERROR] Не удалось загрузить аннотации для {video_path}")
                return None, None
            
            # Получаем информацию о видео
            info = self._get_video_info(video_path)
            if not info.exists:
                print(f"[ERROR] Видео не существует: {video_path}")
                return None, None
            
            # Определяем начальный индекс
            if force_positive:
                positive_indices = np.where(frame_labels == 1)[0]
                if len(positive_indices) == 0:
                    print(f"[DEBUG] Нет положительных кадров в видео {video_path}")
                    return None, None
                start_idx = np.random.choice(positive_indices)
            else:
                start_idx = np.random.randint(0, len(frame_labels) - sequence_length + 1)
            
            # Проверяем, что последовательность помещается в видео
            if start_idx + sequence_length > len(frame_labels):
                if is_validation:
                    print(f"[ERROR] Неполная последовательность в валидации")
                    return None, None
                else:
                    print(f"[DEBUG] Последовательность не помещается в видео")
                    return None, None
                
            # Открываем видео
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"[ERROR] Не удалось открыть видео {video_path}")
                return None, None
            
            try:
                # Устанавливаем позицию
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
                
                # Собираем последовательность
                sequence = []
                labels = []
                has_positive = False
                has_background = False
                
                for i in range(sequence_length):
                    ret, frame = cap.read()
                    if not ret:
                        if is_validation:
                            # Для валидации требуем полную последовательность
                            print(f"[ERROR] Неполная последовательность в валидации")
                            return None, None
                        else:
                            # Для обучения проверяем содержимое
                            if len(sequence) > 0:
                                if has_positive:
                                    # Если есть положительные кадры, дополняем
                                    print(f"[DEBUG] Последовательность неполная, но содержит положительные кадры")
                                    while len(sequence) < sequence_length:
                                        sequence.append(sequence[-1])
                                        labels.append(labels[-1])
                                    break
                                elif has_background and len(sequence) >= sequence_length // 2:
                                    # Если есть достаточно фоновых кадров, дополняем
                                    print(f"[DEBUG] Последовательность неполная, но содержит достаточно фоновых кадров")
                                    while len(sequence) < sequence_length:
                                        sequence.append(sequence[-1])
                                        labels.append(labels[-1])
                                    break
                                else:
                                    # Если недостаточно кадров, отбрасываем
                                    print(f"[DEBUG] Последовательность слишком короткая")
                                    return None, None
                            else:
                                return None, None
                    
                    # Обработка кадра
                    frame = cv2.resize(frame, target_size)
                    frame = frame.astype(np.float32) / 255.0
                    
                    sequence.append(frame)
                    current_label = frame_labels[start_idx + i]
                    labels.append(current_label)
                    
                    # Отслеживаем типы кадров
                    if np.any(current_label == 1.0):
                        has_positive = True
                    if np.all(current_label == 0.0):
                        has_background = True
                
                # Преобразуем в numpy массивы
                sequence = np.array(sequence)
                labels = np.array(labels)
                
                print(f"[DEBUG] Статистика последовательности:")
                print(f"  - Содержит положительные кадры: {has_positive}")
                print(f"  - Содержит фоновые кадры: {has_background}")
                print(f"  - Длина последовательности: {len(sequence)}")
                
                return sequence, labels
                
            finally:
                cap.release()
            
        except Exception as e:
            print(f"[ERROR] Ошибка при получении последовательности: {str(e)}")
            return None, None

    def _save_batch_statistics(self, batch_number: int, positive_count: int, negative_count: int, video_path: str):
        """
        Сохранение статистики по батчам в файл
        
        Args:
            batch_number: номер батча
            positive_count: количество положительных примеров
            negative_count: количество отрицательных примеров
            video_path: путь к видео
        """
        try:
            stats_file = "batch_statistics.txt"
            file_exists = os.path.exists(stats_file)
            
            with open(stats_file, 'a', encoding='utf-8') as f:
                if not file_exists:
                    f.write("Номер батча | Положительные | Отрицательные | Соотношение | Видео\n")
                    f.write("-" * 80 + "\n")
                
                ratio = positive_count / negative_count if negative_count > 0 else float('inf')
                f.write(f"{batch_number:10d} | {positive_count:12d} | {negative_count:12d} | {ratio:10.2f} | {os.path.basename(video_path)}\n")
                
        except Exception as e:
            print(f"[ERROR] Ошибка при сохранении статистики: {str(e)}")

    def get_batch(self, batch_size, sequence_length, target_size, one_hot=True, max_sequences_per_video=None, force_positive=False, is_validation=False):
        """
        Получение батча данных
        
        Args:
            batch_size: размер батча
            sequence_length: длина последовательности
            target_size: размер кадра (ширина, высота)
            one_hot: использовать one-hot кодирование для меток
            max_sequences_per_video: максимальное количество последовательностей из одного видео
            force_positive: принудительно использовать положительные примеры
            is_validation: флаг валидации
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: батч данных и меток
        """
        X_batch = []
        y_batch = []
        attempts = 0
        max_attempts = 10
        
        positive_count = 0
        negative_count = 0
        
        while len(X_batch) < batch_size and attempts < max_attempts:
            try:
                sequence, label = self._get_sequence(
                    sequence_length=sequence_length,
                    target_size=target_size,
                    force_positive=force_positive,
                    is_validation=is_validation
                )
                
                if sequence is None or label is None:
                    attempts += 1
                    continue
                
                # Проверяем баланс классов только для тренировочных данных
                if not is_validation:
                    current_positive_ratio = (positive_count + 1) / (len(X_batch) + 1)
                    if current_positive_ratio > 0.7:  # Допускаем до 70% положительных примеров
                        attempts += 1
                        continue
                
                X_batch.append(sequence)
                y_batch.append(label)
                
                if label == 1:
                    positive_count += 1
                else:
                    negative_count += 1
                
            except Exception as e:
                print(f"[ERROR] Ошибка при получении последовательности: {str(e)}")
                attempts += 1
                continue
        
        # Проверяем, удалось ли собрать полный батч
        if len(X_batch) < batch_size:
            print(f"[WARNING] Не удалось собрать полный батч после {attempts} попыток")
            print(f"[WARNING] Собрано последовательностей: {len(X_batch)}/{batch_size}")
            return None, None  # Всегда возвращаем None для неполного батча
        
        # Преобразуем в numpy массивы
        X_batch = np.array(X_batch)
        y_batch = np.array(y_batch)
        
        # Применяем one-hot кодирование если нужно
        if one_hot:
            y_batch = tf.keras.utils.to_categorical(y_batch, num_classes=2)
        
        # Выводим статистику батча
        print(f"[DEBUG] Размер батча: {len(X_batch)}")
        print(f"[DEBUG] Положительных примеров: {positive_count}")
        print(f"[DEBUG] Отрицательных примеров: {negative_count}")
        print(f"[DEBUG] Попыток собрать батч: {attempts}")
        
        return X_batch, y_batch

    def data_generator(self, force_positive: bool = True, is_validation: bool = False) -> Generator[Tuple[tf.Tensor, tf.Tensor], None, None]:
        """
        Генератор данных для обучения
        
        Args:
            force_positive: принудительно использовать положительные примеры
            is_validation: флаг валидации
            
        Yields:
            Tuple[tf.Tensor, tf.Tensor]: батч данных и меток
        """
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
                
                if X_batch is None or y_batch is None:
                    print("[WARNING] Пропускаем неполный батч")
                    continue  # Пропускаем неполный батч и пробуем собрать новый
                    
                yield X_batch, y_batch
                
            except Exception as e:
                print(f"[ERROR] Ошибка в генераторе данных: {str(e)}")
                continue
    
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
            for batch in self.data_generator():
                batch_count += 1
                print(f"[DEBUG] Получен батч {batch_count}")
                if batch_count >= 10:  # Ограничиваем количество батчей для отладки
                    print("[DEBUG] Достигнуто максимальное количество батчей для отладки")
                    break
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

    def _load_frame(self, video_path: str, frame_idx: int, target_size: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Загрузка кадра из видео
        
        Args:
            video_path: путь к видео файлу
            frame_idx: индекс кадра
            target_size: целевой размер кадра (height, width)
            
        Returns:
            Optional[np.ndarray]: кадр или None в случае ошибки
        """
        try:
            # Проверяем кэш
            if video_path in self.video_cache:
                cap = self.video_cache[video_path]
            else:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"[ERROR] Не удалось открыть видео: {video_path}")
                    return None
                self.video_cache[video_path] = cap
            
            # Устанавливаем позицию
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            
            # Читаем кадр
            ret, frame = cap.read()
            if not ret:
                print(f"[ERROR] Не удалось прочитать кадр {frame_idx}")
                return None
            
            # Изменяем размер
            frame = cv2.resize(frame, target_size)
            
            # Нормализуем
            frame = frame.astype(np.float32) / 255.0
            
            return frame
            
        except Exception as e:
            print(f"[ERROR] Ошибка при загрузке кадра: {str(e)}")
            print("[DEBUG] Stack trace:", flush=True)
            import traceback
            traceback.print_exc()
            return None 
