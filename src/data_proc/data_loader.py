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
        self.annotations_cache: Dict[str, np.ndarray] = {}  # Кэш для аннотаций
        self.open_videos: Set[str] = set()  # Множество открытых видео
        self.processed_video_names: Set[str] = set()  # Множество имен обработанных видео
        self.sequence_counter: Dict[str, int] = {}    # Счетчик последовательностей для каждого видео
        self.used_sequences: Set[str] = set()         # Множество использованных последовательностей
        
        # Счетчики для отслеживания прогресса
        self.total_processed_videos = 0  # Общее количество обработанных видео
        self.total_processed_frames = 0  # Общее количество обработанных кадров
        self.total_processed_sequences = 0  # Общее количество обработанных последовательностей
        
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
            
            # Очищаем все кэши, кроме annotations_cache
            self.video_cache.clear()
            self.used_frames_cache.clear()
            self.positive_indices_cache.clear()
            self.file_info_cache.clear()
            self.open_videos.clear()
            # Не очищаем processed_video_names и annotations_cache, так как это постоянные хранилища
            
            # Принудительная очистка памяти
            gc.collect()
            
            logger.debug("Все кэши очищены (кроме annotations_cache)")
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
            
            # Проверяем, что указатель кадров работает корректно
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Возвращаемся к началу
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                frame_count += 1
            
            # Проверяем разницу между заявленным и фактическим количеством кадров
            frame_diff = abs(frame_count - info.total_frames)
            frame_diff_percent = (frame_diff / info.total_frames) * 100
            
            if frame_diff > 0:
                if frame_diff_percent <= 6:  # Допускаем погрешность до 6%
                    logger.warning(f"Небольшое несоответствие количества кадров: заявлено {info.total_frames}, фактически {frame_count} (разница: {frame_diff_percent:.1f}%)")
                    # Используем фактическое количество кадров
                    info.total_frames = frame_count
                    self.file_info_cache[video_path] = info
                else:
                    logger.error(f"Значительное несоответствие количества кадров: заявлено {info.total_frames}, фактически {frame_count} (разница: {frame_diff_percent:.1f}%)")
                    raise CorruptedVideoError(f"Значительное несоответствие количества кадров в видео: {video_path}")
            
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
        try:
            # Очищаем кэш перед загрузкой новой порции
            self.clear_cache()
            
            start_idx = self.current_video_index
            end_idx = min(start_idx + self.max_videos, self.total_videos)
            
            # Проверяем существование файлов
            valid_videos = []
            for video_path in self.all_video_paths[start_idx:end_idx]:
                if os.path.exists(video_path):
                    # Проверяем, что видео можно открыть
                    cap = cv2.VideoCapture(video_path)
                    if cap.isOpened():
                        valid_videos.append(video_path)
                        cap.release()
                    else:
                        print(f"[WARNING] Видео повреждено или недоступно: {video_path}")
                else:
                    print(f"[WARNING] Видео не найдено: {video_path}")
            
            self.video_paths = valid_videos
            self.video_count = len(self.video_paths)
            self.labels = [None] * self.video_count
            
            print(f"[DEBUG] Загружена порция видео: {self.video_count} видео (индексы {start_idx}:{end_idx})")
            
            # Если все видео в текущей порции невалидны, пробуем следующую порцию
            if not valid_videos and start_idx < self.total_videos:
                print("[DEBUG] Все видео в текущей порции невалидны, пробуем следующую порцию")
                self.current_video_index = end_idx
                self._load_video_chunk()
            
        except Exception as e:
            print(f"[ERROR] Ошибка при загрузке порции видео: {str(e)}")
            self.video_paths = []
            self.video_count = 0
            self.labels = []

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
                print(f"[DEBUG] Аннотации для {os.path.basename(video_path)} найдены в кэше")
                return self.annotations_cache[video_path]
            
            # Если файл аннотаций не существует, возвращаем пустые метки
            if not os.path.exists(annotation_path):
                print(f"[DEBUG] Файл аннотаций не найден: {annotation_path}")
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
            
            print(f"[DEBUG] Статистика меток для {os.path.basename(video_path)}:")
            print(f"  - Всего кадров: {total_frames}")
            print(f"  - Кадры с фоном: {np.sum(labels[:, 0] == 1)}")
            print(f"  - Кадры с действием: {np.sum(labels[:, 1] == 1)}")
            print(f"  - Сумма всех меток: {np.sum(labels)}")
            
            # Сохраняем в кэш
            self.annotations_cache[video_path] = labels
            print(f"[DEBUG] Аннотации для {os.path.basename(video_path)} сохранены в кэш")
            
            return labels
            
        except Exception as e:
            print(f"[ERROR] Ошибка при загрузке аннотаций для {video_path}: {str(e)}")
            # В случае ошибки возвращаем пустые метки
            empty_labels = np.zeros((total_frames, 2))
            empty_labels[:, 0] = 1  # Все кадры - фон [1, 0]
            return empty_labels

    def create_sequences(self, video_path: str, labels: np.ndarray, sequence_length: int = 12, 
                        max_sequences: int = 200, step: int = 1, force_positive: bool = False) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Создает последовательности кадров из видео.
        
        Args:
            video_path: Путь к видео файлу
            labels: Массив меток для каждого кадра
            sequence_length: Длина последовательности
            max_sequences: Максимальное количество последовательностей
            step: Шаг между последовательностями
            force_positive: Принудительно использовать только последовательности с действиями
            
        Returns:
            Tuple[Optional[np.ndarray], Optional[np.ndarray]]: Последовательность и её метки
        """
        try:
            # Загружаем видео и получаем количество кадров
            cap, total_frames = self.load_video(video_path)
            logger.debug(f"Видео содержит {total_frames} кадров")
            
            if cap is None:
                return None, None
                
            # Проверяем, что у нас достаточно кадров
            if total_frames < sequence_length:
                logger.warning(f"Видео слишком короткое: {total_frames} кадров < {sequence_length}")
                return None, None
            
            # Вычисляем оптимальный шаг для равномерного распределения
            n_possible_sequences = (total_frames - sequence_length) // step + 1
            if n_possible_sequences > max_sequences:
                step = (total_frames - sequence_length) // (max_sequences - 1)
                step = max(1, step)  # Убеждаемся, что шаг не меньше 1
                logger.debug(f"Корректируем шаг до {step} для ограничения количества последовательностей")
            
            # Создаем последовательности
            action_dominant_sequences = []  # Последовательности с преобладанием действия
            action_dominant_labels = []
            background_dominant_sequences = []  # Последовательности с преобладанием фона
            background_dominant_labels = []
            
            for start_idx in range(0, total_frames - sequence_length + 1, step):
                # Проверяем, не выходим ли за пределы видео
                if start_idx + sequence_length > total_frames:
                    logger.warning(f"Пропускаем последовательность: начало {start_idx} + длина {sequence_length} > всего кадров {total_frames}")
                    continue
                
                # Проверяем, есть ли хотя бы один кадр с действием в последовательности
                sequence_label = labels[start_idx:start_idx + sequence_length]
                has_action = np.any(sequence_label[:, 1] == 1)
                
                # Если force_positive=True и нет действия, пропускаем
                if force_positive and not has_action:
                    continue
                
                # Читаем кадры для последовательности
                frames = []
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
                
                for _ in range(sequence_length):
                    ret, frame = cap.read()
                    if not ret:
                        logger.warning(f"Не удалось прочитать кадр на позиции {start_idx + len(frames)}")
                        continue
                    
                    try:
                        frame = cv2.resize(frame, (self.frame_size, self.frame_size))
                        frames.append(frame)
                    except Exception as e:
                        logger.error(f"Ошибка при изменении размера кадра: {str(e)}")
                        continue
                
                if len(frames) == sequence_length:
                    # Преобразуем кадры в numpy массив с правильной формой
                    frames_array = np.array(frames)  # Форма: (sequence_length, height, width, channels)
                    
                    # Вычисляем долю кадров с действием
                    action_ratio = np.mean(sequence_label[:, 1])
                    
                    # Распределяем последовательности по группам
                    if action_ratio > 0.5:  # Больше половины кадров - действие
                        action_dominant_sequences.append(frames_array)
                        action_dominant_labels.append(sequence_label)
                    else:  # Больше половины кадров - фон
                        background_dominant_sequences.append(frames_array)
                        background_dominant_labels.append(sequence_label)
                    
                    # Если набрали достаточно последовательностей, выходим
                    if len(action_dominant_sequences) + len(background_dominant_sequences) >= max_sequences:
                        break
            
            cap.release()
            
            # Балансируем количество последовательностей в каждой группе
            max_per_group = max_sequences // 2
            if len(action_dominant_sequences) > max_per_group:
                indices = np.random.choice(len(action_dominant_sequences), max_per_group, replace=False)
                action_dominant_sequences = [action_dominant_sequences[i] for i in indices]
                action_dominant_labels = [action_dominant_labels[i] for i in indices]
            
            if len(background_dominant_sequences) > max_per_group:
                indices = np.random.choice(len(background_dominant_sequences), max_per_group, replace=False)
                background_dominant_sequences = [background_dominant_sequences[i] for i in indices]
                background_dominant_labels = [background_dominant_labels[i] for i in indices]
            
            # Объединяем последовательности
            all_sequences = action_dominant_sequences + background_dominant_sequences
            all_labels = action_dominant_labels + background_dominant_labels
            
            if not all_sequences:
                logger.warning("Не удалось создать последовательности")
                return None, None
            
            # Перемешиваем
            indices = np.random.permutation(len(all_sequences))
            all_sequences = [all_sequences[i] for i in indices]
            all_labels = [all_labels[i] for i in indices]
            
            # Выбираем случайную последовательность
            idx = np.random.randint(len(all_sequences))
            X = all_sequences[idx]
            y = all_labels[idx]
            
            logger.debug(f"Создана последовательность (действие: {len(action_dominant_sequences)}, фон: {len(background_dominant_sequences)})")
            return X, y
            
        except Exception as e:
            logger.error(f"Ошибка при создании последовательности: {str(e)}")
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
            max_attempts = len(self.video_paths)  # Максимальное количество попыток = количество видео в текущей группе
            attempts = 0
            
            while attempts < max_attempts:
                # Проверяем, все ли видео в текущей порции обработаны
                available_videos = []
                for video_path in self.video_paths:
                    video_name = os.path.basename(video_path)
                    if video_name not in self.processed_video_names:
                        available_videos.append(video_path)
                
                # Если нет доступных видео, переходим к следующей порции
                if not available_videos:
                    logger.info("Все видео в текущей порции обработаны")
                    self.current_video_index += self.max_videos
                    
                    # Проверяем, есть ли еще видео для обработки
                    if self.current_video_index >= self.total_videos:
                        logger.info("Все видео обработаны, завершаем")
                        return None
                        
                    # Загружаем следующую порцию видео
                    self._load_video_chunk()
                    # Очищаем только кэши, связанные с текущей порцией
                    self.used_frames_cache.clear()
                    self.used_sequences.clear()
                    self.sequence_counter.clear()
                    attempts = 0
                    continue
                
                video_path = np.random.choice(available_videos)
                attempts += 1
                
                # Проверяем существование видео
                if not os.path.exists(video_path):
                    logger.error(f"Видео не найдено: {video_path}")
                    self.processed_video_names.add(os.path.basename(video_path))
                    continue
                
                # Проверяем, все ли кадры использованы
                if video_path in self.used_frames_cache:
                    video_info = self._get_video_info(video_path)
                    if video_info and len(self.used_frames_cache[video_path]) >= video_info.total_frames - self.sequence_length:
                        logger.debug(f"Все кадры видео {os.path.basename(video_path)} уже использованы")
                        self.processed_video_names.add(os.path.basename(video_path))
                        # После добавления видео в processed_video_names, проверяем снова available_videos
                        continue
                
                # Проверяем, не превышен ли лимит последовательностей
                if video_path in self.sequence_counter and self.sequence_counter[video_path] >= self.max_sequences_per_video:
                    logger.debug(f"Достигнут лимит последовательностей для видео {os.path.basename(video_path)}")
                    self.processed_video_names.add(os.path.basename(video_path))
                    # После добавления видео в processed_video_names, проверяем снова available_videos
                    continue
                
                # Проверяем, не является ли это последним видео в группе
                if len(available_videos) == 1 and video_path == available_videos[0]:
                    # Если это последнее видео в группе, помечаем его как обработанное
                    self.processed_video_names.add(os.path.basename(video_path))
                    logger.info("Последнее видео в группе обработано, переходим к следующей порции")
                    self.current_video_index += self.max_videos
                    
                    if self.current_video_index >= self.total_videos:
                        logger.info("Все видео обработаны, завершаем")
                        return None
                        
                    self._load_video_chunk()
                    self.used_frames_cache.clear()
                    self.used_sequences.clear()
                    self.sequence_counter.clear()
                    attempts = 0
                    continue
                
                return video_path
            
            logger.warning(f"Превышено максимальное количество попыток ({max_attempts})")
            return None
            
        except Exception as e:
            logger.error(f"Ошибка при выборе случайного видео: {str(e)}")
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
                    return None, None
            except Exception as e:
                logger.error(f"Ошибка при загрузке аннотаций: {str(e)}")
                # Создаем пустые аннотации и добавляем в кэш
                total_frames = self._get_video_info(video_path).total_frames
                empty_labels = np.zeros((total_frames, 2))
                self.annotations_cache[video_path] = empty_labels
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
                force_positive=force_positive
            )

            if X_seq is not None and y_seq is not None:
                # Создаём уникальный идентификатор только для одной последовательности
                seq_id = f"{os.path.basename(video_path)}_{self.sequence_counter[video_path]}"
                if seq_id in self.used_sequences:
                    logger.debug(f"Последовательность {seq_id} уже использована")
                    return None, None

                self.used_sequences.add(seq_id)
                self.sequence_counter[video_path] += 1

                # Проверяем, все ли прочитанные кадры использованы
                video_info = self._get_video_info(video_path)
                if video_info:
                    total_readable_frames = len(self.used_frames_cache.get(video_path, set()))
                    if total_readable_frames >= video_info.total_frames - sequence_length:
                        logger.debug(f"Все прочитанные кадры видео {os.path.basename(video_path)} использованы")
                        self.processed_video_names.add(os.path.basename(video_path))
                        logger.debug(f"Обработано видео: {len(self.processed_video_names)}/{self.total_videos}")

                return X_seq, y_seq

        except Exception as e:
            logger.error(f"Ошибка при создании последовательности: {str(e)}")
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
            if video_path not in self.processed_video_names:
                self.processed_video_names.add(video_path)
                self.total_processed_videos += 1
            
            print(f"[DEBUG] Статистика батча {batch_number}:")
            print(f"  - Размер батча: {X_batch.shape}")
            print(f"  - Положительных примеров: {positive_count}")
            print(f"  - Отрицательных примеров: {negative_count}")
            print(f"  - Видео: {video_path}")
            print(f"[DEBUG] Общая статистика обработки:")
            print(f"  - Обработано видео: {self.total_processed_videos}/{self.total_videos} ({self.total_processed_videos/self.total_videos*100:.1f}%)")
            print(f"  - Обработано кадров: {self.total_processed_frames}")
            print(f"  - Обработано последовательностей: {self.total_processed_sequences}")
        except Exception as e:
            print(f"[WARNING] Ошибка при сохранении статистики батча: {str(e)}")

    def get_batch(self, batch_size, sequence_length, target_size, one_hot=True, max_sequences_per_video=None, force_positive=False, is_validation=False):
        """
        Получение батча данных с исправлением некорректных форм
        """
        X_batch = []
        y_batch = []
        attempts = 0
        max_attempts = batch_size * 5  # Увеличиваем количество попыток
        max_empty_sequences = 10  # Увеличиваем допустимое количество пустых последовательностей
        empty_sequence_count = 0
        
        # Ожидаемая форма последовательности
        expected_shape = (sequence_length, *target_size, 3) if target_size else (sequence_length,)
        
        logger.debug(f"[DEBUG] Начало сбора батча {self.current_batch + 1}/{self.total_batches}")
        logger.debug(f"[DEBUG] Всего батчей: {self.total_batches}")
        logger.debug(f"[DEBUG] Собрано последовательностей: {self.total_processed_sequences}")
        
        while len(X_batch) < batch_size and attempts < max_attempts:
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
                        logger.warning(f"Некорректная форма последовательности: {X_seq.shape}, ожидалось: {expected_shape}")
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
                                logger.error(f"Не удалось исправить форму последовательности: {X_seq.shape} -> {expected_shape}")
                                empty_sequence_count += 1
                                attempts += 1
                                continue
                                
                            logger.debug(f"Форма после исправления: {X_seq.shape}")
                        except Exception as e:
                            logger.error(f"Не удалось исправить форму последовательности: {str(e)}")
                            empty_sequence_count += 1
                            attempts += 1
                            continue
                    
                    X_batch.append(X_seq)
                    y_batch.append(y_seq)
                    empty_sequence_count = 0
                    logger.debug(f"[DEBUG] Добавлена последовательность {len(X_batch)}/{batch_size} в батч {self.current_batch + 1}")
                else:
                    empty_sequence_count += 1
                    attempts += 1
                    logger.debug(f"[DEBUG] Пустая последовательность (попытка {attempts}/{max_attempts})")
                    
                    if empty_sequence_count >= max_empty_sequences:
                        logger.warning(f"Слишком много пустых последовательностей подряд ({empty_sequence_count})")
                        if len(X_batch) > 0:
                            logger.debug("Возвращаем неполный батч")
                            break
                        else:
                            logger.debug("Не удалось собрать батч")
                            return None, None
                    continue
                
            except Exception as e:
                logger.error(f"Ошибка при получении последовательности: {str(e)}")
                attempts += 1
                empty_sequence_count += 1
                if empty_sequence_count >= max_empty_sequences:
                    logger.warning("Слишком много ошибок подряд")
                    if len(X_batch) > 0:
                        break
                    else:
                        return None, None
                continue
        
        if len(X_batch) == 0:
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
                positive_count=sum(1 for y in y_batch if np.any(y[:, 1] == 1)),  # Считаем последовательности с действиями
                negative_count=sum(1 for y in y_batch if not np.any(y[:, 1] == 1)),  # Считаем последовательности без действий
                video_path=os.path.basename(self.video_paths[self.current_video_index]) if self.current_video_index < len(self.video_paths) else "unknown"
            )
            
            self.current_batch += 1
            logger.debug(f"[DEBUG] Батч {self.current_batch}/{self.total_batches} собран успешно")
            logger.debug(f"[DEBUG] Всего собрано последовательностей: {self.total_processed_sequences}")
            return X_batch_array, y_batch_array
            
        except Exception as e:
            logger.error(f"Ошибка при формировании батча: {str(e)}")
            return None, None

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
                yield X_batch, y_batch
            else:
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
            print("[DEBUG] Начало расчета общего количества батчей")
            
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
            
            print(f"[DEBUG] Рассчитано батчей: {self.total_batches}")
            print(f"[DEBUG] Общее количество последовательностей: {total_sequences}")
            
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
                print("[DEBUG] Очистка кэша видео")
                self.clear_cache()
            
            # Проверяем кэш
            if video_path in self.video_cache:
                cap = self.video_cache[video_path]
                if not cap.isOpened():
                    print(f"[WARNING] Видео в кэше повреждено: {video_path}")
                    del self.video_cache[video_path]
                    cap = None
            else:
                cap = None
            
            # Открываем видео если нужно
            if cap is None:
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
                print(f"[WARNING] Не удалось прочитать кадр {frame_idx} из {video_path}")
                return None
            
            # Проверяем размер кадра
            if frame.size == 0:
                print(f"[WARNING] Пустой кадр {frame_idx} из {video_path}")
                return None
            
            # Изменяем размер
            try:
                frame = cv2.resize(frame, target_size)
            except Exception as e:
                print(f"[WARNING] Ошибка при изменении размера кадра {frame_idx}: {str(e)}")
                return None
            
            # Нормализуем
            try:
                frame = frame.astype(np.float32) / 255.0
            except Exception as e:
                print(f"[WARNING] Ошибка при нормализации кадра {frame_idx}: {str(e)}")
                return None
            
            return frame
            
        except Exception as e:
            print(f"[ERROR] Ошибка при загрузке кадра: {str(e)}")
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
