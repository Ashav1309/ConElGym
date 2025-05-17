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
        Загрузка аннотаций для видео с кэшированием
        
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
            # Проверяем кэш
            print(f"[DEBUG] Проверяем кэш")
            if video_path in self.annotations_cache:
                print(f"[DEBUG] Используем кэшированные аннотации для {os.path.basename(video_path)}")
                print(f"[DEBUG] Размер кэшированных аннотаций: {self.annotations_cache[video_path].shape}")
                print(f"[DEBUG] Сумма меток в кэше: {np.sum(self.annotations_cache[video_path])}")
                return self.annotations_cache[video_path]
            else:
                print(f"[DEBUG] Кэш не найден")

            # Получаем путь к файлу аннотаций
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            annotation_path = os.path.join(os.path.dirname(video_path), 'annotations', f'{base_name}.json')
            
            print(f"[DEBUG] Загрузка аннотаций из: {annotation_path}")
            
            if not os.path.exists(annotation_path):
                logger.warning(f"Аннотации не найдены для {video_path}")
                empty_labels = np.zeros((self._get_video_info(video_path).total_frames, 3))
                self.annotations_cache[video_path] = empty_labels
                return empty_labels
            
            with open(annotation_path, 'r') as f:
                annotations = json.load(f)
            
            print(f"[DEBUG] Содержимое файла аннотаций: {json.dumps(annotations, indent=2)}")
            
            # Проверяем формат аннотаций
            if not isinstance(annotations, dict) or 'annotations' not in annotations:
                raise InvalidAnnotationError(f"Некорректный формат аннотаций в {annotation_path}")
            
            # Создаем массив меток
            total_frames = self._get_video_info(video_path).total_frames
            labels = np.zeros((total_frames, 3))  # 3 класса: фон, действие, переход
            
            # Сначала помечаем все кадры как фоновые
            labels[:, 0] = 1
            
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
                labels[start_frame:end_frame + 1, 0] = 0  # Убираем метку фона для кадров действия
                
                # Отмечаем переходы
                labels[start_frame, 2] = 1  # [0,0,1] - начало
                labels[end_frame, 2] = 1    # [0,0,1] - конец
            
            # Считаем статистику
            action_frames = np.sum(labels[:, 1] == 1)  # Количество кадров действия
            transition_frames = np.sum(labels[:, 2] == 1)  # Количество кадров перехода
            background_frames = np.sum(labels[:, 0] == 1)  # Количество фоновых кадров
            overlapping_frames = np.sum((labels[:, 1] == 1) & (labels[:, 2] == 1))  # Кадры, которые являются и действием, и переходом
            
            print(f"[DEBUG] Статистика аннотаций:")
            print(f"  - Всего кадров: {total_frames}")
            print(f"  - Фоновых кадров: {background_frames}")
            print(f"  - Кадров действия: {action_frames}")
            print(f"  - Кадров перехода: {transition_frames}")
            print(f"  - Перекрывающихся кадров: {overlapping_frames}")
            
            print(f"[DEBUG] Проверка меток кадров:")
            print(f"  - Форма labels: {labels.shape}")
            print(f"  - Уникальные значения в labels[:, 0]: {np.unique(labels[:, 0])}")
            print(f"  - Уникальные значения в labels[:, 1]: {np.unique(labels[:, 1])}")
            print(f"  - Уникальные значения в labels[:, 2]: {np.unique(labels[:, 2])}")
            print(f"  - Сумма всех меток: {np.sum(labels)}")

            # Выводим информацию о прогрессе обработки видео
            current_video_number = len(self.processed_video_names)
            remaining_videos = self.total_videos - current_video_number
            print(f"[DEBUG] Прогресс обработки видео: {current_video_number}/{self.total_videos} ({(current_video_number/self.total_videos)*100:.1f}%), осталось: {remaining_videos}")
            
            # Выводим информацию о прогрессе обработки аннотаций
            total_annotations = len(annotations['annotations'])
            processed_annotations = sum(1 for v in self.annotations_cache.values() if np.any(v[:, 1] == 1))  # Считаем только видео с действиями
            print(f"[DEBUG] Прогресс обработки аннотаций: {processed_annotations}/{self.total_videos} ({(processed_annotations/self.total_videos)*100:.1f}%)")
            print(f"[DEBUG] Количество аннотаций в текущем видео: {total_annotations}")
            
            # Сохраняем в кэш
            self.annotations_cache[video_path] = labels
            print(f"[DEBUG] Аннотации для {os.path.basename(video_path)} сохранены в кэш")
            
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
            
            # Используем аннотации из кэша (они уже должны быть там)
            frame_labels = self.annotations_cache[video_path]
            
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
        Получение случайного видео для обработки с проверкой на повторное использование
        
        Returns:
            Optional[str]: путь к видео или None, если все видео обработаны
        """
        try:
            max_attempts = len(self.video_paths)  # Максимальное количество попыток = количество видео в текущей группе
            attempts = 0
            
            while attempts < max_attempts:
                # Проверяем, все ли видео обработаны
                available_videos = []
                for video_path in self.video_paths:
                    video_name = os.path.basename(video_path)
                    if video_name not in self.processed_video_names:
                        available_videos.append(video_path)
                
                if not available_videos:
                    logger.info("Все видео обработаны, переходим к следующей порции")
                    self.current_video_index += self.max_videos
                    if self.current_video_index >= self.total_videos:
                        logger.info("Все видео обработаны, завершаем")
                        return None
                    self._load_video_chunk()
                    self.used_frames_cache.clear()
                    self.used_sequences.clear()
                    self.sequence_counter.clear()
                    attempts = 0  # Сбрасываем счетчик для новой группы
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
                        logger.debug(f"Обработано видео: {len(self.processed_video_names)}/{self.total_videos}")
                        self.processed_video_names.add(os.path.basename(video_path))
                        continue
                
                # Проверяем, не превышен ли лимит последовательностей для этого видео
                if video_path in self.sequence_counter and self.sequence_counter[video_path] >= self.max_sequences_per_video:
                    logger.debug(f"Достигнут лимит последовательностей для видео {os.path.basename(video_path)}")
                    self.processed_video_names.add(os.path.basename(video_path))
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
                self._load_annotations(video_path)
            except Exception as e:
                logger.error(f"Ошибка при загрузке аннотаций: {str(e)}")
                # Создаем пустые аннотации и добавляем в кэш
                total_frames = self._get_video_info(video_path).total_frames
                empty_labels = np.zeros((total_frames, 3))
                self.annotations_cache[video_path] = empty_labels
                self.processed_video_names.add(os.path.basename(video_path))
                return None, None

        # Инициализируем счетчик для видео, если его еще нет
        if video_path not in self.sequence_counter:
            self.sequence_counter[video_path] = 0

        # Получаем последовательность
        try:
            X_seq, y_seq = self.create_sequences(
                video_path=video_path,
                sequence_length=sequence_length,
                target_size=target_size
            )

            if X_seq is not None and y_seq is not None:
                # Создаем уникальный идентификатор последовательности
                sequence_id = f"{os.path.basename(video_path)}_{self.sequence_counter[video_path]}"

                # Проверяем, не использовалась ли эта последовательность ранее
                if sequence_id in self.used_sequences:
                    logger.debug(f"Последовательность {sequence_id} уже использована")
                    return None, None

                # Добавляем последовательность в использованные
                self.used_sequences.add(sequence_id)
                self.sequence_counter[video_path] += 1

                # Если достигнут лимит последовательностей или успешно обработали видео, добавляем в processed_video_names
                if self.sequence_counter[video_path] >= self.max_sequences_per_video:
                    self.processed_video_names.add(os.path.basename(video_path))
                    logger.debug(f"Достигнут лимит последовательностей для видео {os.path.basename(video_path)}, добавляем в общий список обработанных")
                elif os.path.basename(video_path) not in self.processed_video_names:
                    # Добавляем видео в processed_video_names только если его там еще нет
                    self.processed_video_names.add(os.path.basename(video_path))
                    logger.debug(f"Успешно обработано видео {os.path.basename(video_path)}, добавляем в общий список обработанных")

                # Выводим информацию о прогрессе обработки видео
                current_video_number = len(self.processed_video_names)
                remaining_videos = self.total_videos - current_video_number
                logger.debug(f"[DEBUG] Прогресс обработки видео: {current_video_number}/{self.total_videos} ({(current_video_number/self.total_videos)*100:.1f}%), осталось: {remaining_videos}")

                return X_seq, y_seq

        except Exception as e:
            logger.error(f"Ошибка при создании последовательности: {str(e)}")
            # При ошибке добавляем видео в processed_video_names
            self.processed_video_names.add(os.path.basename(video_path))
            return None, None

        return None, None

    def get_batch(self, batch_size, sequence_length, target_size, one_hot=True, max_sequences_per_video=None, force_positive=False, is_validation=False):
        """
        Получение батча данных с пропуском некорректных кадров
        """
        X_batch = []
        y_batch = []
        attempts = 0
        max_attempts = batch_size * 2
        max_empty_sequences = 5
        empty_sequence_count = 0
        
        while len(X_batch) < batch_size and attempts < max_attempts:
            try:
                X_seq, y_seq = self._get_sequence(
                    sequence_length=sequence_length,
                    target_size=target_size,
                    force_positive=force_positive,
                    is_validation=is_validation
                )
                
                if X_seq is not None and y_seq is not None:
                    X_batch.append(X_seq)
                    y_batch.append(y_seq)
                    empty_sequence_count = 0
                else:
                    empty_sequence_count += 1
                    attempts += 1
                    
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
        
        # Сохраняем статистику батча
        self._save_batch_statistics(
            batch_number=self.current_batch,
            positive_count=sum(1 for y in y_batch if np.any(y == 1)),
            negative_count=sum(1 for y in y_batch if not np.any(y == 1)),
            video_path=os.path.basename(self.video_paths[self.current_video_index]) if self.current_video_index < len(self.video_paths) else "unknown"
        )
        
        self.current_batch += 1
        return np.array(X_batch), np.array(y_batch)

    def data_generator(self, force_positive: bool = True, is_validation: bool = False) -> Generator[Tuple[tf.Tensor, tf.Tensor], None, None]:
        """
        Генератор данных для обучения
        
        Args:
            force_positive: принудительно использовать положительные примеры
            is_validation: флаг валидации
            
        Yields:
            Tuple[tf.Tensor, tf.Tensor]: батч данных и меток
        """
        batch_count = 0
        max_empty_batches = 10  # Максимальное количество пустых батчей подряд
        empty_batch_count = 0
        
        while batch_count < self.total_batches:
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
                    empty_batch_count += 1
                    print(f"[WARNING] Пропускаем неполный батч ({empty_batch_count}/{max_empty_batches})")
                    
                    if empty_batch_count >= max_empty_batches:
                        print("[ERROR] Слишком много пустых батчей подряд, завершаем генерацию")
                        break
                        
                    # Проверяем, все ли видео обработаны
                    available_videos = []
                    for video_path in self.video_paths:
                        video_name = os.path.basename(video_path)
                        if video_name not in self.processed_video_names:
                            available_videos.append(video_path)
                    
                    if not available_videos:
                        print("[DEBUG] Все видео обработаны, переходим к следующей порции")
                        self.current_video_index += self.max_videos
                        if self.current_video_index >= self.total_videos:
                            print("[DEBUG] Все видео обработаны, завершаем")
                            break
                        self._load_video_chunk()
                        self.used_frames_cache.clear()
                        self.used_sequences.clear()
                        self.sequence_counter.clear()
                    continue
                
                empty_batch_count = 0  # Сбрасываем счетчик при успешном батче
                batch_count += 1
                yield X_batch, y_batch
                
            except Exception as e:
                print(f"[ERROR] Ошибка в генераторе данных: {str(e)}")
                empty_batch_count += 1
                if empty_batch_count >= max_empty_batches:
                    print("[ERROR] Слишком много ошибок подряд, завершаем генерацию")
                    break
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
            
            # Рассчитываем количество батчей на основе количества видео и кадров
            total_frames = 0
            for video_path in self.video_paths:
                info = self._get_video_info(video_path)
                if info.exists:
                    total_frames += info.total_frames
            
            # Количество последовательностей = (общее количество кадров - длина последовательности + 1)
            total_sequences = total_frames - self.sequence_length + 1
            
            # Количество батчей = количество последовательностей / размер батча
            self.total_batches = total_sequences // self.batch_size
            
            print(f"[DEBUG] Рассчитано батчей: {self.total_batches}")
            print(f"[DEBUG] Общее количество кадров: {total_frames}")
            print(f"[DEBUG] Количество последовательностей: {total_sequences}")
            
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
