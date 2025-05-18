import os
import cv2
import json
from src.config import Config
from src.data_proc.data_loader import VideoDataLoader
import numpy as np
from typing import List, Tuple, Dict
import logging
import tensorflow as tf

logger = logging.getLogger(__name__)

def validate_dataset() -> Tuple[List[str], List[str]]:
    """
    Проверка наличия и качества данных перед началом обучения
    Returns:
        Tuple[List[str], List[str]]: списки полных путей к видео для обучения и валидации
    """
    logger.info("Начало валидации датасета...")
    
    # Проверка наличия файлов
    train_videos = [os.path.join(Config.TRAIN_DATA_PATH, f) 
                   for f in os.listdir(Config.TRAIN_DATA_PATH) 
                   if f.endswith('.mp4')]
    val_videos = [os.path.join(Config.VALID_DATA_PATH, f) 
                 for f in os.listdir(Config.VALID_DATA_PATH) 
                 if f.endswith('.mp4')]
    
    if len(train_videos) < Config.MIN_TRAIN_VIDEOS:
        raise ValueError(f"Недостаточно видео для обучения: {len(train_videos)} < {Config.MIN_TRAIN_VIDEOS}")
    
    if len(val_videos) < Config.MIN_VAL_VIDEOS:
        raise ValueError(f"Недостаточно видео для валидации: {len(val_videos)} < {Config.MIN_VAL_VIDEOS}")
    
    # Проверка аннотаций
    train_annotations = [f for f in os.listdir(Config.TRAIN_ANNOTATION_PATH) if f.endswith('.json')]
    val_annotations = [f for f in os.listdir(Config.VALID_ANNOTATION_PATH) if f.endswith('.json')]
    
    if len(train_annotations) < len(train_videos):
        raise ValueError(f"Не все видео имеют аннотации: {len(train_annotations)} < {len(train_videos)}")
    
    if len(val_annotations) < len(val_videos):
        raise ValueError(f"Не все видео имеют аннотации: {len(val_annotations)} < {len(val_videos)}")
    
    logger.info(f"Найдено {len(train_videos)} видео для обучения и {len(val_videos)} для валидации")
    return train_videos, val_videos

def check_data_quality(video_paths: List[str]) -> None:
    """
    Проверка качества видео файлов
    Args:
        video_paths: список путей к видео файлам
    """
    logger.info("Проверка качества видео...")
    
    for video_path in video_paths:
        try:
            # Проверяем существование файла
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Файл не найден: {video_path}")
            
            # Проверяем права доступа
            if not os.access(video_path, os.R_OK):
                raise PermissionError(f"Нет прав на чтение файла: {video_path}")
            
            # Проверяем размер файла
            file_size = os.path.getsize(video_path)
            logger.info(f"Размер файла {video_path}: {file_size / (1024*1024):.2f} MB")
            
            if file_size == 0:
                raise ValueError(f"Файл пустой: {video_path}")
            
            # Пробуем открыть видео
            logger.info(f"Попытка открыть видео: {video_path}")
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                # Пробуем альтернативный способ открытия
                logger.warning(f"Первый способ открытия не удался, пробуем альтернативный для {video_path}")
                cap = cv2.VideoCapture(str(video_path))
                
                if not cap.isOpened():
                    raise ValueError(f"Не удалось открыть видео: {video_path}")
            
            # Проверка размера
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info(f"Размер видео {video_path}: {width}x{height}")
            
            if width < Config.MIN_VIDEO_WIDTH or height < Config.MIN_VIDEO_HEIGHT:
                raise ValueError(f"Видео {video_path} слишком маленького размера: {width}x{height}")
            
            # Проверка FPS
            fps = cap.get(cv2.CAP_PROP_FPS)
            logger.info(f"FPS видео {video_path}: {fps}")
            
            if fps < Config.MIN_FPS:
                raise ValueError(f"Видео {video_path} имеет слишком низкий FPS: {fps}")
            
            # Проверка количества кадров
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            logger.info(f"Количество кадров в видео {video_path}: {total_frames}")
            
            if total_frames < Config.MIN_FRAMES_PER_VIDEO:
                raise ValueError(f"Видео {video_path} слишком короткое: {total_frames} кадров")
            
            # Пробуем прочитать первый кадр
            ret, frame = cap.read()
            if not ret:
                raise ValueError(f"Не удалось прочитать первый кадр из видео: {video_path}")
            
            logger.info(f"Видео {video_path} успешно проверено")
            cap.release()
            
        except Exception as e:
            logger.error(f"Ошибка при проверке видео {video_path}: {str(e)}")
            logger.error(f"Полный путь к файлу: {os.path.abspath(video_path)}")
            logger.error(f"Тип ошибки: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

def calculate_positive_examples() -> Tuple[int, int]:
    """
    Подсчет положительных примеров (кадров) в датасете
    Returns:
        Tuple[int, int]: количество положительных кадров и общее количество кадров
    """
    logger.info("Подсчет положительных примеров...")

    total_count = 0
    background_count = 0
    action_count = 0

    # Проверяем тренировочные данные
    train_loader = VideoDataLoader(Config.TRAIN_DATA_PATH)
    for video_path in train_loader.video_paths:
        annotation_path = train_loader._get_annotation_path(video_path)

        if os.path.exists(annotation_path):
            with open(annotation_path, 'r') as f:
                ann_data = json.load(f)
            # Получаем общее количество кадров в видео
            cap = cv2.VideoCapture(video_path)
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            total_count += num_frames

            # Создаем массив меток для каждого кадра
            frame_labels = np.zeros((num_frames, 2), dtype=np.float32)  # 2 класса: фон, действие
            frame_labels[:, 0] = 1  # По умолчанию все кадры - фон
            
            for annotation in ann_data['annotations']:
                start_frame = annotation['start_frame']
                end_frame = annotation['end_frame']
                
                # Отмечаем действие
                for frame_idx in range(start_frame, end_frame + 1):
                    if frame_idx < len(frame_labels):
                        frame_labels[frame_idx, 1] = 1  # [0,1] - действие
                        frame_labels[frame_idx, 0] = 0  # Убираем метку фона
            
            # Считаем кадры каждого класса
            background_count += np.sum(frame_labels[:, 0] == 1)
            action_count += np.sum(frame_labels[:, 1] == 1)

    # Аналогично для валидационных данных
    val_loader = VideoDataLoader(Config.VALID_DATA_PATH)
    for video_path in val_loader.video_paths:
        annotation_path = val_loader._get_annotation_path(video_path)

        if os.path.exists(annotation_path):
            with open(annotation_path, 'r') as f:
                ann_data = json.load(f)
            cap = cv2.VideoCapture(video_path)
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            total_count += num_frames

            frame_labels = np.zeros((num_frames, 2), dtype=np.float32)
            frame_labels[:, 0] = 1  # По умолчанию все кадры - фон
            
            for annotation in ann_data['annotations']:
                start_frame = annotation['start_frame']
                end_frame = annotation['end_frame']
                
                # Отмечаем действие
                for frame_idx in range(start_frame, end_frame + 1):
                    if frame_idx < len(frame_labels):
                        frame_labels[frame_idx, 1] = 1  # [0,1] - действие
                        frame_labels[frame_idx, 0] = 0  # Убираем метку фона
            
            # Считаем кадры каждого класса
            background_count += np.sum(frame_labels[:, 0] == 1)
            action_count += np.sum(frame_labels[:, 1] == 1)

    logger.info(f"Статистика датасета:")
    logger.info(f"  - Всего кадров: {total_count}")
    logger.info(f"  - Фоновых кадров: {background_count}")
    logger.info(f"  - Кадров действия: {action_count}")
    logger.info(f"  - Соотношение фона к действию: {background_count/action_count:.2f}:1")

    return action_count, total_count

def check_class_balance(positive_count: int, total_count: int) -> None:
    """
    Проверка баланса классов
    Args:
        positive_count: количество положительных примеров
        total_count: общее количество примеров
    """
    logger.info("Проверка баланса классов...")
    
    min_positive_ratio = Config.MIN_POSITIVE_RATIO  # Теперь из конфига
    if total_count > 0 and positive_count / total_count < min_positive_ratio:
        raise ValueError(f"Слишком мало положительных примеров: {positive_count / total_count:.2%} < {min_positive_ratio * 100:.2f}%")
    
    if total_count > 0 and positive_count / total_count > Config.MAX_POSITIVE_RATIO:
        raise ValueError(f"Слишком много положительных примеров: {positive_count / total_count:.2%} > {Config.MAX_POSITIVE_RATIO:.2%}")
    
    logger.info(f"Баланс классов в норме: {positive_count / total_count:.2%} положительных примеров")

def count_batches(data_loader: VideoDataLoader) -> int:
    """
    Подсчет количества батчей в датасете
    Args:
        data_loader: загрузчик данных
    Returns:
        int: количество батчей
    """
    logger.info("Подсчет количества батчей...")
    
    batch_count = 0
    for _ in data_loader.data_generator():
        batch_count += 1
    
    logger.info(f"Найдено {batch_count} батчей")
    return batch_count

def validate_data_pipeline() -> None:
    """
    Полная валидация пайплайна данных перед началом обучения
    """
    logger.info("Начало полной валидации пайплайна данных...")
    
    # Проверка наличия данных
    train_videos, val_videos = validate_dataset()
    
    # Проверка качества видео
    check_data_quality(train_videos + val_videos)
    
    # Подсчет и проверка положительных примеров
    positive_count, total_count = calculate_positive_examples()
    min_positive_ratio = Config.MIN_POSITIVE_RATIO  # Теперь из конфига
    if total_count > 0 and positive_count / total_count < min_positive_ratio:
        raise ValueError(f"Слишком мало положительных примеров: {positive_count / total_count:.2%} < {min_positive_ratio * 100:.2f}%")
    
    # Проверка баланса классов
    check_class_balance(positive_count, total_count)
    
    # Проверка количества батчей
    train_loader = VideoDataLoader(Config.TRAIN_DATA_PATH)
    val_loader = VideoDataLoader(Config.VALID_DATA_PATH)
    
    train_batches = count_batches(train_loader)
    val_batches = count_batches(val_loader)
    
    if train_batches < Config.MIN_TRAIN_BATCHES:
        raise ValueError(f"Недостаточно батчей для обучения: {train_batches} < {Config.MIN_TRAIN_BATCHES}")
    
    if val_batches < Config.MIN_VAL_BATCHES:
        raise ValueError(f"Недостаточно батчей для валидации: {val_batches} < {Config.MIN_VAL_BATCHES}")
    
    logger.info("Валидация пайплайна данных успешно завершена")

def validate_training_data(train_data, val_data):
    """Проверка качества данных перед подбором гиперпараметров"""
    try:
        print("[DEBUG] Валидация данных...")
        
        # Получаем первый батч из датасетов
        train_batch = next(iter(train_data))
        val_batch = next(iter(val_data))
        
        # Проверка на NaN и Inf
        if tf.reduce_any(tf.math.is_nan(train_batch[0])) or tf.reduce_any(tf.math.is_inf(train_batch[0])):
            raise ValueError("Обнаружены NaN или Inf в обучающих данных")
        if tf.reduce_any(tf.math.is_nan(val_batch[0])) or tf.reduce_any(tf.math.is_inf(val_batch[0])):
            raise ValueError("Обнаружены NaN или Inf в валидационных данных")
            
        # Проверка размерностей
        if train_batch[0].shape[1:] != val_batch[0].shape[1:]:
            raise ValueError("Несоответствие размерностей обучающих и валидационных данных")
            
        # Проверка баланса классов
        train_labels = train_batch[1]
        val_labels = val_batch[1]
        
        train_dist = tf.math.bincount(tf.argmax(train_labels, axis=1))
        val_dist = tf.math.bincount(tf.argmax(val_labels, axis=1))
        
        train_ratio = tf.reduce_min(train_dist) / tf.reduce_max(train_dist)
        val_ratio = tf.reduce_min(val_dist) / tf.reduce_max(val_dist)
        
        if train_ratio < 0.1:
            print(f"[WARNING] Сильный дисбаланс классов в обучающих данных: {train_ratio:.2f}")
        if val_ratio < 0.1:
            print(f"[WARNING] Сильный дисбаланс классов в валидационных данных: {val_ratio:.2f}")
            
        print("[DEBUG] Валидация данных успешно завершена")
        
    except Exception as e:
        print(f"[ERROR] Ошибка валидации данных: {str(e)}")
        raise 