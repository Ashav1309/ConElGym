import os
import cv2
import json
from src.config import Config
from src.data_proc.data_loader import VideoDataLoader
import numpy as np
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

def validate_dataset() -> Tuple[List[str], List[str]]:
    """
    Проверка наличия и качества данных перед началом обучения
    Returns:
        Tuple[List[str], List[str]]: списки путей к видео для обучения и валидации
    """
    logger.info("Начало валидации датасета...")
    
    # Проверка наличия файлов
    train_videos = [f for f in os.listdir(Config.TRAIN_DATA_PATH) if f.endswith('.mp4')]
    val_videos = [f for f in os.listdir(Config.VALID_DATA_PATH) if f.endswith('.mp4')]
    
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
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Не удалось открыть видео: {video_path}")
            
            # Проверка размера
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if width < Config.MIN_VIDEO_WIDTH or height < Config.MIN_VIDEO_HEIGHT:
                raise ValueError(f"Видео {video_path} слишком маленького размера: {width}x{height}")
            
            # Проверка FPS
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps < Config.MIN_FPS:
                raise ValueError(f"Видео {video_path} имеет слишком низкий FPS: {fps}")
            
            # Проверка количества кадров
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames < Config.MIN_FRAMES_PER_VIDEO:
                raise ValueError(f"Видео {video_path} слишком короткое: {total_frames} кадров")
            
            cap.release()
            
        except Exception as e:
            logger.error(f"Ошибка при проверке видео {video_path}: {str(e)}")
            raise

def calculate_positive_examples() -> Tuple[int, int]:
    """
    Подсчет положительных примеров в датасете
    Returns:
        Tuple[int, int]: количество положительных примеров и общее количество примеров
    """
    logger.info("Подсчет положительных примеров...")
    
    total_count = 0
    positive_count = 0
    
    # Проверяем тренировочные данные
    train_loader = VideoDataLoader(Config.TRAIN_DATA_PATH)
    for video_path in train_loader.video_paths:
        annotation_path = os.path.join(Config.TRAIN_ANNOTATION_PATH, 
                                     os.path.splitext(os.path.basename(video_path))[0] + '.json')
        
        if os.path.exists(annotation_path):
            with open(annotation_path, 'r') as f:
                ann_data = json.load(f)
                for annotation in ann_data['annotations']:
                    total_count += 1
                    if annotation.get('is_positive', False):
                        positive_count += 1
    
    # Проверяем валидационные данные
    val_loader = VideoDataLoader(Config.VALID_DATA_PATH)
    for video_path in val_loader.video_paths:
        annotation_path = os.path.join(Config.VALID_ANNOTATION_PATH, 
                                     os.path.splitext(os.path.basename(video_path))[0] + '.json')
        
        if os.path.exists(annotation_path):
            with open(annotation_path, 'r') as f:
                ann_data = json.load(f)
                for annotation in ann_data['annotations']:
                    total_count += 1
                    if annotation.get('is_positive', False):
                        positive_count += 1
    
    logger.info(f"Найдено {positive_count} положительных примеров из {total_count} всего")
    return positive_count, total_count

def check_class_balance(positive_count: int, total_count: int) -> None:
    """
    Проверка баланса классов
    Args:
        positive_count: количество положительных примеров
        total_count: общее количество примеров
    """
    logger.info("Проверка баланса классов...")
    
    positive_ratio = positive_count / total_count if total_count > 0 else 0
    
    if positive_ratio < Config.MIN_POSITIVE_RATIO:
        raise ValueError(f"Слишком мало положительных примеров: {positive_ratio:.2%} < {Config.MIN_POSITIVE_RATIO:.2%}")
    
    if positive_ratio > Config.MAX_POSITIVE_RATIO:
        raise ValueError(f"Слишком много положительных примеров: {positive_ratio:.2%} > {Config.MAX_POSITIVE_RATIO:.2%}")
    
    logger.info(f"Баланс классов в норме: {positive_ratio:.2%} положительных примеров")

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
    if positive_count < Config.MIN_POSITIVE_EXAMPLES:
        raise ValueError(f"Недостаточно положительных примеров: {positive_count} < {Config.MIN_POSITIVE_EXAMPLES}")
    
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