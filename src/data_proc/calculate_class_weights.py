import os
import json
import cv2
from src.config import Config
from src.data_proc.data_loader import VideoDataLoader
import numpy as np
from tqdm import tqdm

def calculate_dataset_weights():
    """
    Расчет весов классов для всего датасета
    """
    print("[DEBUG] Начало расчета весов классов для всего датасета...")
    
    # Создаем загрузчик данных без ограничения на количество видео
    train_loader = VideoDataLoader(Config.TRAIN_DATA_PATH, max_videos=None)
    val_loader = VideoDataLoader(Config.VALID_DATA_PATH, max_videos=None)
    
    total_samples = 0
    class_counts = {0: 0, 1: 0}
    
    # Обрабатываем обучающий датасет
    print("[DEBUG] Обработка обучающего датасета...")
    for video_path in tqdm(train_loader.video_paths):
        try:
            annotation_path = train_loader.labels[train_loader.video_paths.index(video_path)]
            if not annotation_path or not os.path.exists(annotation_path):
                print(f"[WARNING] Аннотация не найдена для {video_path}")
                continue
                
            with open(annotation_path, 'r') as f:
                annotations = json.load(f)
            
            # Подсчитываем количество кадров каждого класса
            for element in annotations['annotations']:
                start_frame = element['start_frame']
                end_frame = element['end_frame']
                
                # Все кадры до start_frame - класс 0
                class_counts[0] += start_frame
                
                # Кадры от start_frame до end_frame - класс 1
                class_counts[1] += (end_frame - start_frame)
                
                # Все кадры после end_frame - класс 0
                total_frames = train_loader.get_video_info(video_path)['total_frames']
                class_counts[0] += (total_frames - end_frame)
            
        except Exception as e:
            print(f"[ERROR] Ошибка при обработке видео {video_path}: {str(e)}")
            continue
    
    # Обрабатываем валидационный датасет
    print("[DEBUG] Обработка валидационного датасета...")
    for video_path in tqdm(val_loader.video_paths):
        try:
            annotation_path = val_loader.labels[val_loader.video_paths.index(video_path)]
            if not annotation_path or not os.path.exists(annotation_path):
                print(f"[WARNING] Аннотация не найдена для {video_path}")
                continue
                
            with open(annotation_path, 'r') as f:
                annotations = json.load(f)
            
            # Подсчитываем количество кадров каждого класса
            for element in annotations['annotations']:
                start_frame = element['start_frame']
                end_frame = element['end_frame']
                
                # Все кадры до start_frame - класс 0
                class_counts[0] += start_frame
                
                # Кадры от start_frame до end_frame - класс 1
                class_counts[1] += (end_frame - start_frame)
                
                # Все кадры после end_frame - класс 0
                total_frames = val_loader.get_video_info(video_path)['total_frames']
                class_counts[0] += (total_frames - end_frame)
            
        except Exception as e:
            print(f"[ERROR] Ошибка при обработке видео {video_path}: {str(e)}")
            continue
    
    total_samples = sum(class_counts.values())
    
    if total_samples == 0:
        print("[ERROR] Не удалось подсчитать количество примеров")
        return None
    
    print("[DEBUG] Распределение классов:")
    print(f"  - Всего примеров: {total_samples}")
    print(f"  - Класс 0 (фон): {class_counts[0]}")
    print(f"  - Класс 1 (элемент): {class_counts[1]}")
    
    # Рассчитываем веса
    weights = {
        0: total_samples / (2 * class_counts[0]),
        1: total_samples / (2 * class_counts[1])
    }
    
    print("[DEBUG] Рассчитанные веса:")
    print(f"  - Вес класса 0: {weights[0]:.2f}")
    print(f"  - Вес класса 1: {weights[1]:.2f}")
    
    return weights

def save_weights_to_config(weights):
    """
    Сохранение весов в конфигурационный файл
    """
    try:
        # Создаем директорию для конфигурации, если её нет
        os.makedirs(os.path.dirname(Config.CONFIG_PATH), exist_ok=True)
        
        # Читаем текущий конфиг
        with open(Config.CONFIG_PATH, 'r') as f:
            config = json.load(f)
        
        # Обновляем веса в конфиге
        config['MODEL_PARAMS']['v3']['positive_class_weight'] = weights[1]
        config['MODEL_PARAMS']['v4']['positive_class_weight'] = weights[1]
        
        # Сохраняем обновленный конфиг
        with open(Config.CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"[DEBUG] Веса успешно сохранены в {Config.CONFIG_PATH}")
        
    except Exception as e:
        print(f"[ERROR] Ошибка при сохранении весов в конфиг: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        weights = calculate_dataset_weights()
        if weights is not None:
            save_weights_to_config(weights)
            print("\n[SUCCESS] Веса классов успешно рассчитаны и сохранены в конфигурацию")
        else:
            print("\n[ERROR] Не удалось рассчитать веса классов")
    except Exception as e:
        print(f"\n[ERROR] Ошибка при расчете весов классов: {str(e)}") 