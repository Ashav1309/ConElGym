import os
import json
import cv2
from src.config import Config
from src.data_proc.data_loader import VideoDataLoader
import numpy as np
from tqdm import tqdm

def calculate_dataset_weights():
    """
    Расчет весов классов на всем датасете
    """
    print("[DEBUG] Начинаем расчет весов классов...")
    
    # Инициализируем счетчики
    total_frames = 0
    positive_frames = 0
    
    # Получаем список всех видео
    video_paths = []
    
    # Добавляем видео из тренировочного набора
    train_videos = [f for f in os.listdir(Config.TRAIN_DATA_DIR) if f.endswith('.mp4')]
    video_paths.extend([os.path.join(Config.TRAIN_DATA_DIR, v) for v in train_videos])
    
    # Добавляем видео из валидационного набора
    val_videos = [f for f in os.listdir(Config.VAL_DATA_DIR) if f.endswith('.mp4')]
    video_paths.extend([os.path.join(Config.VAL_DATA_DIR, v) for v in val_videos])
    
    print(f"[DEBUG] Всего найдено видео: {len(video_paths)}")
    
    # Обрабатываем каждое видео
    for video_path in video_paths:
        video_name = os.path.basename(video_path)
        print(f"[DEBUG] Обработка видео: {video_name}")
        
        # Загружаем видео
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[WARNING] Не удалось открыть видео: {video_path}")
            continue
            
        # Загружаем аннотации
        annotation_path = os.path.join(
            os.path.dirname(video_path),
            'annotations',
            f"{os.path.splitext(video_name)[0]}.json"
        )
        
        if not os.path.exists(annotation_path):
            print(f"[WARNING] Аннотации не найдены для видео: {video_name}")
            cap.release()
            continue
            
        with open(annotation_path, 'r') as f:
            annotations = json.load(f)
            
        # Считаем кадры для каждого класса
        frame_count = 0
        video_positive_frames = 0
        
        while True:
            ret, _ = cap.read()
            if not ret:
                break
                
            frame_count += 1
            frame_time = frame_count / cap.get(cv2.CAP_PROP_FPS)
            
            # Проверяем, есть ли аннотация для текущего кадра
            for annotation in annotations:
                if annotation['start_time'] <= frame_time <= annotation['end_time']:
                    video_positive_frames += 1
                    break
        
        # Обновляем общие счетчики
        total_frames += frame_count
        positive_frames += video_positive_frames
        
        print(f"[DEBUG] Видео {video_name}:")
        print(f"  - Всего кадров: {frame_count}")
        print(f"  - Позитивных кадров: {video_positive_frames}")
        
        cap.release()
    
    # Рассчитываем веса
    negative_frames = total_frames - positive_frames
    weights = [1.0, negative_frames / positive_frames if positive_frames > 0 else 1.0]
    
    print("\n[DEBUG] Итоговая статистика:")
    print(f"Всего кадров: {total_frames}")
    print(f"Позитивных кадров: {positive_frames}")
    print(f"Негативных кадров: {negative_frames}")
    print(f"Веса классов: {weights}")
    
    return weights

def save_weights_to_config(weights):
    """
    Сохранение весов в конфигурационный файл
    """
    try:
        # Создаем директорию для конфигурации, если её нет
        os.makedirs(os.path.dirname(Config.CONFIG_PATH), exist_ok=True)
        
        # Базовый конфиг, если файл не существует
        default_config = {
            'MODEL_PARAMS': {
                'v3': {
                    'dropout_rate': 0.3,
                    'lstm_units': 128,
                    'positive_class_weight': None,
                    'base_input_shape': [224, 224, 3]
                },
                'v4': {
                    'dropout_rate': 0.3,
                    'expansion_factor': 4,
                    'se_ratio': 0.25,
                    'positive_class_weight': None,
                    'base_input_shape': [224, 224, 3]
                }
            }
        }
        
        # Читаем текущий конфиг или используем дефолтный
        if os.path.exists(Config.CONFIG_PATH):
            with open(Config.CONFIG_PATH, 'r') as f:
                config = json.load(f)
        else:
            config = default_config
            print(f"[DEBUG] Создаем новый конфигурационный файл: {Config.CONFIG_PATH}")
        
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