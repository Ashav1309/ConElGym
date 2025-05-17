import os
import json
import cv2
from src.config import Config
from src.data_proc.data_loader import VideoDataLoader
import numpy as np
from tqdm import tqdm

def calculate_dataset_weights():
    """
    Расчет весов классов для датасета
    """
    print("[INFO] Начало расчета весов классов")
    
    # Инициализация счетчиков
    total_frames = 0
    positive_frames = set()
    video_stats = {}
    
    # Получаем пути к видео
    train_video_paths = [os.path.join(Config.TRAIN_DATA_PATH, f) for f in os.listdir(Config.TRAIN_DATA_PATH) 
                        if f.endswith('.mp4')]
    valid_video_paths = [os.path.join(Config.VALID_DATA_PATH, f) for f in os.listdir(Config.VALID_DATA_PATH) 
                        if f.endswith('.mp4')]
    video_paths = train_video_paths + valid_video_paths
    
    # Обрабатываем каждое видео
    for video_path in tqdm(video_paths, desc="Обработка видео"):
        video_name = os.path.basename(video_path)
        video_stats[video_name] = {
            'total_frames': 0,
            'background_frames': 0,
            'action_frames': 0,
            'transition_frames': 0,
            'annotations_count': 0
        }
        
        # Загружаем видео
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[WARNING] Не удалось открыть видео: {video_name}")
            continue
        
        video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames += video_frames
        video_stats[video_name]['total_frames'] = video_frames
        
        # Загружаем аннотации
        base = os.path.splitext(video_name)[0]
        if 'train' in video_path:
            ann_path = os.path.join(Config.TRAIN_ANNOTATION_PATH, base + '.json')
        else:
            ann_path = os.path.join(Config.VALID_ANNOTATION_PATH, base + '.json')
        
        if not os.path.exists(ann_path):
            print(f"[WARNING] Аннотации не найдены для {video_name}")
            cap.release()
            continue
        
        # Загружаем аннотации
        with open(ann_path, 'r') as f:
            ann_data = json.load(f)
            frame_labels = np.zeros((video_frames, 3), dtype=np.float32)  # 3 класса: фон, действие, переход
            
            # Считаем количество аннотаций
            video_stats[video_name]['annotations_count'] = len(ann_data['annotations'])
            
            for annotation in ann_data['annotations']:
                start_frame = annotation['start_frame']
                end_frame = annotation['end_frame']
                
                # Отмечаем действие
                for frame_idx in range(start_frame, end_frame + 1):
                    if frame_idx < len(frame_labels):
                        frame_labels[frame_idx, 1] = 1  # [0,1,0] - действие
                        video_stats[video_name]['action_frames'] += 1
                
                # Отмечаем переходы
                if start_frame < len(frame_labels):
                    frame_labels[start_frame, 2] = 1  # [0,0,1] - начало
                    video_stats[video_name]['transition_frames'] += 1
                if end_frame < len(frame_labels):
                    frame_labels[end_frame, 2] = 1  # [0,0,1] - конец
                    video_stats[video_name]['transition_frames'] += 1
        
        # Считаем фоновые кадры
        video_stats[video_name]['background_frames'] = video_frames - (
            video_stats[video_name]['action_frames'] + video_stats[video_name]['transition_frames']
        )
        
        # Выводим отладочную информацию для каждого видео
        print(f"\n[DEBUG] Обработка видео {video_name}:")
        print(f"  - Всего кадров: {video_frames}")
        print(f"  - Количество аннотаций: {video_stats[video_name]['annotations_count']}")
        print(f"  - Фоновых кадров: {video_stats[video_name]['background_frames']}")
        print(f"  - Кадров действия: {video_stats[video_name]['action_frames']}")
        print(f"  - Кадров перехода: {video_stats[video_name]['transition_frames']}")
        
        cap.release()
    
    # Рассчитываем веса классов
    total_background = sum(stats['background_frames'] for stats in video_stats.values())
    total_action = sum(stats['action_frames'] for stats in video_stats.values())
    total_transition = sum(stats['transition_frames'] for stats in video_stats.values())
    
    # Нормализуем веса
    max_count = max(total_background, total_action, total_transition)
    weights = {
        'MODEL_PARAMS': {
            'v3': {
                'dropout_rate': 0.3,
                'lstm_units': 128,
                'class_weights': {
                    'background': max_count / total_background if total_background > 0 else 1.0,
                    'action': max_count / total_action if total_action > 0 else 1.0,
                    'transition': max_count / total_transition if total_transition > 0 else 1.0
                },
                'base_input_shape': [224, 224, 3]
            },
            'v4': {
                'dropout_rate': 0.3,
                'expansion_factor': 4,
                'se_ratio': 0.25,
                'class_weights': {
                    'background': max_count / total_background if total_background > 0 else 1.0,
                    'action': max_count / total_action if total_action > 0 else 1.0,
                    'transition': max_count / total_transition if total_transition > 0 else 1.0
                },
                'base_input_shape': [224, 224, 3]
            }
        }
    }
    
    print("\n[INFO] Веса классов:")
    print(f"  - Фон: {weights['MODEL_PARAMS']['v3']['class_weights']['background']:.2f}")
    print(f"  - Действие: {weights['MODEL_PARAMS']['v3']['class_weights']['action']:.2f}")
    print(f"  - Переход: {weights['MODEL_PARAMS']['v3']['class_weights']['transition']:.2f}")
    
    return weights

def save_weights_to_config(weights):
    """
    Сохранение весов в конфигурационный файл
    """
    try:
        # Создаем директорию для конфигурации, если её нет
        config_dir = os.path.dirname(Config.CONFIG_PATH)
        if config_dir:  # Если путь содержит директории
            os.makedirs(config_dir, exist_ok=True)
            print(f"[DEBUG] Создана директория для конфигурации: {config_dir}")
        
        # Базовый конфиг
        default_config = {
            'MODEL_PARAMS': {
                'v3': {
                    'dropout_rate': 0.3,
                    'lstm_units': 128,
                    'class_weights': {
                        'background': None,
                        'action': None,
                        'transition': None
                    },
                    'base_input_shape': [224, 224, 3]
                },
                'v4': {
                    'dropout_rate': 0.3,
                    'expansion_factor': 4,
                    'se_ratio': 0.25,
                    'class_weights': {
                        'background': None,
                        'action': None,
                        'transition': None
                    },
                    'base_input_shape': [224, 224, 3]
                }
            }
        }
        
        # Проверяем существование файла
        if os.path.exists(Config.CONFIG_PATH):
            print(f"[DEBUG] Найден существующий конфигурационный файл: {Config.CONFIG_PATH}")
            try:
                with open(Config.CONFIG_PATH, 'r') as f:
                    config = json.load(f)
                print("[DEBUG] Успешно загружен существующий конфиг")
            except json.JSONDecodeError:
                print("[WARNING] Ошибка чтения конфигурационного файла, создаем новый")
                config = default_config
        else:
            print(f"[DEBUG] Конфигурационный файл не найден, создаем новый: {Config.CONFIG_PATH}")
            config = default_config
        
        # Обновляем веса в конфиге для обеих моделей
        config['MODEL_PARAMS']['v3']['class_weights'] = weights['MODEL_PARAMS']['v3']['class_weights']
        config['MODEL_PARAMS']['v4']['class_weights'] = weights['MODEL_PARAMS']['v4']['class_weights']
        
        # Сохраняем обновленный конфиг
        with open(Config.CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"[DEBUG] Веса успешно сохранены в {Config.CONFIG_PATH}")
        print(f"[DEBUG] Сохраненные веса: {weights}")
        
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