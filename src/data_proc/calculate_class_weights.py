import os
import json
import cv2
from src.config import Config
from src.data_proc.data_loader import VideoDataLoader
import numpy as np
from tqdm import tqdm

def calculate_dataset_weights():
    """
    Расчет весов классов на всем датасете с учетом новой логики формирования батчей
    """
    print("[DEBUG] Начинаем расчет весов классов...")
    
    # Инициализируем счетчики
    total_sequences = 0
    positive_sequences = 0
    
    # Получаем список всех видео
    video_paths = []
    
    # Добавляем видео из тренировочного набора
    train_videos = [f for f in os.listdir(Config.TRAIN_DATA_PATH) if f.endswith('.mp4')]
    video_paths.extend([os.path.join(Config.TRAIN_DATA_PATH, v) for v in train_videos])
    
    # Добавляем видео из валидационного набора
    val_videos = [f for f in os.listdir(Config.VALID_DATA_PATH) if f.endswith('.mp4')]
    video_paths.extend([os.path.join(Config.VALID_DATA_PATH, v) for v in val_videos])
    
    print(f"[DEBUG] Всего найдено видео: {len(video_paths)}")
    
    # Обрабатываем каждое видео
    for video_path in tqdm(video_paths, desc="Обработка видео"):
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
            annotations_data = json.load(f)
            annotations = annotations_data.get('annotations', [])
            
        # Создаем массив меток для каждого кадра
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_labels = np.zeros((total_frames, Config.NUM_CLASSES), dtype=np.float32)
        
        # Заполняем метки
        for annotation in annotations:
            start_frame = annotation['start_frame']
            end_frame = annotation['end_frame']
            for frame_idx in range(start_frame, end_frame + 1):
                if frame_idx < len(frame_labels):
                    if frame_idx == start_frame:
                        frame_labels[frame_idx] = [1, 0]
                    elif frame_idx == end_frame:
                        frame_labels[frame_idx] = [0, 1]
                    else:
                        frame_labels[frame_idx] = [0, 0]
        
        # Считаем последовательности с учетом пропуска проблемных участков
        sequence_length = Config.SEQUENCE_LENGTH
        current_frame = 0
        
        while current_frame < total_frames:
            # Проверяем, можем ли мы прочитать последовательность
            if current_frame + sequence_length > total_frames:
                break
                
            # Пытаемся прочитать последовательность
            frames_read = 0
            sequence_labels = []
            
            for i in range(sequence_length):
                ret, _ = cap.read()
                if not ret:
                    # Если не удалось прочитать кадр, пропускаем этот участок
                    print(f"[DEBUG] Пропуск проблемного участка с кадра {current_frame}")
                    current_frame += sequence_length
                    break
                frames_read += 1
                sequence_labels.append(frame_labels[current_frame + i])
            
            if frames_read == sequence_length:
                # Проверяем, есть ли положительные примеры в последовательности
                sequence_labels = np.array(sequence_labels)
                if np.any(sequence_labels == 1):
                    positive_sequences += 1
                total_sequences += 1
                current_frame += sequence_length // 2  # Перекрытие последовательностей
            else:
                current_frame += sequence_length  # Пропускаем проблемный участок
        
        cap.release()
        
        print(f"[DEBUG] Видео {video_name}:")
        print(f"  - Всего последовательностей: {total_sequences}")
        print(f"  - Позитивных последовательностей: {positive_sequences}")
    
    # Рассчитываем веса на основе последовательностей
    negative_sequences = total_sequences - positive_sequences
    weights = [1.0, negative_sequences / positive_sequences if positive_sequences > 0 else 1.0]
    
    print("\n[DEBUG] Итоговая статистика:")
    print(f"Всего последовательностей: {total_sequences}")
    print(f"Позитивных последовательностей: {positive_sequences}")
    print(f"Негативных последовательностей: {negative_sequences}")
    print(f"Веса классов: {weights}")
    
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