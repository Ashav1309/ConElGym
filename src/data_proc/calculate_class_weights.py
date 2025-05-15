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
    total_frames = 0
    positive_frames = set()  # Используем множество для уникальных кадров
    video_stats = {}  # Словарь для хранения статистики по каждому видео
    
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
        video_stats[video_name] = {
            'total_frames': 0,
            'positive_frames': set(),
            'total_sequences': 0,
            'positive_sequences': 0
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
            frame_labels = np.zeros((video_frames, Config.NUM_CLASSES), dtype=np.float32)
            
            for annotation in ann_data['annotations']:
                start_frame = annotation['start_frame']
                end_frame = annotation['end_frame']
                
                # Добавляем уникальные положительные кадры
                positive_frames.add(start_frame)
                positive_frames.add(end_frame)
                video_stats[video_name]['positive_frames'].add(start_frame)
                video_stats[video_name]['positive_frames'].add(end_frame)
                
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
        
        while current_frame < video_frames:
            # Проверяем, можем ли мы прочитать последовательность
            if current_frame + sequence_length > video_frames:
                break
            
            # Проверяем наличие положительных примеров в последовательности
            sequence_labels = frame_labels[current_frame:current_frame + sequence_length]
            # Используем np.maximum вместо побитового OR для проверки наличия положительных классов
            has_positive = np.any(np.maximum(sequence_labels[:, 0], sequence_labels[:, 1]) > 0)
            
            if has_positive:
                positive_sequences += 1
                video_stats[video_name]['positive_sequences'] += 1
            total_sequences += 1
            video_stats[video_name]['total_sequences'] += 1
            
            # Переходим к следующей последовательности с перекрытием
            current_frame += sequence_length // 2
        
        cap.release()
    
    # Рассчитываем веса на основе последовательностей и кадров
    negative_sequences = total_sequences - positive_sequences
    sequence_weight = negative_sequences / positive_sequences if positive_sequences > 0 else 1.0
    
    # Учитываем также соотношение положительных и отрицательных кадров
    positive_frames_count = len(positive_frames)  # Используем количество уникальных кадров
    negative_frames = total_frames - positive_frames_count
    frame_weight = negative_frames / positive_frames_count if positive_frames_count > 0 else 1.0
    
    # Используем взвешенное среднее для комбинирования весов
    # Даем больший вес последовательностям, так как они важнее для обучения
    sequence_weight_factor = 0.7
    frame_weight_factor = 0.3
    final_weight = (sequence_weight * sequence_weight_factor + frame_weight * frame_weight_factor)
    
    # Выводим детальную статистику по каждому видео
    print("\n[DEBUG] Детальная статистика по видео:")
    for video_name, stats in video_stats.items():
        print(f"\nВидео: {video_name}")
        print(f"  - Всего кадров: {stats['total_frames']}")
        print(f"  - Позитивных кадров: {len(stats['positive_frames'])}")
        print(f"  - Всего последовательностей: {stats['total_sequences']}")
        print(f"  - Позитивных последовательностей: {stats['positive_sequences']}")
        if stats['total_sequences'] > 0:
            pos_ratio = stats['positive_sequences'] / stats['total_sequences']
            print(f"  - Доля позитивных последовательностей: {pos_ratio:.2%}")
    
    print("\n[DEBUG] Итоговая статистика:")
    print(f"Всего последовательностей: {total_sequences}")
    print(f"Позитивных последовательностей: {positive_sequences}")
    print(f"Негативных последовательностей: {negative_sequences}")
    print(f"Всего кадров: {total_frames}")
    print(f"Позитивных кадров: {positive_frames_count}")
    print(f"Негативных кадров: {negative_frames}")
    print(f"Вес на основе последовательностей: {sequence_weight:.2f}")
    print(f"Вес на основе кадров: {frame_weight:.2f}")
    print(f"Итоговый вес: {final_weight:.2f}")
    
    # Проверяем корректность весов
    if final_weight < 1.0:
        print("[WARNING] Итоговый вес меньше 1.0, что может указывать на проблемы с данными")
    elif final_weight > 100.0:
        print("[WARNING] Итоговый вес больше 100.0, что может привести к нестабильному обучению")
    
    return [1.0, final_weight]

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