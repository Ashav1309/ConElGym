import os
import json
import cv2
from src.config import Config
from src.data_proc.data_loader import VideoDataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

def visualize_class_weights(weights, save_path=None):
    """
    Визуализация весов классов
    
    Args:
        weights (dict): Словарь с весами классов
        save_path (str, optional): Путь для сохранения графика
    """
    # Получаем веса классов
    class_names = list(weights['class_weights'].keys())
    weight_values = list(weights['class_weights'].values())
    
    # Создаем фигуру
    plt.figure(figsize=(12, 6))
    
    # Создаем bar plot
    bars = plt.bar(class_names, weight_values)
    
    # Добавляем значения над столбцами
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    # Настраиваем график
    plt.title('Распределение весов классов', fontsize=14, pad=20)
    plt.xlabel('Классы', fontsize=12)
    plt.ylabel('Веса', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Добавляем логарифмическую шкалу для лучшей визуализации
    plt.yscale('log')
    
    # Настраиваем отступы
    plt.tight_layout()
    
    # Сохраняем или показываем график
    if save_path:
        plt.savefig(save_path)
        print(f"\n[INFO] График сохранен в {save_path}")
    else:
        plt.show()
    
    plt.close()

def visualize_data_processing(video_stats, save_dir):
    """
    Визуализация процесса обработки данных
    
    Args:
        video_stats (dict): Статистика по видео
        save_dir (str): Директория для сохранения графиков
    """
    # Создаем директорию для графиков
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Распределение кадров по классам
    plt.figure(figsize=(15, 6))
    
    # Подготовка данных
    video_names = list(video_stats.keys())
    background_frames = [stats['background_frames'] for stats in video_stats.values()]
    action_frames = [stats['action_frames'] for stats in video_stats.values()]
    transition_frames = [stats['transition_frames'] for stats in video_stats.values()]
    
    # Создаем stacked bar plot
    plt.bar(video_names, background_frames, label='Фон', alpha=0.7)
    plt.bar(video_names, action_frames, bottom=background_frames, label='Действие', alpha=0.7)
    plt.bar(video_names, transition_frames, 
            bottom=[b + a for b, a in zip(background_frames, action_frames)], 
            label='Переход', alpha=0.7)
    
    plt.title('Распределение кадров по классам в каждом видео', fontsize=14, pad=20)
    plt.xlabel('Видео', fontsize=12)
    plt.ylabel('Количество кадров', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'frame_distribution.png'))
    plt.close()
    
    # 2. Соотношение классов
    plt.figure(figsize=(10, 6))
    total_frames = sum(background_frames) + sum(action_frames) + sum(transition_frames)
    class_ratios = [
        sum(background_frames) / total_frames * 100,
        sum(action_frames) / total_frames * 100,
        sum(transition_frames) / total_frames * 100
    ]
    plt.pie(class_ratios, labels=['Фон', 'Действие', 'Переход'], 
            autopct='%1.1f%%', startangle=90)
    plt.title('Соотношение классов в датасете', fontsize=14, pad=20)
    plt.savefig(os.path.join(save_dir, 'class_ratios.png'))
    plt.close()
    
    # 3. Тепловая карта статистики
    plt.figure(figsize=(12, 8))
    stats_matrix = np.array([
        background_frames,
        action_frames,
        transition_frames,
        [stats['annotations_count'] for stats in video_stats.values()]
    ])
    sns.heatmap(stats_matrix, 
                annot=True, 
                fmt='d',
                cmap='YlOrRd',
                xticklabels=video_names,
                yticklabels=['Фон', 'Действие', 'Переход', 'Аннотации'])
    plt.title('Тепловая карта статистики по видео', fontsize=14, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'stats_heatmap.png'))
    plt.close()

def normalize_weights(weights, method='log'):
    """
    Нормализация весов классов
    
    Args:
        weights (dict): Словарь с весами классов
        method (str): Метод нормализации ('log' или 'minmax')
    
    Returns:
        dict: Нормализованные веса
    """
    weight_values = list(weights['class_weights'].values())
    
    if method == 'log':
        # Логарифмическая нормализация
        normalized_values = np.log1p(weight_values)
        # Масштабирование к диапазону [1, 10]
        scaler = MinMaxScaler(feature_range=(1, 10))
        normalized_values = scaler.fit_transform(normalized_values.reshape(-1, 1)).flatten()
    else:
        # MinMax нормализация
        scaler = MinMaxScaler(feature_range=(1, 10))
        normalized_values = scaler.fit_transform(np.array(weight_values).reshape(-1, 1)).flatten()
    
    # Создаем новый словарь с нормализованными весами
    normalized_weights = {
        'class_weights': dict(zip(weights['class_weights'].keys(), normalized_values))
    }
    
    return normalized_weights

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
                        if frame_labels[frame_idx, 1] == 0:  # Если кадр еще не помечен как действие
                            frame_labels[frame_idx, 1] = 1  # [0,1,0] - действие
                            video_stats[video_name]['action_frames'] += 1
                
                # Отмечаем переходы
                if start_frame < len(frame_labels):
                    if frame_labels[start_frame, 2] == 0:  # Если кадр еще не помечен как переход
                        frame_labels[start_frame, 2] = 1  # [0,0,1] - начало
                        video_stats[video_name]['transition_frames'] += 1
                if end_frame < len(frame_labels):
                    if frame_labels[end_frame, 2] == 0:  # Если кадр еще не помечен как переход
                        frame_labels[end_frame, 2] = 1  # [0,0,1] - конец
                        video_stats[video_name]['transition_frames'] += 1
        
            # Считаем фоновые кадры
            # Сначала считаем уникальные кадры действия и перехода
            action_frames = np.sum(frame_labels[:, 1] == 1)  # Количество кадров действия
            transition_frames = np.sum(frame_labels[:, 2] == 1)  # Количество кадров перехода
            # Считаем кадры, которые являются и действием, и переходом
            overlapping_frames = np.sum((frame_labels[:, 1] == 1) & (frame_labels[:, 2] == 1))
            # Вычитаем из общего числа кадров действия и переходы, учитывая перекрытие
            video_stats[video_name]['background_frames'] = video_frames - (action_frames + transition_frames - overlapping_frames)
        
        # Выводим отладочную информацию для каждого видео
        print(f"\n[DEBUG] Обработка видео {video_name}:")
        print(f"  - Всего кадров: {video_frames}")
        print(f"  - Количество аннотаций: {video_stats[video_name]['annotations_count']}")
        print(f"  - Фоновых кадров: {video_stats[video_name]['background_frames']}")
        print(f"  - Кадров действия: {action_frames}")
        print(f"  - Кадров перехода: {transition_frames}")
        print(f"  - Перекрывающихся кадров: {overlapping_frames}")
        
        cap.release()
    
    # Рассчитываем веса классов
    total_background = sum(stats['background_frames'] for stats in video_stats.values())
    total_action = sum(stats['action_frames'] for stats in video_stats.values())
    total_transition = sum(stats['transition_frames'] for stats in video_stats.values())
    
    # Нормализуем веса
    max_count = max(total_background, total_action, total_transition)
    raw_weights = {
        'class_weights': {
            'background': max_count / total_background if total_background > 0 else 1.0,
            'action': max_count / total_action if total_action > 0 else 1.0,
            'transition': max_count / total_transition if total_transition > 0 else 1.0
        }
    }
    
    # Создаем директории для графиков
    plots_dir = os.path.join(os.path.dirname(Config.CONFIG_PATH), 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Визуализируем процесс обработки данных
    visualize_data_processing(video_stats, plots_dir)
    
    # Нормализуем веса
    normalized_weights = normalize_weights(raw_weights, method='log')
    
    print("\n[INFO] Исходные веса классов:")
    print(f"  - Фон: {raw_weights['class_weights']['background']:.2f}")
    print(f"  - Действие: {raw_weights['class_weights']['action']:.2f}")
    print(f"  - Переход: {raw_weights['class_weights']['transition']:.2f}")
    
    print("\n[INFO] Нормализованные веса классов:")
    print(f"  - Фон: {normalized_weights['class_weights']['background']:.2f}")
    print(f"  - Действие: {normalized_weights['class_weights']['action']:.2f}")
    print(f"  - Переход: {normalized_weights['class_weights']['transition']:.2f}")
    
    # Визуализируем веса до и после нормализации
    plt.figure(figsize=(15, 6))
    
    # Исходные веса
    plt.subplot(1, 2, 1)
    bars1 = plt.bar(raw_weights['class_weights'].keys(), raw_weights['class_weights'].values())
    plt.title('Исходные веса классов', fontsize=12)
    plt.yscale('log')
    for bar in bars1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    # Нормализованные веса
    plt.subplot(1, 2, 2)
    bars2 = plt.bar(normalized_weights['class_weights'].keys(), 
                   normalized_weights['class_weights'].values())
    plt.title('Нормализованные веса классов', fontsize=12)
    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'weights_comparison.png'))
    plt.close()
    
    return normalized_weights

def save_weights_to_config(weights):
    """
    Сохранение весов в конфигурационный файл
    """
    try:
        # Создаем директорию для конфигурации, если её нет
        config_dir = os.path.dirname(Config.CONFIG_PATH)
        if config_dir:  # Если путь содержит директории
            os.makedirs(config_dir, exist_ok=True)
        
        # Создаем конфиг только с весами классов
        config = {
            "class_weights": weights['class_weights']
        }
        
        # Сохраняем конфиг
        with open(Config.CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=4)
        
        print("\n[INFO] Нормализованные веса успешно сохранены:")
        print(f"  - Фон: {weights['class_weights']['background']:.2f}")
        print(f"  - Действие: {weights['class_weights']['action']:.2f}")
        print(f"  - Переход: {weights['class_weights']['transition']:.2f}")
        
    except Exception as e:
        print(f"\n[ERROR] Ошибка при сохранении весов: {str(e)}")
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