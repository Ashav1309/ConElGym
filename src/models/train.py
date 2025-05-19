import tensorflow as tf
from src.models.model import create_model_with_params
from src.data_proc.data_loader import VideoDataLoader
from src.config import Config, plot_training_results
import os
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    Callback,
    CSVLogger
)
import numpy as np
import gc
from tqdm.auto import tqdm
from tqdm.keras import TqdmCallback
from tensorflow.keras.metrics import Precision, Recall
import json
import re
import psutil
from src.data_proc.augmentation import VideoAugmenter, augment_rare_classes
from src.models.losses import focal_loss, F1ScoreAdapter
from src.models.metrics import get_training_metrics
from src.models.callbacks import AdaptiveThresholdCallback, get_training_callbacks
from src.utils.gpu_config import setup_gpu
import time
import argparse
import pickle

# Настройка GPU
setup_gpu()

# Глобальные переменные для кэширования данных
cached_train_sequences = None
cached_train_labels = None
cached_val_sequences = None
cached_val_labels = None

def clear_memory():
    """Очистка памяти"""
    global cached_train_sequences, cached_train_labels, cached_val_sequences, cached_val_labels
    
    # Очищаем все сессии TensorFlow
    tf.keras.backend.clear_session()
    
    # Очистка Python garbage collector
    gc.collect()
    
    # Очистка CUDA кэша если используется GPU
    if Config.DEVICE_CONFIG['use_gpu']:
        try:
            import numba
            numba.cuda.close()
        except:
            pass

def create_data_pipeline(loader, sequence_length, batch_size, target_size, is_training=True, force_positive=False, cache_dataset=False):
    """
    Создание оптимизированного pipeline данных для обучения и подбора гиперпараметров
    
    Args:
        loader: загрузчик данных
        sequence_length: длина последовательности
        batch_size: размер батча
        target_size: размер кадра
        is_training: флаг обучения
        force_positive: принудительно использовать положительные примеры
        cache_dataset: кэшировать датасет (используется только для небольших наборов данных)
    """
    global cached_train_sequences, cached_train_labels, cached_val_sequences, cached_val_labels
    
    try:
        print("\n[DEBUG] Создание pipeline данных...")
        print(f"[DEBUG] Параметры:")
        print(f"  - sequence_length: {sequence_length}")
        print(f"  - batch_size: {batch_size}")
        print(f"  - target_size: {target_size}")
        print(f"  - is_training: {is_training}")
        print(f"  - force_positive: {force_positive}")
        print(f"  - cache_dataset: {cache_dataset}")
        print(f"[DEBUG] RAM до создания датасета: {psutil.virtual_memory().used / 1024**3:.2f} GB")

        # Проверяем наличие кэшированных данных
        if is_training and cached_train_sequences is not None and cached_train_labels is not None:
            print("[DEBUG] Используем кэшированные обучающие данные...")
            dataset = tf.data.Dataset.from_tensor_slices((cached_train_sequences, cached_train_labels))
            dataset = dataset.shuffle(len(cached_train_sequences))
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(Config.MEMORY_OPTIMIZATION['prefetch_buffer_size'])
            return dataset
            
        if not is_training and cached_val_sequences is not None and cached_val_labels is not None:
            print("[DEBUG] Используем кэшированные валидационные данные...")
            dataset = tf.data.Dataset.from_tensor_slices((cached_val_sequences, cached_val_labels))
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(Config.MEMORY_OPTIMIZATION['prefetch_buffer_size'])
            return dataset

        def generator():
            sequences = []
            labels = []
            
            while True:
                # Если все видео обработаны, загружаем новую порцию
                if len(loader.processed_video_paths) >= len(loader.video_paths):
                    print("[DEBUG] Все видео обработаны, загружаем новую порцию")
                    loader.current_batch_videos = []
                    loader.video_paths = loader._get_video_paths()
                    loader.processed_video_paths = set()
                    continue

                X, y = loader._get_sequence(
                    sequence_length=sequence_length,
                    target_size=target_size,
                    force_positive=force_positive,
                    is_validation=not is_training
                )
                
                if X is None or y is None:
                    print("[DEBUG] Получена пустая последовательность, пропускаем")
                    continue
                
                if len(X) == 0 or len(y) == 0:
                    print("[DEBUG] Получена последовательность нулевой длины, пропускаем")
                    continue
                
                if not np.any(y):  # Проверка на пустые метки
                    print("[DEBUG] Получены пустые метки, пропускаем")
                    continue

                y_one_hot = np.zeros((sequence_length, 2), dtype=np.float32)
                for i in range(sequence_length):
                    try:
                        if isinstance(y[i], np.ndarray):
                            if y[i].size == 2:
                                label = np.argmax(y[i])
                            else:
                                label = int(y[i].item())
                        else:
                            label = int(y[i])
                        y_one_hot[i, label] = 1
                    except Exception as e:
                        print(f"[ERROR] Ошибка при обработке метки {i}: {str(e)}")
                        print(f"[DEBUG] Значение y[{i}]: {y[i]}")
                        print(f"[DEBUG] Тип y[{i}]: {type(y[i])}")
                        if isinstance(y[i], np.ndarray):
                            print(f"[DEBUG] Форма y[{i}]: {y[i].shape}")
                        continue  # Пропускаем проблемную метку вместо прерывания

                # Сохраняем последовательности для кэширования
                if cache_dataset:
                    sequences.append(X)
                    labels.append(y_one_hot)
                    
                    # Если накопили достаточно последовательностей, кэшируем их
                    if len(sequences) >= Config.MEMORY_OPTIMIZATION['cache_size']:
                        if is_training:
                            cached_train_sequences = np.array(sequences)
                            cached_train_labels = np.array(labels)
                        else:
                            cached_val_sequences = np.array(sequences)
                            cached_val_labels = np.array(labels)
                        sequences = []
                        labels = []

                yield X, y_one_hot

        output_signature = (
            tf.TensorSpec(shape=(sequence_length, *target_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(sequence_length, 2), dtype=tf.float32)
        )

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=output_signature
        )

        if cache_dataset and (not hasattr(loader, 'video_count') or loader.video_count <= 50):
            dataset = dataset.cache()
        if is_training:
            dataset = dataset.shuffle(Config.MEMORY_OPTIMIZATION['shuffle_buffer_size'])
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(Config.MEMORY_OPTIMIZATION['prefetch_buffer_size'])

        if is_training and Config.AUGMENTATION['enabled']:
            dataset = dataset.map(
                lambda x, y: tf.py_function(
                    lambda x, y: augment_rare_classes(x, y, is_training=True),
                    [x, y],
                    [tf.float32, tf.float32]
                ),
                num_parallel_calls=Config.MEMORY_OPTIMIZATION['num_parallel_calls']
            )

        print(f"[DEBUG] RAM после создания датасета: {psutil.virtual_memory().used / 1024**3:.2f} GB")
        print("[DEBUG] Pipeline данных успешно создан")
        return dataset

    except Exception as e:
        print(f"[ERROR] Ошибка при создании pipeline данных: {str(e)}")
        print("[DEBUG] Stack trace:", flush=True)
        import traceback
        traceback.print_exc()
        raise

def create_tuning_data_pipeline(data_loader, sequence_length, batch_size, target_size, force_positive=False):
    """
    Создание оптимизированного pipeline данных для подбора гиперпараметров
    
    Args:
        data_loader: VideoDataLoader
        sequence_length: длина последовательности
        batch_size: размер батча
        target_size: целевой размер изображения
        force_positive: принудительно использовать положительные примеры
        
    Returns:
        tf.data.Dataset: оптимизированный dataset
    """
    try:
        print("\n[DEBUG] Создание pipeline данных для подбора гиперпараметров...")
        print(f"[DEBUG] Параметры:")
        print(f"  - sequence_length: {sequence_length}")
        print(f"  - batch_size: {batch_size}")
        print(f"  - target_size: {target_size}")
        print(f"  - force_positive: {force_positive}")
        
        def generator():
            while True:  # Бесконечный цикл для повторного использования данных
                try:
                    # Если все видео обработаны, загружаем новую порцию
                    if len(data_loader.processed_videos) >= len(data_loader.video_paths):
                        print("[DEBUG] Все видео обработаны, загружаем новую порцию")
                        data_loader.current_batch_videos = []
                        data_loader.video_paths = data_loader._get_video_paths()
                        data_loader.processed_videos = set()
                        continue

                    X, y = data_loader._get_sequence(
                        sequence_length=sequence_length,
                        target_size=target_size,
                        force_positive=force_positive,
                        is_validation=True
                    )
                    
                    if X is None or y is None:
                        print("[DEBUG] Получена пустая последовательность, пропускаем")
                        continue
                        
                    if len(X) == 0 or len(y) == 0:
                        print("[DEBUG] Получена последовательность нулевой длины, пропускаем")
                        continue
                        
                    if not np.any(y):  # Проверка на пустые метки
                        print("[DEBUG] Получены пустые метки, пропускаем")
                        continue
                    
                    y_one_hot = np.zeros((sequence_length, 2), dtype=np.float32)
                    for i in range(sequence_length):
                        try:
                            if isinstance(y[i], np.ndarray):
                                if y[i].size == 2:
                                    label = np.argmax(y[i])
                                    y_one_hot[i, label] = 1
                                else:
                                    label = int(y[i].item())
                                    y_one_hot[i, label] = 1
                            else:
                                label = int(y[i])
                                y_one_hot[i, label] = 1
                        except Exception as e:
                            print(f"[ERROR] Ошибка при обработке метки {i}: {str(e)}")
                            print(f"[DEBUG] Значение y[{i}]: {y[i]}")
                            print(f"[DEBUG] Тип y[{i}]: {type(y[i])}")
                            if isinstance(y[i], np.ndarray):
                                print(f"[DEBUG] Форма y[{i}]: {y[i].shape}")
                            continue  # Пропускаем проблемную метку вместо прерывания
                    
                    yield X, y_one_hot
                    
                except Exception as e:
                    print(f"[ERROR] Ошибка в генераторе: {str(e)}")
                    continue

        output_signature = (
            tf.TensorSpec(shape=(sequence_length, *target_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(sequence_length, 2), dtype=tf.float32)
        )
        
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=output_signature
        )
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(Config.MEMORY_OPTIMIZATION['prefetch_buffer_size'])
        
        if Config.AUGMENTATION['enabled']:
            dataset = dataset.map(
                lambda x, y: tf.py_function(
                    lambda x, y: augment_rare_classes(x, y, is_training=False),  # Меняем на False для валидации
                    [x, y],
                    [tf.float32, tf.float32]
                ),
                num_parallel_calls=Config.MEMORY_OPTIMIZATION['num_parallel_calls']
            )
        
        print("[DEBUG] Pipeline данных для подбора гиперпараметров успешно создан")
        return dataset
        
    except Exception as e:
        print(f"[ERROR] Ошибка при создании pipeline данных: {str(e)}")
        print("[DEBUG] Stack trace:", flush=True)
        traceback.print_exc()
        raise

class OverfittingMonitor(Callback):
    """Мониторинг переобучения"""
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold
        self.overfitting_epochs = 0
        
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
            
        train_acc = logs.get('accuracy')
        val_acc = logs.get('val_accuracy')
        
        if train_acc is not None and val_acc is not None:
            diff = train_acc - val_acc
            if diff > self.threshold:
                self.overfitting_epochs += 1
                print(f"\nWarning: Possible overfitting detected! "
                      f"Train-Val accuracy difference: {diff:.4f}")
            else:
                self.overfitting_epochs = 0

class TrainingPlotter(Callback):
    def __init__(self, save_path):
        super().__init__()
        self.save_path = save_path
        self.epochs = []
        self.losses = []
        self.val_losses = []
        self.accuracies = []
        self.val_accuracies = []
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
    def on_epoch_end(self, epoch, logs=None):
        self.epochs.append(epoch)
        self.losses.append(logs['loss'])
        self.val_losses.append(logs['val_loss'])
        self.accuracies.append(logs['accuracy'])
        self.val_accuracies.append(logs['val_accuracy'])
        
        # Очищаем предыдущие графики
        self.ax1.clear()
        self.ax2.clear()
        
        # График потерь
        self.ax1.plot(self.epochs, self.losses, label='Training Loss')
        self.ax1.plot(self.epochs, self.val_losses, label='Validation Loss')
        self.ax1.set_title('Loss')
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Loss')
        self.ax1.legend()
        self.ax1.grid(True, linestyle='--', alpha=0.7)
        self.ax1.xticks(fontsize=12, rotation=45)
        self.ax1.yticks(fontsize=12)
        
        # График точности
        self.ax2.plot(self.epochs, self.accuracies, label='Training Accuracy')
        self.ax2.plot(self.epochs, self.val_accuracies, label='Validation Accuracy')
        self.ax2.set_title('Accuracy')
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('Accuracy')
        self.ax2.legend()
        self.ax2.grid(True, linestyle='--', alpha=0.7)
        self.ax2.xticks(fontsize=12, rotation=45)
        self.ax2.yticks(fontsize=12)
        
        # Сохраняем графики
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, 'training_plot.png'))
        plt.close()

def plot_confusion_matrix(y_true, y_pred, save_path):
    """Визуализация матрицы ошибок"""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))
    plt.close()

def load_best_params(model_type=None):
    """
    Загрузка лучших параметров из файла best_params.json
    Args:
        model_type: тип модели ('v3' или 'v4'). Если None, используется Config.MODEL_TYPE
    """
    try:
        model_type = model_type or Config.MODEL_TYPE
        results_dir = os.path.join(Config.MODEL_SAVE_PATH, 'tuning', model_type)
        params_path = os.path.join(results_dir, 'best_params.json')

        # Проверяем существование директории
        if not os.path.exists(results_dir):
            print(f"[DEBUG] Создание директории для результатов: {results_dir}")
            os.makedirs(results_dir, exist_ok=True)

        # Загружаем веса классов из config_weights.json
        if os.path.exists(Config.CONFIG_PATH):
            print(f"[DEBUG] Загрузка весов классов из {Config.CONFIG_PATH}")
            with open(Config.CONFIG_PATH, 'r') as f:
                config = json.load(f)
                class_weights = config.get('class_weights', {'background': 1.0, 'action': 1.0})
                positive_class_weight = class_weights.get('action', 1.0)
        else:
            print(f"[WARNING] Конфигурационный файл не найден: {Config.CONFIG_PATH}")
            class_weights = {'background': 1.0, 'action': 1.0}
            positive_class_weight = 1.0

        if not os.path.exists(params_path):
            print(f"[DEBUG] Файл с параметрами не найден. Используем параметры по умолчанию для {model_type}.")
            default_params = {
                'learning_rate': 1e-4,
                'dropout_rate': 0.3,
                'batch_size': Config.BATCH_SIZE,
                'positive_class_weight': positive_class_weight
            }
            if model_type == 'v3':
                default_params['lstm_units'] = 128
            return default_params

        print(f"[DEBUG] Загрузка параметров из {params_path}")
        with open(params_path, 'r') as f:
            params = json.load(f)
            # Объединяем параметры модели и аугментации
            best_params = {**params['model_params'], **params['augmentation_params']}
            best_params['positive_class_weight'] = positive_class_weight
            print(f"[DEBUG] Загружены лучшие параметры для {model_type}: {best_params}")
            return best_params

    except Exception as e:
        print(f"[ERROR] Ошибка при загрузке параметров: {str(e)}")
        print("[DEBUG] Stack trace:", flush=True)
        import traceback
        traceback.print_exc()
        positive_class_weight = 1.0

    print(f"[DEBUG] Не удалось загрузить параметры для {model_type}. Используем параметры по умолчанию.")
    default_params = {
        'learning_rate': 1e-4,
        'dropout_rate': 0.3,
        'batch_size': Config.BATCH_SIZE,
        'positive_class_weight': positive_class_weight
    }
    if model_type == 'v3':
        default_params['lstm_units'] = 128
    return default_params

def cache_all_data(data_loader, sequence_length, target_size, force_positive, is_validation):
    sequences = []
    labels = []
    while len(data_loader.processed_video_paths) < len(data_loader.video_paths):
        X, y = data_loader._get_sequence(
            sequence_length=sequence_length,
            target_size=target_size,
            force_positive=force_positive,
            is_validation=is_validation
        )
        if X is not None and y is not None and len(X) > 0 and len(y) > 0 and np.any(y):
            y_one_hot = np.zeros((sequence_length, 2), dtype=np.float32)
            for i in range(sequence_length):
                if isinstance(y[i], np.ndarray):
                    if y[i].size == 2:
                        label = np.argmax(y[i])
                    else:
                        label = int(y[i].item())
                else:
                    label = int(y[i])
                y_one_hot[i, label] = 1
            sequences.append(X)
            labels.append(y_one_hot)
    return np.array(sequences), np.array(labels)

def train(model_type: str = None, epochs: int = 50, batch_size: int = None):
    if model_type is None:
        model_type = Config.MODEL_TYPE
    if batch_size is None:
        batch_size = Config.BATCH_SIZE
    print(f"[DEBUG] Выбран тип модели для обучения: {model_type}")
    try:
        # Загружаем лучшие параметры
        best_params = load_best_params(model_type)
        if best_params is None:
            print("[WARNING] Не удалось загрузить лучшие параметры, используем значения по умолчанию")
            best_params = {
                'learning_rate': 1e-4,
                'dropout_rate': 0.3,
                'batch_size': batch_size,
                'positive_class_weight': 1.0
            }
        # Загружаем веса классов из config_weights.json (только для модели)
        if os.path.exists(Config.CONFIG_PATH):
            with open(Config.CONFIG_PATH, 'r') as f:
                config = json.load(f)
                class_weights = config.get('class_weights', {'background': 1.0, 'action': 1.0})
        else:
            class_weights = {'background': 1.0, 'action': 1.0}

        # Используем create_tuning_data_pipeline для генерации данных (как при тюнинге)
        print("[DEBUG] Создание обучающего pipeline данных...")
        train_loader = VideoDataLoader(
            Config.TRAIN_DATA_PATH,
            max_videos=Config.MAX_VIDEOS if hasattr(Config, 'MAX_VIDEOS') else None
        )
        train_data = create_tuning_data_pipeline(
            train_loader,
            Config.SEQUENCE_LENGTH,
            batch_size,
            Config.INPUT_SIZE,
            force_positive=True
        )

        print("[DEBUG] Создание валидационного pipeline данных...")
        val_loader = VideoDataLoader(
            Config.VALID_DATA_PATH,
            max_videos=Config.MAX_VIDEOS if hasattr(Config, 'MAX_VIDEOS') else None
        )
        val_data = create_tuning_data_pipeline(
            val_loader,
            Config.SEQUENCE_LENGTH,
            batch_size,
            Config.INPUT_SIZE,
            force_positive=False
        )

        print(f"[DEBUG] Используемые веса классов для модели: {class_weights}")
        
        # Получаем callbacks
        callbacks = get_training_callbacks(val_data)
        
        # Создаем модель
        model = create_model_with_params(
            model_type=model_type,
            input_shape=(Config.SEQUENCE_LENGTH, *Config.INPUT_SIZE, 3),
            num_classes=Config.NUM_CLASSES,
            params=best_params,
            class_weights=class_weights
        )
        
        # Обучаем модель (без class_weight)
        history = model.fit(
            train_data,
            epochs=epochs,
            validation_data=val_data,
            callbacks=callbacks,
            verbose=1
        )
        
        # Сохраняем метаданные модели и саму модель
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join(Config.MODEL_SAVE_PATH, f"model_{model_type}_{timestamp}")
        os.makedirs(model_dir, exist_ok=True)
        
        # Сохраняем модель
        model_path = os.path.join(model_dir, f"best_model_{model_type}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump({'model': model}, f)
        print(f"[INFO] Модель сохранена: {model_path}")
        
        # Сохраняем метаданные
        metadata = {
            'model_type': model_type,
            'epochs': epochs,
            'batch_size': batch_size,
            'best_params': best_params,
            'class_weights': class_weights,
            'history': history.history
        }
        with open(os.path.join(model_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=4)
        
        # Сохраняем графики обучения
        plot_training_results(history, model_dir)
        
        return model, history
        
    except Exception as e:
        print(f"[ERROR] Ошибка при обучении модели: {str(e)}")
        print("[DEBUG] Stack trace:", flush=True)
        import traceback
        traceback.print_exc()
        # Аварийное сохранение модели
        if 'model' in locals():
            emergency_path = f"emergency_save_{model_type}.pkl"
            with open(emergency_path, 'wb') as f:
                pickle.dump({'model': model}, f)
            print(f"[WARNING] Модель аварийно сохранена: {emergency_path}")
        raise

def plot_training_results(history, save_dir):
    """
    Сохранение графиков процесса обучения с улучшенной читаемостью
    
    Args:
        history: история обучения модели
        save_dir: директория для сохранения графиков
    """
    import matplotlib.ticker as mticker
    # Создаем директорию для графиков
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    font = {'size': 14}
    plt.rc('font', **font)
    
    # График функции потерь и F1
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train', linewidth=2, marker='o')
    plt.plot(history.history['val_loss'], label='Validation', linewidth=2, marker='s')
    plt.title('Loss', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=12, rotation=45)
    plt.yticks(fontsize=12)
    
    plt.subplot(1, 2, 2)
    if 'f1_score' in history.history and 'val_f1_score' in history.history:
        plt.plot(history.history['f1_score'], label='Train', linewidth=2, marker='o')
        plt.plot(history.history['val_f1_score'], label='Validation', linewidth=2, marker='s')
        plt.title('F1 Score', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('F1 Score', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(fontsize=12, rotation=45)
        plt.yticks(fontsize=12)
    else:
        plt.text(0.5, 0.5, 'Нет данных F1-score', fontsize=16, ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'training_history.png'))
    plt.close()
    
    # График точности и AUC
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    if 'accuracy' in history.history and 'val_accuracy' in history.history:
        plt.plot(history.history['accuracy'], label='Train', linewidth=2, marker='o')
        plt.plot(history.history['val_accuracy'], label='Validation', linewidth=2, marker='s')
        plt.title('Accuracy', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(fontsize=12, rotation=45)
        plt.yticks(fontsize=12)
    else:
        plt.text(0.5, 0.5, 'Нет данных Accuracy', fontsize=16, ha='center')
    
    plt.subplot(1, 2, 2)
    if 'auc' in history.history and 'val_auc' in history.history:
        plt.plot(history.history['auc'], label='Train', linewidth=2, marker='o')
        plt.plot(history.history['val_auc'], label='Validation', linewidth=2, marker='s')
        plt.title('AUC', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('AUC', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(fontsize=12, rotation=45)
        plt.yticks(fontsize=12)
    else:
        plt.text(0.5, 0.5, 'Нет данных AUC', fontsize=16, ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'metrics.png'))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обучение модели (v3 или v4)")
    parser.add_argument('--model_type', type=str, default=None, help='Тип модели: v3 или v4')
    args = parser.parse_args()

    model_type = args.model_type if args.model_type is not None else Config.MODEL_TYPE
    print(f"Обучение основной модели: {model_type}")
    model, history = train(model_type) 