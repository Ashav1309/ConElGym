import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Фильтрация логов TensorFlow
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Используем первую GPU
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async" 
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda-12.2'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
os.environ['TF_DISABLE_JIT'] = '1'
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')
import tensorflow as tf
# Отключаем JIT компиляцию
tf.config.optimizer.set_jit(False)

import optuna
from src.models.model import create_model
from src.data_proc.data_loader import VideoDataLoader
from src.config import Config
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime, timedelta
import time
import gc
import traceback
from tensorflow.keras.metrics import Precision, Recall, F1Score
import subprocess
import sys
import json
import cv2
from tensorflow.keras.optimizers import Adam
import psutil

# Объявляем глобальные переменные в начале файла
train_loader = None
val_loader = None

def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        # Преобразуем входные данные в тензоры
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)
        
        # Добавляем отладочную информацию
        print(f"[DEBUG] Focal Loss - Формы входных данных:")
        print(f"  - y_true shape: {y_true.shape}")
        print(f"  - y_pred shape: {y_pred.shape}")
        
        # Преобразуем one-hot encoded метки в индексы классов
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
        
        # Преобразуем 3D в 2D
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        
        # Преобразуем обратно в one-hot
        y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=2)
        y_pred = tf.one_hot(tf.cast(y_pred, tf.int32), depth=2)
        
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        # Вычисляем веса для каждого класса
        alpha_weight = alpha * y_true + (1 - alpha) * (1 - y_true)
        
        # Вычисляем фокусный вес
        pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = tf.pow(1 - pt, gamma)
        
        # Вычисляем кросс-энтропию
        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        
        # Применяем веса
        loss = alpha_weight * focal_weight * cross_entropy
        
        # Возвращаем среднее значение по батчу
        return tf.reduce_mean(loss)
    return focal_loss_fixed

def clear_memory():
    """Очистка памяти"""
    print("\n[DEBUG] ===== Начало очистки памяти =====")
    
    try:
        # Очищаем все сессии TensorFlow
        tf.keras.backend.clear_session()
        
        # Очистка Python garbage collector
        gc.collect()
        
        # Очистка CUDA кэша если используется GPU
        if Config.DEVICE_CONFIG['use_gpu']:
            try:
                # Сброс статистики памяти GPU
                tf.config.experimental.reset_memory_stats('GPU:0')
                
                # Принудительно очищаем CUDA кэш
                tf.keras.backend.clear_session()
                
                # Очищаем все переменные
                for var in tf.compat.v1.global_variables():
                    del var
                
                # Очищаем все операции
                tf.keras.backend.clear_session()
                
            except Exception as e:
                print(f"[DEBUG] ✗ Ошибка при очистке GPU: {str(e)}")
        
    except Exception as e:
        print(f"[DEBUG] ✗ Критическая ошибка при очистке памяти: {str(e)}")
        print("[DEBUG] Stack trace:", flush=True)
        traceback.print_exc()
    
    print("[DEBUG] ===== Очистка памяти завершена =====\n")

def setup_device():
    """Настройка устройства (CPU/GPU)"""
    try:
        if Config.DEVICE_CONFIG['use_gpu']:
            # Настройка GPU
            gpus = tf.config.list_physical_devices('GPU')
            if not gpus:
                print("No GPU devices found")
                return False
            
            # Настройка памяти GPU
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, Config.DEVICE_CONFIG['allow_gpu_memory_growth'])
            
            # Включаем mixed precision если нужно
            if Config.MEMORY_OPTIMIZATION['use_mixed_precision']:
                from tensorflow.keras.mixed_precision import Policy
                policy = Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                print("Mixed precision policy set:", policy.name)
            
            print("GPU optimization enabled")
            return True
        else:
            # Настройка CPU
            tf.config.set_visible_devices([], 'GPU')
            tf.config.threading.set_intra_op_parallelism_threads(Config.DEVICE_CONFIG['cpu_threads'])
            tf.config.threading.set_inter_op_parallelism_threads(Config.DEVICE_CONFIG['cpu_threads'])
            print("CPU optimization enabled")
            return True
            
    except RuntimeError as e:
        print(f"Error setting up device: {e}")
        return False

# Инициализация устройства
device_available = setup_device()

def create_data_pipeline(data_loader, batch_size, sequence_length, input_size, is_training=True):
    """
    Создание оптимизированного пайплайна данных.
    
    Args:
        data_loader: Загрузчик данных
        batch_size: Размер батча
        sequence_length: Длина последовательности
        input_size: Размер входного изображения
        is_training: Флаг обучения
        
    Returns:
        tf.data.Dataset: Оптимизированный датасет
    """
    try:
        print("\n[DEBUG] ===== Создание пайплайна данных =====")
        print(f"[DEBUG] Параметры:")
        print(f"  - batch_size: {batch_size}")
        print(f"  - sequence_length: {sequence_length}")
        print(f"  - input_size: {input_size}")
        print(f"  - is_training: {is_training}")
        
        # Проверяем количество загруженных видео
        if hasattr(data_loader, 'video_count'):
            print(f"[DEBUG] Количество загруженных видео: {data_loader.video_count}")
            if data_loader.video_count > Config.MAX_VIDEOS:
                print(f"[WARNING] Загружено слишком много видео: {data_loader.video_count} > {Config.MAX_VIDEOS}")
        
        # Устанавливаем размер батча в загрузчике данных
        data_loader.batch_size = batch_size
        
        # Создаем датасет из генератора
        dataset = tf.data.Dataset.from_generator(
            data_loader.data_generator,
            output_signature=(
                tf.TensorSpec(shape=(batch_size, sequence_length, *input_size, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(batch_size, sequence_length, Config.NUM_CLASSES), dtype=tf.float32)
            )
        )
        
        # Применяем оптимизации
        if Config.MEMORY_OPTIMIZATION['cache_dataset']:
            dataset = dataset.cache()
        
        if is_training:
            dataset = dataset.shuffle(buffer_size=1000)
        
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        print("[DEBUG] Pipeline данных успешно создан")
        return dataset
        
    except Exception as e:
        print(f"[ERROR] Ошибка при создании пайплайна данных: {str(e)}")
        print("[DEBUG] Stack trace:", flush=True)
        traceback.print_exc()
        raise

def create_and_compile_model(input_shape, num_classes, learning_rate, dropout_rate, lstm_units=None, model_type='v3', positive_class_weight=None):
    """
    Создание и компиляция модели с заданными параметрами
    Args:
        input_shape: форма входных данных
        num_classes: количество классов
        learning_rate: скорость обучения
        dropout_rate: коэффициент dropout
        lstm_units: количество юнитов в LSTM слоях (только для v3)
        model_type: тип модели ('v3' или 'v4')
        positive_class_weight: вес положительного класса (если None, будет рассчитан автоматически)
    """
    clear_memory()  # Очищаем память перед созданием модели
    
    print(f"[DEBUG] Creating model with parameters:")
    print(f"  - Model type: {model_type}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Dropout rate: {dropout_rate}")
    if model_type == 'v3':
        print(f"  - LSTM units: {lstm_units}")
    print(f"  - Input shape: {input_shape}")
    print(f"  - Number of classes: {num_classes}")
    
    # Если positive_class_weight не указан, рассчитываем его
    if positive_class_weight is None:
        positive_class_weight = calculate_balanced_weights(train_loader)
    
    print(f"  - Positive class weight: {positive_class_weight}")
    
    # Проверяем и корректируем input_shape
    if len(input_shape) == 3:  # Если это (height, width, channels)
        full_input_shape = (Config.SEQUENCE_LENGTH,) + input_shape
    elif len(input_shape) == 4:  # Если это (sequence_length, height, width, channels)
        full_input_shape = input_shape
    elif len(input_shape) == 5:  # Если есть лишняя размерность
        full_input_shape = tuple(s for i, s in enumerate(input_shape) if i != 1 or s != 1)
    else:
        raise ValueError(f"Неверная форма входных данных: {input_shape}")
    
    print(f"[DEBUG] Исправленный input_shape: {full_input_shape}")
    
    model, class_weights = create_model(
        input_shape=full_input_shape,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        lstm_units=lstm_units,
        model_type=model_type,
        positive_class_weight=positive_class_weight
    )
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Включаем mixed precision если используется GPU
    if Config.DEVICE_CONFIG['use_gpu'] and Config.MEMORY_OPTIMIZATION['use_mixed_precision']:
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    
    # Создаем метрики
    print("[DEBUG] Создание метрик...")
    metrics = [
        'accuracy',
        tf.keras.metrics.Precision(name='precision_element', class_id=1, thresholds=0.5),
        tf.keras.metrics.Recall(name='recall_element', class_id=1, thresholds=0.5)
    ]

    print("[DEBUG] Добавление F1Score...")
    try:
        # Создаем адаптер для F1Score
        class F1ScoreAdapter(tf.keras.metrics.F1Score):
            def update_state(self, y_true, y_pred, sample_weight=None):
                # Преобразуем one-hot encoded метки в индексы классов
                y_true = tf.argmax(y_true, axis=-1)
                y_pred = tf.argmax(y_pred, axis=-1)
                
                # Преобразуем 3D в 2D
                y_true = tf.reshape(y_true, [-1])
                y_pred = tf.reshape(y_pred, [-1])
                
                # Преобразуем обратно в one-hot
                y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=2)
                y_pred = tf.one_hot(tf.cast(y_pred, tf.int32), depth=2)
                
                return super().update_state(y_true, y_pred, sample_weight)
            
            def result(self):
                # Получаем результат от родительского класса
                result = super().result()
                # Возвращаем среднее значение по всем классам
                return tf.reduce_mean(result)
        
        f1_metric = F1ScoreAdapter(name='f1_score_element', threshold=0.5)
        print(f"[DEBUG] F1Score создан успешно: {f1_metric}")
        metrics.append(f1_metric)
    except Exception as e:
        print(f"[ERROR] Ошибка при создании F1Score: {str(e)}")
        print("[DEBUG] Stack trace:", flush=True)
        import traceback
        traceback.print_exc()

    print(f"[DEBUG] Итоговый список метрик: {metrics}")

    # Компилируем модель
    model.compile(
        optimizer=optimizer,
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=metrics
    )
    
    print("[DEBUG] Модель успешно создана и скомпилирована")
    return model, class_weights

def load_and_prepare_data(batch_size):
    """
    Загрузка и подготовка данных для обучения
    """
    print("[DEBUG] Начало загрузки данных...")
    clear_memory()  # Очищаем память перед загрузкой данных
    
    try:
        # Проверяем существование директорий
        if not os.path.exists(Config.TRAIN_DATA_PATH):
            raise FileNotFoundError(f"Директория с обучающими данными не найдена: {Config.TRAIN_DATA_PATH}")
        if not os.path.exists(Config.VALID_DATA_PATH):
            raise FileNotFoundError(f"Директория с валидационными данными не найдена: {Config.VALID_DATA_PATH}")
            
        print(f"[DEBUG] Проверка директорий успешна:")
        print(f"  - TRAIN_DATA_PATH: {Config.TRAIN_DATA_PATH}")
        print(f"  - VALID_DATA_PATH: {Config.VALID_DATA_PATH}")
        
        train_loader = VideoDataLoader(Config.TRAIN_DATA_PATH)
        val_loader = VideoDataLoader(Config.VALID_DATA_PATH)
        print("[DEBUG] VideoDataLoader создан успешно")
        
        target_size = Config.INPUT_SIZE
        
        # Создание оптимизированных pipeline данных
        train_dataset = create_data_pipeline(train_loader, Config.BATCH_SIZE, Config.SEQUENCE_LENGTH, Config.INPUT_SIZE, True)
        val_dataset = create_data_pipeline(val_loader, Config.BATCH_SIZE, Config.SEQUENCE_LENGTH, Config.INPUT_SIZE, False)
        
        return train_dataset, val_dataset
    except Exception as e:
        print(f"[ERROR] Ошибка при загрузке данных: {str(e)}")
        print("[DEBUG] Stack trace:", flush=True)
        traceback.print_exc()
        clear_memory()
        raise

def calculate_balanced_weights(data_loader):
    """
    Расчет сбалансированных весов классов для всего датасета
    """
    print("[DEBUG] Расчет весов классов для всего датасета...")
    
    total_samples = 0
    class_counts = {0: 0, 1: 0}
    
    # Проходим по всем видео в датасете
    for video_path in data_loader.video_paths:
        try:
            # Получаем путь к аннотации для видео
            annotation_path = data_loader.labels[data_loader.video_paths.index(video_path)]
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
                total_frames = data_loader.get_video_info(video_path)['total_frames']
                class_counts[0] += (total_frames - end_frame)
            
            total_samples = sum(class_counts.values())
            
        except Exception as e:
            print(f"[ERROR] Ошибка при обработке видео {video_path}: {str(e)}")
            continue
    
    if total_samples == 0:
        print("[WARNING] Не удалось подсчитать количество примеров")
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
    
    return weights[1]  # Возвращаем только вес положительного класса

def objective(trial):
    """
    Целевая функция для оптимизации гиперпараметров
    """
    try:
        print("\n[DEBUG] Начало нового trial...")
        
        # Очищаем память перед каждым trial
        clear_memory()
        
        # Определяем тип модели
        model_type = Config.MODEL_TYPE
        
        # Загружаем веса из конфигурационного файла
        if os.path.exists(Config.CONFIG_PATH):
            print(f"[DEBUG] Загрузка весов из {Config.CONFIG_PATH}")
            with open(Config.CONFIG_PATH, 'r') as f:
                config = json.load(f)
                positive_class_weight = config['MODEL_PARAMS'][model_type]['positive_class_weight']
                print(f"[DEBUG] Загружен вес положительного класса: {positive_class_weight}")
        else:
            print(f"[WARNING] Конфигурационный файл не найден: {Config.CONFIG_PATH}")
            positive_class_weight = None
        
        # Определяем гиперпараметры для оптимизации
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        
        if model_type == 'v3':
            lstm_units = trial.suggest_int('lstm_units', 32, 256)
        else:
            lstm_units = None
        
        print(f"[DEBUG] Параметры trial:")
        print(f"  - learning_rate: {learning_rate}")
        print(f"  - dropout_rate: {dropout_rate}")
        if lstm_units:
            print(f"  - lstm_units: {lstm_units}")
        print(f"  - positive_class_weight: {positive_class_weight}")
        
        # Создаем модель
        input_shape = (Config.SEQUENCE_LENGTH,) + Config.INPUT_SIZE + (3,)
        model, class_weights = create_model(
            input_shape=input_shape,
            num_classes=Config.NUM_CLASSES,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            lstm_units=lstm_units,
            model_type=model_type,
            positive_class_weight=positive_class_weight
        )
        
        # Создаем загрузчики данных
        train_loader = VideoDataLoader(Config.TRAIN_DATA_PATH, max_videos=None)
        val_loader = VideoDataLoader(Config.VALID_DATA_PATH, max_videos=None)
        
        # Создаем оптимизированные pipeline данных
        train_dataset = create_data_pipeline(
            train_loader,
            Config.SEQUENCE_LENGTH,
            Config.BATCH_SIZE,
            Config.INPUT_SIZE,
            is_training=True
        )
        
        val_dataset = create_data_pipeline(
            val_loader,
            Config.SEQUENCE_LENGTH,
            Config.BATCH_SIZE,
            Config.INPUT_SIZE,
            is_training=False
        )
        
        # Создаем метрики
        metrics = [
            'accuracy',
            tf.keras.metrics.Precision(name='precision_element', class_id=1, thresholds=0.5),
            tf.keras.metrics.Recall(name='recall_element', class_id=1, thresholds=0.5)
        ]
        
        # Создаем адаптер для F1Score
        class F1ScoreAdapter(tf.keras.metrics.F1Score):
            def update_state(self, y_true, y_pred, sample_weight=None):
                # Преобразуем one-hot encoded метки в индексы классов
                y_true = tf.argmax(y_true, axis=-1)
                y_pred = tf.argmax(y_pred, axis=-1)
                
                # Преобразуем 3D в 2D
                y_true = tf.reshape(y_true, [-1])
                y_pred = tf.reshape(y_pred, [-1])
                
                # Вызываем родительский метод
                super().update_state(y_true, y_pred, sample_weight)
        
        # Добавляем F1Score в метрики
        metrics.append(F1ScoreAdapter(name='f1_score_element', class_id=1, threshold=0.5))
        
        # Компилируем модель
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=focal_loss(),
            metrics=metrics
        )
        
        # Создаем колбэки
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_f1_score_element',
                patience=3,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_f1_score_element',
                factor=0.2,
                patience=2,
                min_lr=1e-6
            )
        ]
        
        # Обучаем модель
        history = model.fit(
            train_dataset,
            epochs=Config.EPOCHS,  # Используем количество эпох из конфигурации
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=1
        )
        
        # Получаем лучший F1-score
        best_f1 = max(history.history['val_f1_score_element'])
        
        # Очищаем память
        clear_memory()
        
        return best_f1
        
    except Exception as e:
        print(f"[ERROR] Ошибка в trial: {str(e)}")
        print("[DEBUG] Stack trace:", flush=True)
        import traceback
        traceback.print_exc()
        return None

def save_tuning_results(study, total_time, n_trials):
    """
    Сохранение результатов подбора гиперпараметров
    """
    try:
        print("\n[DEBUG] Сохранение результатов подбора гиперпараметров...")
        
        # Создаем директорию для результатов
        tuning_dir = os.path.join(Config.MODEL_SAVE_PATH, 'tuning')
        os.makedirs(tuning_dir, exist_ok=True)
        
        # Сохраняем результаты в текстовый файл
        with open(os.path.join(tuning_dir, 'tuning_results.txt'), 'w') as f:
            f.write(f"Время выполнения: {total_time:.2f} секунд\n")
            f.write(f"Количество триалов: {n_trials}\n\n")
            
            # Получаем лучший триал
            best_trial = study.best_trial
            f.write(f"Лучший триал: {best_trial.number}\n")
            f.write(f"Лучшее значение: {best_trial.value}\n")
            f.write("\nПараметры лучшего триала:\n")
            for key, value in best_trial.params.items():
                f.write(f"{key}: {value}\n")
            
            # Сохраняем историю всех триалов
            f.write("\nИстория всех триалов:\n")
            for trial in study.trials:
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    f.write(f"\nТриал {trial.number}:\n")
                    f.write(f"Значение: {trial.value}\n")
                    f.write("Параметры:\n")
                    for key, value in trial.params.items():
                        f.write(f"{key}: {value}\n")
        
        print("[DEBUG] Результаты подбора гиперпараметров успешно сохранены")
        
    except Exception as e:
        print(f"[ERROR] Ошибка при сохранении результатов: {str(e)}")
        print("[DEBUG] Stack trace:", flush=True)
        import traceback
        traceback.print_exc()
        raise

def plot_tuning_results(study):
    """
    Визуализация результатов подбора гиперпараметров
    """
    try:
        tuning_dir = os.path.join(Config.MODEL_SAVE_PATH, 'tuning')
        
        # График истории оптимизации
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_image(os.path.join(tuning_dir, 'optimization_history.png'))
        
        # График важности параметров
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_image(os.path.join(tuning_dir, 'param_importances.png'))
    except Exception as e:
        print(f"Warning: Could not create visualization plots: {str(e)}")

def tune_hyperparameters(n_trials=100):
    """
    Подбор гиперпараметров с использованием Optuna
    """
    global train_loader, val_loader
    try:
        print("\n[DEBUG] Начало подбора гиперпараметров...")
        start_time = time.time()

        # Инициализация загрузчиков данных без ограничения на количество видео
        train_loader = VideoDataLoader(Config.TRAIN_DATA_PATH, max_videos=None)
        val_loader = VideoDataLoader(Config.VALID_DATA_PATH, max_videos=None)
        print(f"[DEBUG] Загружено {len(train_loader.video_paths)} обучающих видео")
        print(f"[DEBUG] Загружено {len(val_loader.video_paths)} валидационных видео")

        # Рассчитываем веса классов для всего датасета
        positive_class_weight = calculate_balanced_weights(train_loader)
        if positive_class_weight is None:
            raise ValueError("Не удалось рассчитать веса классов")

        # Сохраняем вес в конфигурации
        Config.MODEL_PARAMS[Config.MODEL_TYPE]['positive_class_weight'] = positive_class_weight

        # Создаем study
        study = optuna.create_study(
            direction='maximize',
            study_name=f'hyperparameter_tuning_{Config.MODEL_TYPE}'
        )

        # Запускаем оптимизацию
        study.optimize(objective, n_trials=n_trials)

        # Сохраняем результаты
        save_tuning_results(study, time.time() - start_time, n_trials)

        return study

    except Exception as e:
        print(f"[ERROR] Ошибка при подборе гиперпараметров: {str(e)}")
        print("[DEBUG] Stack trace:", flush=True)
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    try:
        result = tune_hyperparameters()
        if result is not None:
            print("\nBest parameters:", result.best_params)
            print("Best validation accuracy:", result.best_value)
        else:
            print("\nFailed to find best parameters. Check the error messages above.")
    except Exception as e:
        print(f"\nError during hyperparameter tuning: {str(e)}") 