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
from src.models.model import create_mobilenetv3_model, create_mobilenetv4_model, create_model, focal_loss
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
from src.data_proc.data_augmentation import VideoAugmenter
from optuna.trial import Trial
from src.utils.network_handler import NetworkErrorHandler, NetworkMonitor

# Объявляем глобальные переменные в начале файла
train_loader = None
val_loader = None

def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
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

def create_data_pipeline(data_loader, sequence_length, batch_size, input_size, is_training=True, force_positive=False):
    """
    Создание оптимизированного пайплайна данных.
    Args:
        data_loader: Загрузчик данных
        sequence_length: Длина последовательности
        batch_size: Размер батча
        input_size: Размер входного изображения
        is_training: Флаг обучения
        force_positive: Флаг принудительного включения положительных примеров
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
        print(f"  - force_positive: {force_positive}")
        
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
        positive_class_weight: вес положительного класса (если None, будет загружен из конфига)
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
    
    # Если positive_class_weight не указан, загружаем из конфига
    if positive_class_weight is None:
        if os.path.exists(Config.CONFIG_PATH):
            with open(Config.CONFIG_PATH, 'r') as f:
                config = json.load(f)
                positive_class_weight = config['MODEL_PARAMS'][model_type]['positive_class_weight']
        else:
            raise ValueError("Конфигурационный файл не найден. Сначала запустите calculate_weights.py")
    
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
                y_true = tf.argmax(y_true, axis=-1)
                y_pred = tf.argmax(y_pred, axis=-1)
                y_true = tf.reshape(y_true, [-1])
                y_pred = tf.reshape(y_pred, [-1])
                y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=2)
                y_pred = tf.one_hot(tf.cast(y_pred, tf.int32), depth=2)
                return super().update_state(y_true, y_pred, sample_weight)
            
            def result(self):
                result = super().result()
                return float(tf.reduce_mean(result))
        
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
        
        train_loader = VideoDataLoader(Config.TRAIN_DATA_PATH, max_videos=Config.MAX_VIDEOS)
        val_loader = VideoDataLoader(Config.VALID_DATA_PATH, max_videos=Config.MAX_VIDEOS)
        print("[DEBUG] VideoDataLoader создан успешно")
        
        target_size = Config.INPUT_SIZE
        
        # Создание оптимизированных pipeline данных
        train_dataset = create_data_pipeline(train_loader, Config.SEQUENCE_LENGTH, Config.BATCH_SIZE, Config.INPUT_SIZE, True, True)
        val_dataset = create_data_pipeline(val_loader, Config.SEQUENCE_LENGTH, Config.BATCH_SIZE, Config.INPUT_SIZE, False, True)
        
        return train_dataset, val_dataset
    except Exception as e:
        print(f"[ERROR] Ошибка при загрузке данных: {str(e)}")
        print("[DEBUG] Stack trace:", flush=True)
        traceback.print_exc()
        clear_memory()
        raise

def objective(trial):
    """
    Целевая функция для оптимизации гиперпараметров
    """
    try:
        print(f"\n[DEBUG] >>> Начало нового trial #{trial.number}")
        # Очищаем память перед каждым trial
        clear_memory()
        
        # Определяем тип модели
        model_type = Config.MODEL_TYPE
        
        # Загружаем веса из конфигурационного файла
        if os.path.exists(Config.CONFIG_PATH):
            print(f"[DEBUG] Загрузка весов из {Config.CONFIG_PATH}")
            with open(Config.CONFIG_PATH, 'r') as f:
                config = json.load(f)
                base_weight = config['MODEL_PARAMS'][model_type]['positive_class_weight']
                
                # Проверяем корректность базового веса
                if base_weight is None or base_weight <= 0:
                    print("[WARNING] Некорректный базовый вес, используем значение по умолчанию")
                    base_weight = 10.0
                
                # Добавляем случайное отклонение ±30%
                weight_variation = base_weight * 0.3  # 30% от базового веса
                positive_class_weight = trial.suggest_float(
                    'positive_class_weight',
                    max(1.0, base_weight - weight_variation),  # Минимальный вес 1.0
                    base_weight + weight_variation
                )
                print(f"[DEBUG] Базовый вес: {base_weight}")
                print(f"[DEBUG] Загружен вес положительного класса с вариацией: {positive_class_weight}")
        else:
            print(f"[WARNING] Конфигурационный файл не найден: {Config.CONFIG_PATH}")
            raise ValueError("Конфигурационный файл не найден. Сначала запустите calculate_weights.py")
        
        # Определяем гиперпараметры для оптимизации
        learning_rate = trial.suggest_float('learning_rate', 1e-7, 1e-2, log=True)  # Расширяем диапазон
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)  # Увеличиваем верхнюю границу
        
        if model_type == 'v3':
            lstm_units = trial.suggest_int('lstm_units', 16, 512)  # Уменьшаем диапазон для экономии памяти
        else:
            lstm_units = None
        
        # Оптимизируем параметры focal loss
        gamma = trial.suggest_float('gamma', 0.5, 5.0)
        alpha = trial.suggest_float('alpha', 0.1, 0.4)
        
        print(f"[DEBUG] Параметры trial:")
        print(f"  - learning_rate: {learning_rate}")
        print(f"  - dropout_rate: {dropout_rate}")
        if lstm_units:
            print(f"  - lstm_units: {lstm_units}")
        print(f"  - positive_class_weight: {positive_class_weight}")
        print(f"  - gamma: {gamma}")
        print(f"  - alpha: {alpha}")
        
        # Создаем модель
        input_shape = (Config.SEQUENCE_LENGTH,) + Config.INPUT_SIZE + (3,)
        model, class_weights = create_and_compile_model(
            input_shape=input_shape,
            num_classes=Config.NUM_CLASSES,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            lstm_units=lstm_units,
            model_type=model_type,
            positive_class_weight=positive_class_weight
        )
        
        # Создаем загрузчики данных
        train_loader = VideoDataLoader(Config.TRAIN_DATA_PATH, max_videos=Config.MAX_VIDEOS)
        val_loader = VideoDataLoader(Config.VALID_DATA_PATH, max_videos=Config.MAX_VIDEOS)
        
        # Создаем оптимизированные pipeline данных
        train_dataset = create_data_pipeline(
            train_loader,
            Config.SEQUENCE_LENGTH,
            Config.BATCH_SIZE,
            Config.INPUT_SIZE,
            is_training=True,
            force_positive=True
        )
        
        val_dataset = create_data_pipeline(
            val_loader,
            Config.SEQUENCE_LENGTH,
            Config.BATCH_SIZE,
            Config.INPUT_SIZE,
            is_training=False,
            force_positive=True
        )
        
        # Создаем метрики
        metrics = [
            'accuracy',
            tf.keras.metrics.Precision(name='precision_element', class_id=1, thresholds=0.5),
            tf.keras.metrics.Recall(name='recall_element', class_id=1, thresholds=0.5),
            tf.keras.metrics.AUC(name='auc')  # Добавляем AUC
        ]
        
        # Создаем адаптер для F1Score
        class F1ScoreAdapter(tf.keras.metrics.F1Score):
            def update_state(self, y_true, y_pred, sample_weight=None):
                y_true = tf.argmax(y_true, axis=-1)
                y_pred = tf.argmax(y_pred, axis=-1)
                y_true = tf.reshape(y_true, [-1])
                y_pred = tf.reshape(y_pred, [-1])
                y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=2)
                y_pred = tf.one_hot(tf.cast(y_pred, tf.int32), depth=2)
                return super().update_state(y_true, y_pred, sample_weight)
        
        # Добавляем F1Score в метрики
        metrics.append(F1ScoreAdapter(name='f1_score_element', threshold=0.5))
        
        # Компилируем модель с оптимизированными параметрами focal loss
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=focal_loss(gamma=gamma, alpha=alpha),
            metrics=metrics
        )
        
        # Создаем колбэки с улучшенными параметрами
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_f1_score_element',
                patience=5,  # Увеличиваем с 3 до 5
                restore_best_weights=True,
                mode='max'  # Явно указываем режим максимизации
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_f1_score_element',
                factor=0.1,  # Уменьшаем с 0.2 до 0.1
                patience=3,  # Увеличиваем с 2 до 3
                min_lr=1e-7,  # Уменьшаем с 1e-6 до 1e-7
                mode='max'  # Явно указываем режим максимизации
            )
        ]
        
        # Обучаем модель
        history = model.fit(
            train_dataset,
            epochs=Config.EPOCHS,
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=1,
            class_weight=class_weights
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
        return float('-inf')

def save_tuning_results(study, total_time, n_trials):
    """
    Сохранение результатов подбора гиперпараметров
    """
    try:
        print("\n[DEBUG] Сохранение результатов подбора гиперпараметров...")
        
        # Создаем директорию для результатов
        tuning_dir = os.path.join(Config.MODEL_SAVE_PATH, 'tuning')
        os.makedirs(tuning_dir, exist_ok=True)
        
        # Загружаем базовые веса из конфигурации
        if os.path.exists(Config.CONFIG_PATH):
            with open(Config.CONFIG_PATH, 'r') as f:
                config = json.load(f)
                base_weight = config['MODEL_PARAMS'][Config.MODEL_TYPE]['positive_class_weight']
        else:
            base_weight = None
        
        # Сохраняем результаты в текстовый файл
        with open(os.path.join(tuning_dir, 'optuna_results.txt'), 'w') as f:
            f.write(f"Время выполнения: {total_time:.2f} секунд\n")
            f.write(f"Количество триалов: {n_trials}\n")
            if base_weight:
                f.write(f"Базовый вес положительного класса: {base_weight}\n")
            f.write("\n")
            
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
                    # Добавляем информацию о весах классов
                    if 'positive_class_weight' in trial.params:
                        weight = trial.params['positive_class_weight']
                        if base_weight:
                            weight_diff = ((weight - base_weight) / base_weight) * 100
                            f.write(f"Отклонение веса от базового: {weight_diff:.2f}%\n")
        
        # Сохраняем результаты в JSON для удобства загрузки
        results = {
            'best_trial': {
                'number': best_trial.number,
                'value': best_trial.value,
                'params': best_trial.params
            },
            'base_weight': base_weight,
            'trials': [
                {
                    'number': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                    'state': trial.state.name
                }
                for trial in study.trials
                if trial.state == optuna.trial.TrialState.COMPLETE
            ]
        }
        
        with open(os.path.join(tuning_dir, 'optuna_results.json'), 'w') as f:
            json.dump(results, f, indent=4)
        
        print("[DEBUG] Результаты подбора гиперпараметров успешно сохранены")
        
    except Exception as e:
        print(f"[ERROR] Ошибка при сохранении результатов: {str(e)}")
        print("[DEBUG] Stack trace:", flush=True)
        import traceback
        traceback.print_exc()
        raise

def tune_hyperparameters(n_trials=20):
    """
    Запуск гиперпараметрического поиска с помощью Optuna
    """
    import optuna
    import time
    print("\n[DEBUG] Начало подбора гиперпараметров...")
    start_time = time.time()
    study = optuna.create_study(direction='maximize', study_name='hyperparameter_tuning')
    study.optimize(objective, n_trials=n_trials)
    total_time = time.time() - start_time
    save_tuning_results(study, total_time, n_trials)
    print("[DEBUG] Гиперпараметрический поиск завершён!")
    return study
