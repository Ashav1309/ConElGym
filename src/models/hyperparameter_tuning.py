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
# Включаем динамический рост памяти для всех GPU
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[DEBUG] Включён динамический рост памяти для {len(gpus)} GPU")
    else:
        print("[DEBUG] GPU не обнаружены")
except Exception as e:
    print(f"[DEBUG] Ошибка при настройке GPU: {e}")
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
from src.data_proc.data_validation import validate_data_pipeline

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
    Создание оптимизированного pipeline данных для подбора гиперпараметров
    
    Args:
        data_loader: загрузчик данных
        sequence_length: длина последовательности
        batch_size: Размер батча (берется из конфига)
        input_size: размер входного изображения
        is_training: флаг обучения
        force_positive: флаг принудительного добавления положительных примеров
    """
    try:
        print("\n[DEBUG] Создание pipeline данных для подбора гиперпараметров...")
        print(f"[DEBUG] Параметры:")
        print(f"  - sequence_length: {sequence_length}")
        print(f"  - batch_size: {batch_size}")
        print(f"  - input_size: {input_size}")
        print(f"  - is_training: {is_training}")
        print(f"  - force_positive: {force_positive}")
        
        # Устанавливаем размер батча в загрузчике
        data_loader.batch_size = batch_size
        
        # Создаем генератор данных
        def data_generator():
            while True:
                try:
                    batch_data = data_loader.get_batch(
                        batch_size=batch_size,
                        sequence_length=sequence_length,
                        target_size=input_size,
                        one_hot=True,
                        max_sequences_per_video=None,
                        force_positive=force_positive
                    )
                    
                    if batch_data is None:
                        print("[WARNING] Получен пустой батч данных")
                        continue
                        
                    X, y = batch_data
                    
                    if X.shape[0] == 0 or y.shape[0] == 0:
                        print("[WARNING] Получен батч с нулевой размерностью")
                        continue
                        
                    yield X, y
                    
                except Exception as e:
                    print(f"[ERROR] Ошибка в генераторе данных: {str(e)}")
                    continue
                
                break
        
        # Создаем dataset
        output_signature = (
            tf.TensorSpec(shape=(batch_size, sequence_length, *input_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(batch_size, sequence_length, Config.NUM_CLASSES), dtype=tf.float32)
        )
        
        dataset = tf.data.Dataset.from_generator(
            data_generator,
            output_signature=output_signature
        )
        
        # Оптимизация производительности
        if is_training:
            dataset = dataset.shuffle(64)
            dataset = dataset.batch(batch_size, drop_remainder=True)
        else:
            dataset = dataset.batch(batch_size)
            
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
        
    except Exception as e:
        print(f"[ERROR] Ошибка при создании pipeline данных: {str(e)}")
        print("[DEBUG] Stack trace:", flush=True)
        import traceback
        traceback.print_exc()
        raise

def create_and_compile_model(input_shape, num_classes, learning_rate, dropout_rate, lstm_units=None, model_type='v3', class_weights=None, rnn_type='lstm', temporal_block_type='rnn', clipnorm=1.0):
    """
    Создание и компиляция модели с заданными параметрами
    Args:
        input_shape: форма входных данных
        num_classes: количество классов (3 для three-hot encoding)
        learning_rate: скорость обучения
        dropout_rate: коэффициент dropout
        lstm_units: количество юнитов в LSTM слоях (только для v3)
        model_type: тип модели ('v3' или 'v4')
        class_weights: веса классов (если None, будут загружены из конфига)
        rnn_type: тип RNN ('lstm' или 'bigru')
        temporal_block_type: тип временного блока ('rnn' или 'hybrid')
        clipnorm: коэффициент градиентного клиппинга
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
    
    # Если class_weights не указаны, загружаем из конфига
    if class_weights is None:
        if os.path.exists(Config.CONFIG_PATH):
            with open(Config.CONFIG_PATH, 'r') as f:
                config = json.load(f)
                class_weights = config['class_weights']
        else:
            raise ValueError("Конфигурационный файл не найден. Сначала запустите calculate_weights.py")
    
    print(f"  - Class weights: {class_weights}")
    
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
    
    model, model_class_weights = create_model(
        input_shape=full_input_shape,
        num_classes=3,  # 3 класса: фон, действие, переход
        dropout_rate=dropout_rate,
        lstm_units=lstm_units,
        model_type=model_type,
        class_weights=class_weights,
        rnn_type=rnn_type,
        temporal_block_type=temporal_block_type
    )
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=clipnorm)
    
    # Включаем mixed precision если используется GPU
    if Config.DEVICE_CONFIG['use_gpu'] and Config.MEMORY_OPTIMIZATION['use_mixed_precision']:
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    
    # Создаем метрики для трех классов
    print("[DEBUG] Создание метрик...")
    metrics = [
        'accuracy',
        tf.keras.metrics.Precision(name='precision_background', class_id=0, thresholds=0.5),
        tf.keras.metrics.Precision(name='precision_action', class_id=1, thresholds=0.5),
        tf.keras.metrics.Precision(name='precision_transition', class_id=2, thresholds=0.5),
        tf.keras.metrics.Recall(name='recall_background', class_id=0, thresholds=0.5),
        tf.keras.metrics.Recall(name='recall_action', class_id=1, thresholds=0.5),
        tf.keras.metrics.Recall(name='recall_transition', class_id=2, thresholds=0.5)
    ]

    print("[DEBUG] Добавление F1Score...")
    try:
        # Создаем адаптер для F1Score для каждого класса
        class F1ScoreAdapter(tf.keras.metrics.F1Score):
            def __init__(self, name, class_id, threshold=0.5):
                super().__init__(name=name, threshold=threshold)
                self.class_id = class_id
                
            def update_state(self, y_true, y_pred, sample_weight=None):
                y_true = tf.argmax(y_true, axis=-1)
                y_pred = tf.argmax(y_pred, axis=-1)
                y_true = tf.reshape(y_true, [-1])
                y_pred = tf.reshape(y_pred, [-1])
                y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=3)
                y_pred = tf.one_hot(tf.cast(y_pred, tf.int32), depth=3)
                return super().update_state(y_true, y_pred, sample_weight)
            
            def result(self):
                result = super().result()
                return result[self.class_id]
        
        # Добавляем F1Score для каждого класса
        metrics.extend([
            F1ScoreAdapter(name='f1_score_background', class_id=0, threshold=0.5),
            F1ScoreAdapter(name='f1_score_action', class_id=1, threshold=0.5),
            F1ScoreAdapter(name='f1_score_transition', class_id=2, threshold=0.5)
        ])
        print(f"[DEBUG] F1Score создан успешно")
    except Exception as e:
        print(f"[ERROR] Ошибка при создании F1Score: {str(e)}")
        print("[DEBUG] Stack trace:", flush=True)
        import traceback
        traceback.print_exc()

    print(f"[DEBUG] Итоговый список метрик: {metrics}")

    # Компилируем модель с focal loss для трех классов
    model.compile(
        optimizer=optimizer,
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=metrics
    )
    
    print("[DEBUG] Модель успешно создана и скомпилирована")
    return model, model_class_weights

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
        val_dataset = create_data_pipeline(val_loader, Config.SEQUENCE_LENGTH, Config.BATCH_SIZE, Config.INPUT_SIZE, False, False)
        
        return train_dataset, val_dataset
    except Exception as e:
        print(f"[ERROR] Ошибка при загрузке данных: {str(e)}")
        print("[DEBUG] Stack trace:", flush=True)
        traceback.print_exc()
        clear_memory()
        raise

def get_num_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total

def count_total_sequences(video_paths, sequence_length, step):
    total = 0
    for video_path in video_paths:
        num_frames = get_num_frames(video_path)
        num_seq = max(0, (num_frames - sequence_length) // step + 1)
        total += num_seq
    return total

def objective(trial):
    """
    Функция оптимизации для Optuna
    """
    try:
        print("\n[DEBUG] Начало trial...")
        
        # Загружаем веса классов из конфига
        if os.path.exists(Config.CONFIG_PATH):
            with open(Config.CONFIG_PATH, 'r') as f:
                config = json.load(f)
                class_weights = config['class_weights']  # Изменено с config['MODEL_PARAMS'][Config.MODEL_TYPE]['class_weights']
        else:
            raise ValueError("Конфигурационный файл не найден. Сначала запустите calculate_weights.py")
        
        print(f"[DEBUG] Загруженные веса классов: {class_weights}")
        
        # Получаем параметры из trial
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-7, 1e-2, log=True),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
            'batch_size': trial.suggest_int('batch_size', 64, 512, step=8),
            'rnn_type': trial.suggest_categorical('rnn_type', ['lstm', 'bigru']),
            'temporal_block_type': trial.suggest_categorical('temporal_block_type', ['rnn', 'hybrid', '3d_attention']),
            'lstm_units': trial.suggest_int('lstm_units', 16, 512),
            'gamma': trial.suggest_float('gamma', 0.5, 5.0),
            'alpha': trial.suggest_float('alpha', 0.1, 0.4),
            'beta': trial.suggest_float('beta', 0.9, 0.999),  # Добавлен параметр beta
            'clipnorm': trial.suggest_float('clipnorm', 0.1, 5.0)
        }
        
        print("[DEBUG] Параметры trial:")
        for key, value in params.items():
            print(f"  - {key}: {value}")
        
        # Создаем и обучаем модель
        model, model_class_weights = create_model(
            input_shape=(Config.SEQUENCE_LENGTH, *Config.INPUT_SIZE, 3),
            num_classes=3,  # 3 класса: фон, действие, переход
            dropout_rate=params['dropout_rate'],
            rnn_type=params['rnn_type'],
            temporal_block_type=params['temporal_block_type'],
            lstm_units=params['lstm_units'],
            class_weights=class_weights
        )
        
        # Создаем загрузчики данных
        train_loader = VideoDataLoader(Config.TRAIN_DATA_PATH, max_videos=Config.MAX_VIDEOS)
        val_loader = VideoDataLoader(Config.VALID_DATA_PATH, max_videos=Config.MAX_VIDEOS)
        
        # Создаем оптимизированные pipeline данных
        train_dataset = create_data_pipeline(
            train_loader,
            Config.SEQUENCE_LENGTH,
            params['batch_size'],
            Config.INPUT_SIZE,
            is_training=True,
            force_positive=True
        )
        
        val_dataset = create_data_pipeline(
            val_loader,
            Config.SEQUENCE_LENGTH,
            params['batch_size'],
            Config.INPUT_SIZE,
            is_training=False,
            force_positive=False
        )
        
        # Создаем метрики для трех классов
        metrics = [
            'accuracy',
            tf.keras.metrics.Precision(name='precision_background', class_id=0, thresholds=0.5),
            tf.keras.metrics.Precision(name='precision_action', class_id=1, thresholds=0.5),
            tf.keras.metrics.Precision(name='precision_transition', class_id=2, thresholds=0.5),
            tf.keras.metrics.Recall(name='recall_background', class_id=0, thresholds=0.5),
            tf.keras.metrics.Recall(name='recall_action', class_id=1, thresholds=0.5),
            tf.keras.metrics.Recall(name='recall_transition', class_id=2, thresholds=0.5)
        ]
        
        # Создаем адаптер для F1Score
        class F1ScoreAdapter(tf.keras.metrics.F1Score):
            def __init__(self, name, class_id, threshold=0.5):
                super().__init__(name=name, threshold=threshold)
                self.class_id = class_id
                
            def update_state(self, y_true, y_pred, sample_weight=None):
                y_true = tf.argmax(y_true, axis=-1)
                y_pred = tf.argmax(y_pred, axis=-1)
                y_true = tf.reshape(y_true, [-1])
                y_pred = tf.reshape(y_pred, [-1])
                y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=3)
                y_pred = tf.one_hot(tf.cast(y_pred, tf.int32), depth=3)
                return super().update_state(y_true, y_pred, sample_weight)
            
            def result(self):
                result = super().result()
                return result[self.class_id]
        
        # Добавляем F1Score для каждого класса
        metrics.extend([
            F1ScoreAdapter(name='f1_score_background', class_id=0, threshold=0.5),
            F1ScoreAdapter(name='f1_score_action', class_id=1, threshold=0.5),
            F1ScoreAdapter(name='f1_score_transition', class_id=2, threshold=0.5)
        ])
        
        # Компилируем модель
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate'], clipnorm=params['clipnorm']),
            loss=focal_loss(gamma=params['gamma'], alpha=params['alpha'], beta=params['beta']),  # Добавлен параметр beta
            metrics=metrics
        )
        
        # Создаем callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_f1_score_action',  # Используем F1-score для класса действия
                patience=5,
                restore_best_weights=True,
                mode='max'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_f1_score_action',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                mode='max'
            )
        ]
        
        # Обучаем модель
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=Config.EPOCHS,
            callbacks=callbacks,
            verbose=1
        )
        
        # Получаем лучший F1-score для класса действия
        best_f1_score = max(history.history['val_f1_score_action'])
        print(f"[DEBUG] Лучший F1-score для класса действия: {best_f1_score}")
        
        return best_f1_score
        
    except Exception as e:
        print(f"[ERROR] Ошибка в objective: {str(e)}")
        print("[DEBUG] Stack trace:", flush=True)
        import traceback
        traceback.print_exc()
        raise

def save_tuning_results(study, total_time, n_trials):
    """
    Сохранение результатов подбора гиперпараметров + визуализация и подробный лог
    """
    try:
        print("\n[DEBUG] Сохранение результатов подбора гиперпараметров...")
        tuning_dir = os.path.join(Config.MODEL_SAVE_PATH, 'tuning')
        os.makedirs(tuning_dir, exist_ok=True)
        
        # Сохраняем результаты в текстовый файл
        results_file = os.path.join(tuning_dir, 'optuna_results.txt')
        with open(results_file, 'w') as f:
            f.write(f"=== Результаты оптимизации гиперпараметров ===\n")
            f.write(f"Модель: {Config.MODEL_TYPE}\n")
            f.write(f"Время выполнения: {timedelta(seconds=int(total_time))}\n")
            f.write(f"Количество trials: {n_trials}\n")
            f.write(f"Лучшее значение: {study.best_value}\n")
            f.write("\nЛучшие параметры:\n")
            for key, value in study.best_params.items():
                f.write(f"{key}: {value}\n")
            
            # Добавляем информацию о настройках модели
            f.write("\nНастройки модели:\n")
            f.write(f"  - sequence_length: {Config.SEQUENCE_LENGTH}\n")
            f.write(f"  - input_size: {Config.INPUT_SIZE}\n")
            f.write(f"  - batch_size: {Config.BATCH_SIZE}\n")
            f.write(f"  - epochs: {Config.EPOCHS}\n")
            f.write(f"  - class_weights: {Config.MODEL_PARAMS[Config.MODEL_TYPE]['class_weights']}\n")
        
        # Сохраняем результаты в JSON
        results_json = {
            'model_type': Config.MODEL_TYPE,
            'model_settings': {
                'sequence_length': Config.SEQUENCE_LENGTH,
                'input_size': Config.INPUT_SIZE,
                'batch_size': Config.BATCH_SIZE,
                'epochs': Config.EPOCHS,
                'class_weights': Config.MODEL_PARAMS[Config.MODEL_TYPE]['class_weights']
            },
            'execution_time': int(total_time),
            'n_trials': n_trials,
            'best_value': float(study.best_value) if study.best_value is not None else None,
            'best_params': study.best_params,
            'all_trials': []
        }
        
        for trial in study.trials:
            if trial.value is not None:
                trial_data = {
                    'number': trial.number,
                    'value': float(trial.value),
                    'params': trial.params,
                    'state': trial.state.name,
                    'duration': float(trial.duration) if trial.duration is not None else None
                }
                results_json['all_trials'].append(trial_data)
        
        json_file = os.path.join(tuning_dir, 'optuna_results.json')
        with open(json_file, 'w') as f:
            json.dump(results_json, f, indent=4)
        
        # Подробный лог по каждому trial
        log_file = os.path.join(tuning_dir, 'trial_logs.txt')
        with open(log_file, 'w') as flog:
            flog.write(f"=== Результаты оптимизации гиперпараметров ===\n")
            flog.write(f"Модель: {Config.MODEL_TYPE}\n")
            flog.write(f"Время выполнения: {timedelta(seconds=int(total_time))}\n")
            flog.write(f"Количество trials: {n_trials}\n")
            flog.write(f"Лучшее значение: {study.best_value}\n\n")
            
            flog.write("Настройки модели:\n")
            flog.write(f"  - sequence_length: {Config.SEQUENCE_LENGTH}\n")
            flog.write(f"  - input_size: {Config.INPUT_SIZE}\n")
            flog.write(f"  - batch_size: {Config.BATCH_SIZE}\n")
            flog.write(f"  - epochs: {Config.EPOCHS}\n")
            flog.write(f"  - class_weights: {Config.MODEL_PARAMS[Config.MODEL_TYPE]['class_weights']}\n\n")
            
            for trial in study.trials:
                flog.write(f"Trial {trial.number} | State: {trial.state.name}\n")
                if trial.value is not None:
                    flog.write(f"  Value: {trial.value}\n")
                for key, value in trial.params.items():
                    flog.write(f"  {key}: {value}\n")
                if trial.user_attrs:
                    flog.write(f"  User attrs: {trial.user_attrs}\n")
                if trial.system_attrs:
                    flog.write(f"  System attrs: {trial.system_attrs}\n")
                flog.write(f"  Duration: {trial.duration}\n")
                flog.write("-"*40 + "\n")
        
        # Визуализация результатов с использованием новой функции из config.py
        try:
            from src.config import plot_tuning_results
            plot_tuning_results(study)
            print("[DEBUG] Визуализация результатов успешно создана")
        except Exception as e:
            print(f"[WARNING] Не удалось создать визуализацию: {e}")
            print("[DEBUG] Stack trace:", flush=True)
            import traceback
            traceback.print_exc()
        
        print("[DEBUG] Результаты подбора гиперпараметров успешно сохранены")
        print(f"[DEBUG] Файлы сохранены в директории: {tuning_dir}")
        
    except Exception as e:
        print(f"[ERROR] Ошибка при сохранении результатов: {str(e)}")
        print("[DEBUG] Stack trace:", flush=True)
        import traceback
        traceback.print_exc()
        raise

def tune_hyperparameters(n_trials=None):
    """
    Запуск гиперпараметрического поиска с помощью Optuna
    Args:
        n_trials: количество trials (если None, берется из конфигурации)
    """
    import optuna
    import time
    print("\n[DEBUG] Начало подбора гиперпараметров...")
    
    # Используем значение из конфигурации, если n_trials не указан
    if n_trials is None:
        n_trials = Config.HYPERPARAM_TUNING['n_trials']
    
    print(f"[DEBUG] Количество trials: {n_trials}")
    
    # Валидация данных перед началом
    try:
        validate_data_pipeline()
    except Exception as e:
        print(f"[ERROR] Ошибка валидации данных: {str(e)}")
        raise
    
    start_time = time.time()
    study = optuna.create_study(direction='maximize', study_name='hyperparameter_tuning')
    study.optimize(objective, n_trials=n_trials)
    total_time = time.time() - start_time
    save_tuning_results(study, total_time, n_trials)
    print("[DEBUG] Гиперпараметрический поиск завершён!")
    return study
