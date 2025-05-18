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
from src.models.model import (
    create_model, create_mobilenetv3_model, create_mobilenetv4_model
)
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
from src.data_proc.data_validation import validate_data_pipeline, validate_training_data
from src.models.losses import focal_loss, F1ScoreAdapter
from src.models.metrics import f1_score_element

# Объявляем глобальные переменные в начале файла
train_loader = None
val_loader = None
train_data = None
val_data = None

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

        def generator():
            while True:
                # Получаем следующую последовательность
                X, y = data_loader._get_sequence(
                    sequence_length=sequence_length,
                    target_size=input_size,
                    force_positive=force_positive
                )
                if X is not None and y is not None:
                    yield X, y

        # Создаем dataset напрямую из генератора
        output_signature = (
            tf.TensorSpec(shape=(sequence_length, *input_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(sequence_length, 3), dtype=tf.float32)  # three-hot encoding
        )

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=output_signature
        )

        # Оптимизация производительности
        if is_training:
            dataset = dataset.shuffle(64)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        print(f"[DEBUG] RAM после создания датасета: {psutil.virtual_memory().used / 1024**3:.2f} GB")
        print("[DEBUG] Pipeline данных успешно создан")
        return dataset

    except Exception as e:
        print(f"[ERROR] Ошибка при создании pipeline данных: {str(e)}")
        print("[DEBUG] Stack trace:", flush=True)
        import traceback
        traceback.print_exc()
        raise

def create_and_compile_model(params, input_shape, num_classes=2, class_weights=None):
    """
    Создает и компилирует модель с заданными параметрами
    """
    print("\n[DEBUG] Создание модели со следующими параметрами:")
    print(f"  - Model type: {params['model_type']}")
    print(f"  - Learning rate: {params['learning_rate']}")
    print(f"  - Dropout rate: {params['dropout_rate']}")
    print(f"  - LSTM units: {params['lstm_units']}")
    print(f"  - Input shape: {input_shape}")
    print(f"  - Number of classes: {num_classes}")
    print(f"  - RNN type: {params['rnn_type']}")
    print(f"  - Temporal block type: {params['temporal_block_type']}")
    print(f"  - Base class weights: {class_weights}")
    
    # Создаем модель
    model = create_model(
        input_shape=input_shape,
        num_classes=num_classes,
        dropout_rate=params['dropout_rate'],
        lstm_units=params['lstm_units'],
        model_type=params['model_type'],
        class_weights=class_weights,
        rnn_type=params['rnn_type'],
        temporal_block_type=params['temporal_block_type']
    )
    
    # Оптимизатор
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=params['learning_rate'],
        clipnorm=1.0
    )
    
    # Метрики для двухклассовой модели
    metrics = [
        'accuracy',
        tf.keras.metrics.Precision(name='precision_action', class_id=1, thresholds=0.5),  # метрика для класса "действие"
        tf.keras.metrics.Recall(name='recall_action', class_id=1, thresholds=0.5),        # метрика для класса "действие"
        F1ScoreAdapter(name='f1_score_action', class_id=1, threshold=0.5)                 # F1-score для класса "действие"
    ]
    
    # Компилируем модель с focal loss для двух классов
    model.compile(
        optimizer=optimizer,
        loss=focal_loss(gamma=2.0, alpha=[class_weights['background'], class_weights['action']]),
        metrics=metrics
    )
    
    return model

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
    try:
        print(f"\n[DEBUG] Начало триала #{trial.number}")
        
        # Очищаем память перед началом триала
        clear_memory()
        
        # Получаем гиперпараметры
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        lstm_units = trial.suggest_int('lstm_units', 32, 256)
        model_type = Config.MODEL_TYPE
        rnn_type = trial.suggest_categorical('rnn_type', ['lstm', 'gru'])
        temporal_block_type = trial.suggest_categorical('temporal_block_type', ['rnn', 'tcn'])
        clipnorm = trial.suggest_float('clipnorm', 0.1, 2.0)
        
        # Подбираем размер батча с шагом 8
        batch_size = trial.suggest_int('batch_size', 8, 64, step=8)
        print(f"[DEBUG] Выбран размер батча: {batch_size}")
        
        # Загружаем базовые веса из конфига
        try:
            with open(Config.CONFIG_PATH, 'r') as f:
                config = json.load(f)
                base_weights = config['class_weights']
        except:
            print("[WARNING] Не удалось загрузить веса классов из конфига. Используем значения по умолчанию.")
            base_weights = {
                'background': 1.0,
                'action': 4.3
            }
        
        # Подбираем веса с отклонением ±30% от базовых значений
        weight_deviation = trial.suggest_float('weight_deviation', -0.3, 0.3)
        action_weight = base_weights['action'] * (1 + weight_deviation)
        
        # Ограничиваем максимальный вес действия до 10.0
        action_weight = min(action_weight, 10.0)
        
        class_weights = {
            'background': 1.0,  # Фон всегда 1.0
            'action': action_weight
        }
        
        print(f"[DEBUG] Подобранные веса классов:")
        print(f"  - Базовые веса: {base_weights}")
        print(f"  - Отклонение: {weight_deviation:.2%}")
        print(f"  - Итоговые веса: {class_weights}")
        
        # Загружаем данные для текущего триала
        train_data, val_data = load_and_prepare_data(batch_size)
        
        # Создаем и компилируем модель
        model = create_and_compile_model(
            params={
                'learning_rate': learning_rate,
                'dropout_rate': dropout_rate,
                'lstm_units': lstm_units,
                'model_type': model_type,
                'rnn_type': rnn_type,
                'temporal_block_type': temporal_block_type
            },
            input_shape=(Config.SEQUENCE_LENGTH, *Config.INPUT_SIZE, 3),
            num_classes=Config.NUM_CLASSES,
            class_weights=class_weights
        )
        
        # Создаем callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_f1_score',
                patience=5,
                restore_best_weights=True,
                mode='max'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_f1_score',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                mode='max'
            ),
            tf.keras.callbacks.ModelCheckpoint(
                f'best_model_trial_{trial.number}.h5',
                monitor='val_f1_score',
                save_best_only=True,
                mode='max'
            ),
            tf.keras.callbacks.CSVLogger(f'trial_{trial.number}_history.csv')
        ]
        
        # Обучаем модель
        history = model.fit(
            train_data,
            epochs=Config.HYPERPARAM_TUNING['epochs'],
            validation_data=val_data,
            callbacks=callbacks,
            verbose=1
        )
        
        # Очищаем память после обучения
        clear_memory()
        
        # Возвращаем среднее значение F1-score за последние 3 эпохи
        if 'val_f1_score' in history.history:
            return np.mean(history.history['val_f1_score'][-3:])
        else:
            print("[WARNING] Метрика val_f1_score не найдена в истории")
            return float('-inf')
            
    except Exception as e:
        print(f"[ERROR] Ошибка в objective: {str(e)}")
        print("[DEBUG] Stack trace:", flush=True)
        traceback.print_exc()
        clear_memory()  # Очищаем память в случае ошибки
        return float('-inf')

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
    """Подбор гиперпараметров с использованием Optuna"""
    try:
        print("[DEBUG] Начало подбора гиперпараметров...")
        
        # Создаем study
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=5,
                interval_steps=1
            )
        )
        
        # Запускаем оптимизацию
        start_time = time.time()
        study.optimize(
            objective,
            n_trials=n_trials or Config.HYPERPARAM_TUNING['n_trials'],
            timeout=Config.HYPERPARAM_TUNING['timeout'],
            show_progress_bar=True
        )
        total_time = time.time() - start_time
        
        # Сохраняем результаты
        save_tuning_results(study, total_time, n_trials or Config.HYPERPARAM_TUNING['n_trials'])
        
        return study
        
    except Exception as e:
        print(f"[ERROR] Ошибка при подборе гиперпараметров: {str(e)}")
        print("[DEBUG] Stack trace:", flush=True)
        traceback.print_exc()
        raise
