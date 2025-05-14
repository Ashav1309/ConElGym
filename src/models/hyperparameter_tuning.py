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
from tensorflow.keras.metrics import Precision, Recall
import subprocess
import sys
import json
import cv2
from tensorflow.keras.optimizers import Adam

def clear_memory():
    """Очистка памяти"""
    print("\n[DEBUG] ===== Начало очистки памяти =====")
    
    try:
    #    print("[DEBUG] 1. Очистка TensorFlow сессии...")
        # Очищаем все сессии TensorFlow
        tf.keras.backend.clear_session()
        # print("[DEBUG] ✓ TensorFlow сессия очищена")
        
        # print("[DEBUG] 2. Запуск garbage collector...")
        # Очистка Python garbage collector
        gc.collect()
        # print("[DEBUG] ✓ Garbage collector выполнен")
        
        # Очистка CUDA кэша если используется GPU
        if Config.DEVICE_CONFIG['use_gpu']:
            # print("[DEBUG] 3. Очистка GPU памяти...")
            try:
                # print("[DEBUG] 3.1. Сброс статистики памяти GPU...")
                # Пробуем очистить CUDA кэш через TensorFlow
                tf.config.experimental.reset_memory_stats('GPU:0')
                # print("[DEBUG] ✓ Статистика памяти GPU сброшена")
                
                # print("[DEBUG] 3.2. Очистка CUDA кэша...")
                # Принудительно очищаем CUDA кэш
                tf.keras.backend.clear_session()
                # print("[DEBUG] ✓ CUDA кэш очищен")
                
                # print("[DEBUG] 3.3. Очистка TensorFlow переменных...")
                # Очищаем все переменные
                for var in tf.compat.v1.global_variables():
                    del var
                # print("[DEBUG] ✓ TensorFlow переменные очищены")
                
                # print("[DEBUG] 3.4. Финальная очистка сессии...")
                # Очищаем все операции
                tf.keras.backend.clear_session()
                    # print("[DEBUG] ✓ Финальная очистка сессии выполнена")
                
            except Exception as e:
                print(f"[DEBUG] ✗ Ошибка при очистке GPU: {str(e)}")
        
        # print("[DEBUG] 4. Финальная очистка...")
        # # Дополнительная очистка
        # gc.collect()
        # print("[DEBUG] ✓ Финальная очистка выполнена")
        
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

def create_data_pipeline(batch_size, data_loader):
    """Создание pipeline данных"""
    print(f"\n[DEBUG] ===== Создание pipeline данных =====")
    print(f"[DEBUG] Параметры:")
    print(f"  - batch_size: {batch_size}")
    print(f"  - sequence_length: {Config.SEQUENCE_LENGTH}")
    print(f"  - input_size: {Config.INPUT_SIZE}")
    print(f"  - ожидаемая форма: ({Config.SEQUENCE_LENGTH}, {Config.INPUT_SIZE[0]}, {Config.INPUT_SIZE[1]}, 3)")
    
    print("[DEBUG] Запуск генератора данных...")
    try:
        # Создаем tf.data.Dataset из генератора
        dataset = tf.data.Dataset.from_generator(
            data_loader.data_generator,
            output_signature=(
                tf.TensorSpec(shape=(Config.SEQUENCE_LENGTH, *Config.INPUT_SIZE, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(Config.SEQUENCE_LENGTH, Config.NUM_CLASSES), dtype=tf.float32)
            )
        )
        
        print("[DEBUG] Применяем оптимизации к dataset...")
        dataset = dataset.cache()
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        print("[DEBUG] Pipeline данных успешно создан")
        return dataset
        
    except Exception as e:
        print(f"[ERROR] Ошибка при создании pipeline данных: {str(e)}")
        print("[DEBUG] Stack trace:", flush=True)
        traceback.print_exc()
        raise

def f1_score_element(y_true, y_pred):
    y_true = tf.argmax(y_true, axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1)
    true_positives = tf.reduce_sum(tf.cast((y_true == 1) & (y_pred == 1), tf.float32))
    predicted_positives = tf.reduce_sum(tf.cast(y_pred == 1, tf.float32))
    possible_positives = tf.reduce_sum(tf.cast(y_true == 1, tf.float32))
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
    return 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())

def create_and_compile_model(input_shape, num_classes, learning_rate, dropout_rate, lstm_units=None, model_type='v3'):
    """
    Создание и компиляция модели с заданными параметрами
    Args:
        input_shape: форма входных данных
        num_classes: количество классов
        learning_rate: скорость обучения
        dropout_rate: коэффициент dropout
        lstm_units: количество юнитов в LSTM слоях (только для v3)
        model_type: тип модели ('v3' или 'v4')
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
    
    # Формируем правильный input_shape с учетом длины последовательности
    full_input_shape = (Config.SEQUENCE_LENGTH,) + input_shape
    
    model, class_weights = create_model(
        input_shape=full_input_shape,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        lstm_units=lstm_units,
        model_type=model_type
    )
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Включаем mixed precision если используется GPU
    if Config.DEVICE_CONFIG['use_gpu'] and Config.MEMORY_OPTIMIZATION['use_mixed_precision']:
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            Precision(class_id=1, name='precision_element'),
            Recall(class_id=1, name='recall_element'),
            f1_score_element
        ]
    )
    
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
        train_dataset = create_data_pipeline(batch_size, train_loader)
        val_dataset = create_data_pipeline(batch_size, val_loader)
        
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
        print(f"\n[DEBUG] Начало триала {trial.number}")
        
        # Очищаем память перед каждым триалом
        clear_memory()
        
        # Определяем гиперпараметры для текущего триала
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        lstm_units = trial.suggest_int('lstm_units', 32, 128)
            
        print(f"[DEBUG] Параметры триала:")
        print(f"  - learning_rate: {learning_rate}")
        print(f"  - dropout_rate: {dropout_rate}")
        print(f"  - lstm_units: {lstm_units}")
        
        # Создаем и компилируем модель
        model, class_weights = create_and_compile_model(
            input_shape=Config.INPUT_SHAPE,
            num_classes=Config.NUM_CLASSES,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            lstm_units=lstm_units,
            model_type=Config.MODEL_TYPE
        )
        
        # Используем глобальный загрузчик данных
        global train_loader, val_loader
        
        # Создаем оптимизированные pipeline данных
        train_dataset = create_data_pipeline(
            batch_size=Config.BATCH_SIZE,
            data_loader=train_loader
        )
        
        val_dataset = create_data_pipeline(
            batch_size=Config.BATCH_SIZE,
            data_loader=val_loader
        )
        
        # Создаем callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_f1_score_element',
                patience=5,
                restore_best_weights=True,
                mode='max'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_f1_score_element',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                mode='max'
            )
        ]
        
        # Обучаем модель
        history = model.fit(
            train_dataset,
            epochs=Config.EPOCHS,
            steps_per_epoch=Config.STEPS_PER_EPOCH,
            validation_data=val_dataset,
            validation_steps=Config.VALIDATION_STEPS,
            callbacks=callbacks,
            class_weight=class_weights
        )
        
        # Получаем лучший результат
        best_val_f1 = max(history.history['val_f1_score_element'])
        
        print(f"[DEBUG] Триал {trial.number} завершен. Лучший val_f1: {best_val_f1:.4f}")
        
        return best_val_f1
        
    except Exception as e:
        print(f"[ERROR] Ошибка в триале {trial.number}: {str(e)}")
        print("[DEBUG] Stack trace:", flush=True)
        traceback.print_exc()
        raise optuna.TrialPruned()

def save_tuning_results(study, total_time, n_trials):
    """
    Сохранение результатов подбора гиперпараметров
    
    Args:
        study: Объект study Optuna
        total_time: Общее время выполнения
        n_trials: Количество испытаний
    """
    tuning_dir = os.path.join(Config.MODEL_SAVE_PATH, 'tuning')
    os.makedirs(tuning_dir, exist_ok=True)
    
    # Сохраняем результаты в файл
    with open(os.path.join(tuning_dir, 'tuning_results.txt'), 'w') as f:
        f.write(f"Best trial: {study.best_trial.number}\n")
        f.write(f"Best value: {study.best_trial.value:.4f}\n")
        f.write("\nBest parameters:\n")
        for param, value in study.best_trial.params.items():
            f.write(f"{param}: {value}\n")
        
        f.write(f"\nTotal trials: {n_trials}\n")
        f.write(f"Total time: {timedelta(seconds=int(total_time))}\n")
        f.write(f"Average time per trial: {timedelta(seconds=int(total_time/n_trials))}\n")
        
        # Добавляем статистику по типам моделей
        v3_trials = sum(1 for t in study.trials if t.params.get('model_type') == 'v3')
        v4_trials = sum(1 for t in study.trials if t.params.get('model_type') == 'v4')
        f.write(f"\nModel type distribution:\n")
        f.write(f"MobileNetV3 trials: {v3_trials}\n")
        f.write(f"MobileNetV4 trials: {v4_trials}\n")
        
        # Добавляем статистику успешных trials
        successful_trials = sum(1 for t in study.trials if t.value is not None)
        f.write(f"\nSuccessful trials: {successful_trials}/{n_trials}\n")
        
        # Добавляем лучшие результаты для каждого типа модели
        v3_best = max((t.value for t in study.trials if t.params.get('model_type') == 'v3' and t.value is not None), default=None)
        v4_best = max((t.value for t in study.trials if t.params.get('model_type') == 'v4' and t.value is not None), default=None)
        
        f.write("\nBest results by model type:\n")
        if v3_best is not None:
            f.write(f"MobileNetV3 best accuracy: {v3_best:.4f}\n")
        if v4_best is not None:
            f.write(f"MobileNetV4 best accuracy: {v4_best:.4f}\n")

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

def tune_hyperparameters():
    """
    Подбор гиперпараметров с использованием Optuna
    """
    try:
        print("\n[DEBUG] Начало подбора гиперпараметров...")
        
        # Создаем глобальные загрузчики данных
        global train_loader, val_loader
        train_loader = VideoDataLoader(Config.TRAIN_DATA_PATH)
        val_loader = VideoDataLoader(Config.VALID_DATA_PATH)
        
        # Создаем study
        study = optuna.create_study(
            direction='maximize',
            study_name='model_hyperparameter_tuning'
        )
        
        # Запускаем оптимизацию
        study.optimize(
            objective,
            n_trials=Config.HYPERPARAM_TUNING['n_trials'],
            timeout=Config.HYPERPARAM_TUNING['timeout'],
            n_jobs=Config.HYPERPARAM_TUNING['n_jobs']
        )
        
        # Сохраняем результаты
        save_tuning_results(study)
        
        print("[DEBUG] Подбор гиперпараметров завершен")
        return study
        
    except Exception as e:
        print(f"[ERROR] Ошибка при подборе гиперпараметров: {str(e)}")
        print("[DEBUG] Stack trace:", flush=True)
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