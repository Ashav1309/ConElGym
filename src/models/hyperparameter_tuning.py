import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Фильтрация логов TensorFlow
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Используем первую GPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda'
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
from tqdm.auto import tqdm
from tqdm.keras import TqdmCallback
from tensorflow.keras.metrics import Precision, Recall
import subprocess
import sys

def clear_memory():
    """Очистка памяти"""
    # Очищаем все сессии TensorFlow
    tf.keras.backend.clear_session()
    
    # Очистка Python garbage collector
    gc.collect()
    
    # Очистка CUDA кэша если используется GPU
    if Config.DEVICE_CONFIG['use_gpu']:
        try:
            # Пробуем очистить CUDA кэш через TensorFlow
            tf.config.experimental.reset_memory_stats('GPU:0')
        except:
            pass

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

def create_data_pipeline(loader, sequence_length, batch_size, target_size, one_hot, infinite_loop, max_sequences_per_video):
    """
    Создает оптимизированный pipeline данных
    """
    print(f"Creating data pipeline with batch_size={batch_size}")
    print(f"Expected input shape: (None, {Config.SEQUENCE_LENGTH}, {Config.INPUT_SIZE[0]}, {Config.INPUT_SIZE[1]}, 3)")
    
    def generator():
        return loader.data_generator(
            sequence_length=sequence_length,
            batch_size=batch_size,
            target_size=target_size,
            one_hot=one_hot,
            infinite_loop=infinite_loop,
            max_sequences_per_video=max_sequences_per_video
        )
    
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(Config.SEQUENCE_LENGTH, *Config.INPUT_SIZE, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(Config.SEQUENCE_LENGTH, 2), dtype=tf.float32)
        )
    )
    
    # Оптимизация загрузки данных
    if Config.MEMORY_OPTIMIZATION['cache_dataset']:
        dataset = dataset.cache()
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    
    return dataset

def f1_score_element(y_true, y_pred):
    y_true = tf.argmax(y_true, axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1)
    true_positives = tf.reduce_sum(tf.cast((y_true == 1) & (y_pred == 1), tf.float32))
    predicted_positives = tf.reduce_sum(tf.cast(y_pred == 1, tf.float32))
    possible_positives = tf.reduce_sum(tf.cast(y_true == 1, tf.float32))
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
    return 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())

def create_and_compile_model(input_shape, num_classes, learning_rate, dropout_rate, lstm_units):
    """
    Создание и компиляция модели с заданными параметрами
    """
    clear_memory()  # Очищаем память перед созданием модели
    
    model = create_model(
        input_shape=input_shape,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        lstm_units=lstm_units
    )
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
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
    
    return model

def load_and_prepare_data(batch_size):
    """
    Загрузка и подготовка данных для обучения
    """
    train_loader = VideoDataLoader(Config.TRAIN_DATA_PATH)
    val_loader = VideoDataLoader(Config.VALID_DATA_PATH)
    
    target_size = Config.INPUT_SIZE
    
    # Создание оптимизированных pipeline данных
    train_dataset = create_data_pipeline(
        loader=train_loader,
        sequence_length=Config.SEQUENCE_LENGTH,
        batch_size=batch_size,
        target_size=target_size,
        one_hot=True,
        infinite_loop=True,
        max_sequences_per_video=100
    )
    
    val_dataset = create_data_pipeline(
        loader=val_loader,
        sequence_length=Config.SEQUENCE_LENGTH,
        batch_size=batch_size,
        target_size=target_size,
        one_hot=True,
        infinite_loop=True,
        max_sequences_per_video=100
    )
    
    # Проверяем размеры данных
    sample_batch = next(iter(train_dataset))
    print(f"[DEBUG] sample_batch type: {type(sample_batch)}")
    print(f"[DEBUG] sample_batch[0] shape: {getattr(sample_batch[0], 'shape', 'None')}")
    print(f"[DEBUG] sample_batch[1] shape: {getattr(sample_batch[1], 'shape', 'None')}")
    print(f"Expected shape: (None, {Config.SEQUENCE_LENGTH}, {Config.INPUT_SIZE[0]}, {Config.INPUT_SIZE[1]}, 3)")
    
    if sample_batch[0].shape[2:] != (*Config.INPUT_SIZE, 3):
        raise ValueError(f"Неверный размер изображения. Получено: {sample_batch[0].shape[2:]}, ожидалось: {(*Config.INPUT_SIZE, 3)}")
    
    return train_dataset, val_dataset

def objective(trial):
    """
    Функция для оптимизации гиперпараметров (запуск в отдельном процессе)
    """
    print(f"\nTrial {trial.number + 1} started")
    
    # Определение гиперпараметров для оптимизации
    learning_rate = trial.suggest_float(
        'learning_rate',
        *Config.HYPERPARAM_TUNING['learning_rate_range'],
        log=True
    )
    dropout_rate = trial.suggest_float(
        'dropout_rate',
        *Config.HYPERPARAM_TUNING['dropout_range']
    )
    lstm_units = trial.suggest_categorical(
        'lstm_units',
        Config.HYPERPARAM_TUNING['lstm_units']
    )
    
    print(f"Parameters: learning_rate={learning_rate:.6f}, dropout_rate={dropout_rate:.2f}, lstm_units={lstm_units}")
    
    # Формируем команду для запуска отдельного процесса
    cmd = [
        sys.executable, 'run_single_trial.py',
        '--learning_rate', str(learning_rate),
        '--dropout_rate', str(dropout_rate),
        '--lstm_units', str(lstm_units),
        '--batch_size', str(Config.BATCH_SIZE),
        '--sequence_length', str(Config.SEQUENCE_LENGTH),
        '--max_sequences_per_video', '100'
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout.strip()
        print(f"[DEBUG] Output from run_single_trial.py: {output}")
        best_val_accuracy = float(output.splitlines()[-1])
        print(f"[DEBUG] Лучшая val_accuracy: {best_val_accuracy}")
        print(f"Trial {trial.number + 1} finished with validation accuracy: {best_val_accuracy:.4f}")
        return best_val_accuracy
    except subprocess.CalledProcessError as e:
        print(f"[DEBUG] Ошибка при запуске run_single_trial.py: {e}")
        print(f"[DEBUG] stdout: {e.stdout}")
        print(f"[DEBUG] stderr: {e.stderr}")
        return float('-inf')
    except Exception as e:
        print(f"[DEBUG] Unexpected error: {e}")
        return float('-inf')

def save_tuning_results(study, total_time, n_trials):
    """
    Сохранение результатов подбора гиперпараметров
    """
    results_path = os.path.join(Config.MODEL_SAVE_PATH, 'tuning', 'optuna_results.txt')
    with open(results_path, 'w') as f:
        f.write(f"Best trial:\n")
        if study.best_trial.value is not None:
            f.write(f"  Value: {study.best_trial.value:.4f}\n")
        else:
            f.write(f"  Value: Failed\n")
        f.write(f"  Params: {study.best_trial.params}\n\n")
        
        f.write("All trials:\n")
        for trial in study.trials:
            f.write(f"Trial {trial.number}:\n")
            if trial.value is not None:
                f.write(f"  Value: {trial.value:.4f}\n")
            else:
                f.write(f"  Value: Failed\n")
            f.write(f"  Params: {trial.params}\n\n")
        
        f.write(f"\nTotal time: {timedelta(seconds=int(total_time))}\n")
        f.write(f"Average time per trial: {timedelta(seconds=int(total_time/n_trials))}\n")
        
        # Добавляем статистику успешных trials
        successful_trials = sum(1 for t in study.trials if t.value is not None)
        f.write(f"\nSuccessful trials: {successful_trials}/{n_trials}\n")

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
    Подбор оптимальных гиперпараметров модели
    """
    if not device_available:
        print("Warning: Device setup failed. This will be very slow.")
    
    # Создание директории для сохранения результатов
    os.makedirs(os.path.join(Config.MODEL_SAVE_PATH, 'tuning'), exist_ok=True)
    
    # Создание study с оптимизированными настройками
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(n_startup_trials=5),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=5,
            interval_steps=1
        )
    )
    
    # Запуск оптимизации
    start_time = time.time()
    n_trials = Config.HYPERPARAM_TUNING['n_trials']
    
    print(f"\nStarting hyperparameter tuning with {n_trials} trials...")
    
    # Создаем прогресс-бар для trials
    pbar = tqdm(total=n_trials, desc="Hyperparameter Tuning", position=0)
    
    def objective_with_progress(trial):
        try:
            result = objective(trial)
            pbar.update(1)
            pbar.set_postfix({
                'trial': trial.number + 1
            })
            return result
        except Exception as e:
            pbar.update(1)
            raise e
    
    try:
        study.optimize(objective_with_progress, n_trials=n_trials, n_jobs=1)
    finally:
        pbar.close()
    
    total_time = time.time() - start_time
    
    # Сохранение результатов
    save_tuning_results(study, total_time, n_trials)
    
    # Визуализация результатов
    plot_tuning_results(study)
    
    return study

if __name__ == "__main__":
    best_params = tune_hyperparameters()
    print("Best parameters:", best_params) 