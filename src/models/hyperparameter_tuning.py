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

def create_data_pipeline(generator, batch_size):
    """
    Создает оптимизированный pipeline данных
    """
    print(f"Creating data pipeline with batch_size={batch_size}")
    print(f"Expected input shape: (None, {Config.SEQUENCE_LENGTH}, {Config.INPUT_SIZE[0]}, {Config.INPUT_SIZE[1]}, 3)")
    
    dataset = tf.data.Dataset.from_generator(
        lambda: generator,
        output_signature=(
            tf.TensorSpec(shape=(None, Config.SEQUENCE_LENGTH, *Config.INPUT_SIZE, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, Config.NUM_CLASSES), dtype=tf.float32)
        )
    )
    
    # Оптимизация загрузки данных
    if Config.MEMORY_OPTIMIZATION['cache_dataset']:
        dataset = dataset.cache()
    
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    
    # Исправление размерности данных
    def reshape_data(x, y):
        print(f"Input shape before reshape: {x.shape}")  # Отладочная информация
        x = tf.squeeze(x, axis=1)
        y = tf.squeeze(y, axis=1)
        print(f"Input shape after reshape: {x.shape}")  # Отладочная информация
        return x, y
    
    dataset = dataset.map(reshape_data, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset

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
        metrics=['accuracy']
    )
    
    return model

def load_and_prepare_data(batch_size):
    """
    Загрузка и подготовка данных для обучения
    """
    train_loader = VideoDataLoader(Config.TRAIN_DATA_PATH)
    val_loader = VideoDataLoader(Config.VALID_DATA_PATH)
    
    # Проверяем и устанавливаем правильный размер изображения
    target_size = Config.INPUT_SIZE
    print(f"Using target size: {target_size}")
    
    train_generator = train_loader.load_data(
        Config.SEQUENCE_LENGTH, 
        batch_size, 
        target_size=target_size,
        one_hot=True,
        infinite_loop=True
    )
    val_generator = val_loader.load_data(
        Config.SEQUENCE_LENGTH, 
        batch_size, 
        target_size=target_size,
        one_hot=True,
        infinite_loop=True
    )
    
    # Проверяем размеры данных
    sample_batch = next(train_generator)
    print(f"Sample batch shape: {sample_batch[0].shape}")
    print(f"Expected shape: (None, {Config.SEQUENCE_LENGTH}, {Config.INPUT_SIZE[0]}, {Config.INPUT_SIZE[1]}, 3)")
    
    if sample_batch[0].shape[2:] != (*Config.INPUT_SIZE, 3):
        raise ValueError(f"Неверный размер изображения. Получено: {sample_batch[0].shape[2:]}, ожидалось: {(*Config.INPUT_SIZE, 3)}")
    
    train_dataset = create_data_pipeline(train_generator, batch_size)
    val_dataset = create_data_pipeline(val_generator, batch_size)
    
    return train_dataset, val_dataset

def objective(trial):
    """
    Функция для оптимизации гиперпараметров
    """
    print(f"\nTrial {trial.number + 1} started")
    
    try:
        # Проверка доступной памяти GPU
        if Config.DEVICE_CONFIG['use_gpu']:
            try:
                from tensorflow.python.client import device_lib
                stats = tf.config.experimental.get_memory_info('GPU:0')
                if stats['current'] / 1024**3 > Config.DEVICE_CONFIG['gpu_memory_limit'] / 1024:
                    print("Not enough GPU memory. Skipping trial.")
                    return float('-inf')
            except:
                pass
        
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
        
        # Очищаем память перед созданием модели
        clear_memory()
        
        # Создание и компиляция модели
        input_shape = (Config.SEQUENCE_LENGTH, *Config.INPUT_SIZE, 3)
        print(f"Model input shape: {input_shape}")
        
        model = create_and_compile_model(
            input_shape=input_shape,
            num_classes=Config.NUM_CLASSES,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            lstm_units=lstm_units
        )
        
        # Загрузка и подготовка данных
        train_dataset, val_dataset = load_and_prepare_data(Config.BATCH_SIZE)
        
        # Добавляем callbacks для раннего прерывания
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=3,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.5,
                patience=2,
                min_lr=1e-6
            )
        ]
        
        # Обучение модели
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=Config.EPOCHS,
            steps_per_epoch=Config.STEPS_PER_EPOCH,
            validation_steps=Config.VALIDATION_STEPS,
            callbacks=callbacks,
            verbose=1
        )
        
        best_val_accuracy = max(history.history['val_accuracy'])
        print(f"Trial {trial.number + 1} finished with validation accuracy: {best_val_accuracy:.4f}")
        
        # Очищаем память после обучения
        del model
        clear_memory()
        
        return best_val_accuracy
        
    except tf.errors.ResourceExhaustedError:
        print("GPU memory exhausted. Skipping trial.")
        clear_memory()
        return float('-inf')
    except Exception as e:
        print(f"Error in trial: {str(e)}")
        clear_memory()
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
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    
    total_time = time.time() - start_time
    
    print(f"\nHyperparameter tuning completed!")
    print(f"Total time: {timedelta(seconds=int(total_time))}")
    print(f"Average time per trial: {timedelta(seconds=int(total_time/n_trials))}")
    
    # Сохранение и визуализация результатов
    save_tuning_results(study, total_time, n_trials)
    plot_tuning_results(study)
    
    return study.best_trial.params

if __name__ == "__main__":
    best_params = tune_hyperparameters()
    print("Best parameters:", best_params) 