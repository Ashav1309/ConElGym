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
from tqdm.auto import tqdm
from tqdm.keras import TqdmCallback
from tensorflow.keras.metrics import Precision, Recall
import subprocess
import sys
from src.data.data_loader import load_data

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
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(Config.MEMORY_OPTIMIZATION['prefetch_buffer_size'])
    
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
    
    print(f"[DEBUG] Creating model with parameters:")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Dropout rate: {dropout_rate}")
    print(f"  - LSTM units: {lstm_units}")
    print(f"  - Input shape: {input_shape}")
    print(f"  - Number of classes: {num_classes}")
    
    model = create_model(
        input_shape=input_shape,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        lstm_units=lstm_units
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
        infinite_loop=True,  # Для обучения используем бесконечный цикл
        max_sequences_per_video=10
    )
    
    val_dataset = create_data_pipeline(
        loader=val_loader,
        sequence_length=Config.SEQUENCE_LENGTH,
        batch_size=batch_size,
        target_size=target_size,
        one_hot=True,
        infinite_loop=False,  # Для валидации не используем бесконечный цикл
        max_sequences_per_video=10
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
    # Очищаем память перед каждым испытанием
    clear_memory()
    
    # Определяем пространство поиска
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
        'lstm_units': trial.suggest_int('lstm_units', 32, 128),
        'batch_size': trial.suggest_int('batch_size', 16, 64)
    }
    
    print(f"\n[DEBUG] Trial {trial.number} parameters:")
    for param, value in params.items():
        print(f"  - {param}: {value}")
    
    try:
        # Загружаем данные
        print("\n[DEBUG] Loading data...")
        train_dataset, val_dataset = load_and_prepare_data(params['batch_size'])
        print("[DEBUG] Data loaded successfully")
        
        # Создаем и компилируем модель
        print("\n[DEBUG] Creating and compiling model...")
        input_shape = (Config.SEQUENCE_LENGTH, *Config.INPUT_SIZE, 3)
        model = create_and_compile_model(
            input_shape=input_shape,
            num_classes=Config.NUM_CLASSES,
            learning_rate=params['learning_rate'],
            dropout_rate=params['dropout_rate'],
            lstm_units=params['lstm_units']
        )
        print("[DEBUG] Model created and compiled successfully")
        
        # Используем раннюю остановку
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        )
        
        # Обучаем модель
        print("\n[DEBUG] Starting model training...")
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=Config.EPOCHS,
            steps_per_epoch=Config.STEPS_PER_EPOCH,
            validation_steps=Config.VALIDATION_STEPS,
            callbacks=[early_stopping],
            verbose=0
        )
        print("[DEBUG] Model training completed")
        
        # Получаем лучшую точность на валидационном наборе
        best_val_accuracy = max(history.history['val_accuracy'])
        print(f"[DEBUG] Best validation accuracy: {best_val_accuracy:.4f}")
        
        # Очищаем память
        clear_memory()
        
        return best_val_accuracy
        
    except Exception as e:
        print(f"\n[ERROR] Error in trial {trial.number}: {str(e)}")
        return None

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
            
            # Безопасное получение лучшего значения
            try:
                best_value = study.best_value if study.best_value is not None else float('-inf')
                best_value_str = f"{best_value:.4f}" if best_value != float('-inf') else "N/A"
            except:
                best_value_str = "N/A"
            
            pbar.set_postfix({
                'trial': trial.number + 1,
                'best_val_acc': best_value_str
            })
            
            return result
        except Exception as e:
            pbar.update(1)
            print(f"\nError in trial {trial.number + 1}: {str(e)}")
            return float('-inf')
    
    try:
        study.optimize(objective_with_progress, n_trials=n_trials)
    except Exception as e:
        print(f"\nError during optimization: {str(e)}")
    finally:
        pbar.close()
    
    total_time = time.time() - start_time
    
    # Проверяем, есть ли успешные trials
    successful_trials = [t for t in study.trials if t.value is not None and t.value != float('-inf')]
    if not successful_trials:
        print("\nNo successful trials completed. Check the error messages above.")
        return None
    
    # Сохранение результатов
    save_tuning_results(study, total_time, n_trials)
    
    # Визуализация результатов
    try:
        plot_tuning_results(study)
    except Exception as e:
        print(f"\nWarning: Could not create visualization plots: {str(e)}")
    
    # Возвращаем и study, и лучшие параметры
    return {
        'study': study,
        'best_params': study.best_params,
        'best_value': study.best_value
    }

if __name__ == "__main__":
    try:
        result = tune_hyperparameters()
        if result is not None:
            print("\nBest parameters:", result['best_params'])
            print("Best validation accuracy:", result['best_value'])
        else:
            print("\nFailed to find best parameters. Check the error messages above.")
    except Exception as e:
        print(f"\nError during hyperparameter tuning: {str(e)}") 