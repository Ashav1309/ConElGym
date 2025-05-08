import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Фильтрация логов TensorFlow
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Используем только первую GPU

import tensorflow as tf
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

def setup_gpu():
    """Настройка GPU с ограничением памяти"""
    try:
        # Очищаем все сессии TensorFlow
        tf.keras.backend.clear_session()
        
        # Получаем список доступных GPU
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            print("No GPU devices found")
            return False
            
        # Настраиваем каждую GPU
        for gpu in gpus:
            # Включаем динамический рост памяти
            tf.config.experimental.set_memory_growth(gpu, True)
            
        print("GPU memory growth enabled")
        return True
        
    except RuntimeError as e:
        print(f"Error setting up GPU: {e}")
        return False

# Инициализация GPU
gpu_available = setup_gpu()

# Включение mixed precision только если GPU доступна
if gpu_available:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("Mixed precision enabled")
else:
    print("Running on CPU")

def clear_memory():
    """Очистка памяти GPU и Python"""
    # Очищаем все сессии TensorFlow
    tf.keras.backend.clear_session()
    
    # Принудительная очистка кэша CUDA
    if gpu_available:
        try:
            import torch
            torch.cuda.empty_cache()
        except:
            pass
    
    # Очистка Python garbage collector
    gc.collect()
    
    # Принудительная очистка памяти
    if gpu_available:
        try:
            tf.config.experimental.reset_memory_stats('GPU:0')
        except:
            pass

def create_data_pipeline(generator, batch_size):
    """
    Создает оптимизированный pipeline данных с использованием tf.data.Dataset
    """
    dataset = tf.data.Dataset.from_generator(
        lambda: generator,
        output_signature=(
            tf.TensorSpec(shape=(None, Config.SEQUENCE_LENGTH, 112, 112, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, Config.NUM_CLASSES), dtype=tf.float32)
        )
    )
    
    # Оптимизация загрузки данных
    dataset = dataset.cache()  # Кэширование данных
    dataset = dataset.prefetch(tf.data.AUTOTUNE)  # Предзагрузка следующих батчей
    dataset = dataset.batch(batch_size)  # Группировка в батчи
    
    # Исправление размерности данных
    def reshape_data(x, y):
        x = tf.squeeze(x, axis=1)
        y = tf.squeeze(y, axis=1)
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
    if gpu_available:
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    
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
    
    train_generator = train_loader.load_data(
        Config.SEQUENCE_LENGTH, 
        batch_size, 
        target_size=(112, 112),
        one_hot=True
    )
    val_generator = val_loader.load_data(
        Config.SEQUENCE_LENGTH, 
        batch_size, 
        target_size=(112, 112),
        one_hot=True
    )
    
    train_dataset = create_data_pipeline(train_generator, batch_size)
    val_dataset = create_data_pipeline(val_generator, batch_size)
    
    return train_dataset, val_dataset

def objective(trial):
    """
    Функция для оптимизации гиперпараметров.
    """
    print(f"\nTrial {trial.number + 1} started")
    
    # Определение гиперпараметров для оптимизации
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.3, 0.7)
    lstm_units = trial.suggest_categorical('lstm_units', [32, 64])
    
    print(f"Parameters: learning_rate={learning_rate:.6f}, dropout_rate={dropout_rate:.2f}, lstm_units={lstm_units}")
    
    try:
        # Очищаем память перед каждым trial
        clear_memory()
        
        # Создание и компиляция модели
        input_shape = (Config.SEQUENCE_LENGTH, 112, 112, 3)
        model = create_and_compile_model(
            input_shape=input_shape,
            num_classes=Config.NUM_CLASSES,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            lstm_units=lstm_units
        )
        
        # Загрузка и подготовка данных с меньшим размером батча
        batch_size = 2  # Уменьшаем размер батча
        train_dataset, val_dataset = load_and_prepare_data(batch_size)
        
        # Обучение модели с меньшим количеством шагов
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=30,  # Уменьшаем количество эпох
            steps_per_epoch=20,  # Уменьшаем количество шагов
            validation_steps=5,  # Уменьшаем количество шагов валидации
            verbose=1,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=3,  # Уменьшаем patience
                    restore_best_weights=True
                )
            ]
        )
        
        # Проверяем, что модель действительно обучилась
        if not history.history['val_accuracy']:
            print("Model training failed - no validation accuracy recorded")
            return None
            
        best_val_accuracy = max(history.history['val_accuracy'])
        print(f"Trial {trial.number + 1} finished with validation accuracy: {best_val_accuracy:.4f}")
        
        # Очищаем память после обучения
        del model
        clear_memory()
        
        return best_val_accuracy
        
    except tf.errors.ResourceExhaustedError:
        print("GPU memory exhausted. Skipping trial.")
        clear_memory()
        return None
    except Exception as e:
        print(f"Error in trial: {str(e)}")
        clear_memory()
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
    Подбор оптимальных гиперпараметров модели с помощью Optuna.
    """
    if not gpu_available:
        print("Warning: Running without GPU. This will be very slow.")
    
    # Создание директории для сохранения результатов
    os.makedirs(os.path.join(Config.MODEL_SAVE_PATH, 'tuning'), exist_ok=True)
    
    # Создание study с оптимизированными настройками
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(n_startup_trials=3),  # Уменьшаем количество начальных trials
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=3,  # Уменьшаем количество начальных trials
            n_warmup_steps=3,    # Уменьшаем количество шагов разогрева
            interval_steps=1
        )
    )
    
    # Запуск оптимизации
    start_time = time.time()
    n_trials = 5  # Уменьшаем количество trials
    
    try:
        study.optimize(objective, n_trials=n_trials)
        
        # Проверяем, есть ли успешные trials
        if not any(t.value is not None for t in study.trials):
            print("No successful trials completed. Please check GPU memory and try again.")
            return None
        
        # Сохранение результатов
        total_time = time.time() - start_time
        save_tuning_results(study, total_time, n_trials)
        
        # Визуализация результатов
        plot_tuning_results(study)
        
        print("\nHyperparameter tuning completed!")
        print(f"Best trial value: {study.best_trial.value}")
        print(f"Best parameters: {study.best_trial.params}")
        
        return study.best_trial.params
        
    except Exception as e:
        print(f"Error during hyperparameter tuning: {str(e)}")
        return None

if __name__ == "__main__":
    best_params = tune_hyperparameters()
    print("Best parameters:", best_params) 