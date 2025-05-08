import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Фильтрация логов TensorFlow

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

# Настройка GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            # Устанавливаем лимит памяти GPU
            tf.config.set_logical_device_configuration(
                gpu,
                [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]  # 4GB
            )
        print("Memory growth enabled for GPUs")
    except RuntimeError as e:
        print(f"Error setting memory growth: {e}")

# Включение mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("GPU Device: ", tf.test.gpu_device_name())

def clear_memory():
    """Очистка памяти GPU и Python"""
    tf.keras.backend.clear_session()
    gc.collect()

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
        # Создание и компиляция модели
        input_shape = (Config.SEQUENCE_LENGTH, 112, 112, 3)
        model = create_and_compile_model(
            input_shape=input_shape,
            num_classes=Config.NUM_CLASSES,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            lstm_units=lstm_units
        )
        
        # Загрузка и подготовка данных
        batch_size = 16  # Уменьшаем размер батча
        train_dataset, val_dataset = load_and_prepare_data(batch_size)
        
        # Обучение модели
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=10,
            steps_per_epoch=10,
            validation_steps=3,
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
        return float('-inf')  # Возвращаем отрицательную бесконечность для пропуска trial
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
        f.write(f"  Value: {study.best_trial.value}\n")
        f.write(f"  Params: {study.best_trial.params}\n\n")
        f.write("All trials:\n")
        for trial in study.trials:
            f.write(f"Trial {trial.number}:\n")
            f.write(f"  Value: {trial.value}\n")
            f.write(f"  Params: {trial.params}\n\n")
        
        f.write(f"\nTotal time: {timedelta(seconds=int(total_time))}\n")
        f.write(f"Average time per trial: {timedelta(seconds=int(total_time/n_trials))}\n")

def plot_tuning_results(study):
    """
    Визуализация результатов подбора гиперпараметров
    """
    tuning_dir = os.path.join(Config.MODEL_SAVE_PATH, 'tuning')
    
    # График истории оптимизации
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image(os.path.join(tuning_dir, 'optimization_history.png'))
    
    # График важности параметров
    fig = optuna.visualization.plot_param_importances(study)
    fig.write_image(os.path.join(tuning_dir, 'param_importances.png'))

def tune_hyperparameters():
    """
    Подбор оптимальных гиперпараметров модели с помощью Optuna.
    """
    # Создание директории для сохранения результатов
    os.makedirs(os.path.join(Config.MODEL_SAVE_PATH, 'tuning'), exist_ok=True)
    
    # Создание study с оптимизированными настройками
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(n_startup_trials=3),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=3,
            n_warmup_steps=3,
            interval_steps=1
        )
    )
    
    # Запуск оптимизации
    start_time = time.time()
    n_trials = 5
    
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