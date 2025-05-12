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
from tensorflow.keras.metrics import Precision, Recall
import subprocess
import sys
import json
import cv2

def clear_memory():
    """Очистка памяти"""
    print("\n[DEBUG] ===== Начало очистки памяти =====")
    
    try:
        print("[DEBUG] 1. Очистка TensorFlow сессии...")
        # Очищаем все сессии TensorFlow
        tf.keras.backend.clear_session()
        print("[DEBUG] ✓ TensorFlow сессия очищена")
        
        print("[DEBUG] 2. Запуск garbage collector...")
        # Очистка Python garbage collector
        gc.collect()
        print("[DEBUG] ✓ Garbage collector выполнен")
        
        # Очистка CUDA кэша если используется GPU
        if Config.DEVICE_CONFIG['use_gpu']:
            print("[DEBUG] 3. Очистка GPU памяти...")
            try:
                print("[DEBUG] 3.1. Сброс статистики памяти GPU...")
                # Пробуем очистить CUDA кэш через TensorFlow
                tf.config.experimental.reset_memory_stats('GPU:0')
                print("[DEBUG] ✓ Статистика памяти GPU сброшена")
                
                print("[DEBUG] 3.2. Очистка CUDA кэша...")
                # Принудительно очищаем CUDA кэш
                tf.keras.backend.clear_session()
                print("[DEBUG] ✓ CUDA кэш очищен")
                
                print("[DEBUG] 3.3. Очистка TensorFlow переменных...")
                # Очищаем все переменные
                for var in tf.keras.backend.get_session().graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                    del var
                print("[DEBUG] ✓ TensorFlow переменные очищены")
                
                print("[DEBUG] 3.4. Финальная очистка сессии...")
                # Очищаем все операции
                tf.keras.backend.clear_session()
                print("[DEBUG] ✓ Финальная очистка сессии выполнена")
                
            except Exception as e:
                print(f"[DEBUG] ✗ Ошибка при очистке GPU: {str(e)}")
        
        print("[DEBUG] 4. Финальная очистка...")
        # Дополнительная очистка
        gc.collect()
        print("[DEBUG] ✓ Финальная очистка выполнена")
        
    except Exception as e:
        print(f"[DEBUG] ✗ Критическая ошибка при очистке памяти: {str(e)}")
        print("[DEBUG] Stack trace:", flush=True)
        import traceback
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

def create_data_pipeline(loader, sequence_length, batch_size, target_size, one_hot, infinite_loop, max_sequences_per_video):
    """
    Создает оптимизированный pipeline данных
    """
    print(f"[DEBUG] Создание pipeline данных: batch_size={batch_size}")
    print(f"[DEBUG] Ожидаемая форма входных данных: (None, {Config.SEQUENCE_LENGTH}, {Config.INPUT_SIZE[0]}, {Config.INPUT_SIZE[1]}, 3)")
    
    def generator():
        print("[DEBUG] Запуск генератора данных...")
        try:
            # Очищаем память перед созданием генератора
            clear_memory()
            
            # Получаем список видео и аннотаций
            video_paths = loader.video_paths
            annotation_paths = loader.annotation_paths
            
            while True:  # Бесконечный цикл для infinite_loop
                for video_path, annotation_path in zip(video_paths, annotation_paths):
                    try:
                        print(f"[DEBUG] Обработка видео: {os.path.basename(video_path)}")
                        
                        # Загружаем аннотацию
                        with open(annotation_path, 'r') as f:
                            annotation = json.load(f)
                        start_frame = annotation['start_frame']
                        end_frame = annotation['end_frame']
                        
                        # Открываем видео
                        cap = cv2.VideoCapture(video_path)
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        
                        # Определяем границы для последовательностей
                        sequence_count = 0
                        while sequence_count < max_sequences_per_video:
                            # Выбираем случайную начальную позицию в пределах аннотации
                            if end_frame - start_frame <= sequence_length:
                                start_pos = start_frame
                            else:
                                start_pos = np.random.randint(start_frame, end_frame - sequence_length)
                            
                            # Загружаем только нужные кадры
                            frames = []
                            cap.set(cv2.CAP_PROP_POS_FRAMES, start_pos)
                            
                            for _ in range(sequence_length):
                                ret, frame = cap.read()
                                if not ret:
                                    break
                                frame = cv2.resize(frame, target_size)
                                frame = frame.astype(np.float32) / 255.0
                                frames.append(frame)
                            
                            if len(frames) == sequence_length:
                                # Создаем метки
                                labels = np.zeros((sequence_length, 2))
                                for i in range(sequence_length):
                                    frame_idx = start_pos + i
                                    if start_frame <= frame_idx <= end_frame:
                                        labels[i, 1] = 1  # Элемент присутствует
                                    else:
                                        labels[i, 0] = 1  # Элемент отсутствует
                                
                                # Преобразуем в one-hot если нужно
                                if one_hot:
                                    labels = tf.keras.utils.to_categorical(labels, num_classes=2)
                                
                                # Очищаем память после создания последовательности
                                frames_array = np.array(frames)
                                del frames
                                gc.collect()
                                
                                yield frames_array, labels
                                sequence_count += 1
                                
                                # Очищаем память после yield
                                del frames_array
                                del labels
                                gc.collect()
                        
                        # Закрываем видео
                        cap.release()
                        
                    except Exception as e:
                        print(f"[DEBUG] Ошибка при обработке видео {video_path}: {str(e)}")
                        continue
                
                if not infinite_loop:
                    break
            
        except Exception as e:
            print(f"[DEBUG] Ошибка в генераторе данных: {str(e)}")
            raise
        finally:
            # Очищаем память после завершения генератора
            clear_memory()
    
    try:
        # Очищаем память перед созданием dataset
        clear_memory()
        
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(Config.SEQUENCE_LENGTH, *Config.INPUT_SIZE, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(Config.SEQUENCE_LENGTH, 2), dtype=tf.float32)
            )
        )
        print("[DEBUG] tf.data.Dataset создан успешно")
        
        # Оптимизация загрузки данных
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(Config.MEMORY_OPTIMIZATION['prefetch_buffer_size'])
        print("[DEBUG] Pipeline данных оптимизирован")
        
        return dataset
    except Exception as e:
        print(f"[DEBUG] Ошибка при создании pipeline данных: {str(e)}")
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
    print("[DEBUG] Начало загрузки данных...")
    clear_memory()  # Очищаем память перед загрузкой данных
    
    try:
        train_loader = VideoDataLoader(Config.TRAIN_DATA_PATH)
        val_loader = VideoDataLoader(Config.VALID_DATA_PATH)
        print("[DEBUG] VideoDataLoader создан успешно")
        
        target_size = Config.INPUT_SIZE
        
        # Создание оптимизированных pipeline данных
        train_dataset = create_data_pipeline(
            loader=train_loader,
            sequence_length=Config.SEQUENCE_LENGTH,
            batch_size=batch_size,
            target_size=target_size,
            one_hot=True,
            infinite_loop=True,
            max_sequences_per_video=10
        )
        print("[DEBUG] Train dataset создан успешно")
        
        # Очищаем память между созданием train и val datasets
        clear_memory()
        
        val_dataset = create_data_pipeline(
            loader=val_loader,
            sequence_length=Config.SEQUENCE_LENGTH,
            batch_size=batch_size,
            target_size=target_size,
            one_hot=True,
            infinite_loop=False,
            max_sequences_per_video=10
        )
        print("[DEBUG] Val dataset создан успешно")
        
        return train_dataset, val_dataset
    except Exception as e:
        print(f"[DEBUG] Ошибка при загрузке данных: {str(e)}")
        clear_memory()
        raise

def objective(trial):
    print(f"\n[DEBUG] ===== Начало trial {trial.number} =====")
    # Очищаем память перед каждым испытанием
    clear_memory()
    
    # Определяем пространство поиска
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
        'lstm_units': trial.suggest_int('lstm_units', 32, 64),
        'batch_size': trial.suggest_int('batch_size', 8, 32)
    }
    
    print(f"\n[DEBUG] Trial {trial.number} parameters:")
    for param, value in params.items():
        print(f"  - {param}: {value}")
    
    try:
        # Загружаем данные
        print("\n[DEBUG] 1. Загрузка данных...")
        train_dataset, val_dataset = load_and_prepare_data(params['batch_size'])
        print("[DEBUG] ✓ Данные загружены успешно")
        
        # Создаем и компилируем модель
        print("\n[DEBUG] 2. Создание и компиляция модели...")
        input_shape = (Config.SEQUENCE_LENGTH, *Config.INPUT_SIZE, 3)
        model = create_and_compile_model(
            input_shape=input_shape,
            num_classes=Config.NUM_CLASSES,
            learning_rate=params['learning_rate'],
            dropout_rate=params['dropout_rate'],
            lstm_units=params['lstm_units']
        )
        print("[DEBUG] ✓ Модель создана и скомпилирована")
        
        # Используем раннюю остановку
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True
        )
        
        # Обучаем модель
        print("\n[DEBUG] 3. Начало обучения модели...")
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=min(Config.EPOCHS, 10),
            steps_per_epoch=min(Config.STEPS_PER_EPOCH, 50),
            validation_steps=min(Config.VALIDATION_STEPS, 20),
            callbacks=[early_stopping],
            verbose=0
        )
        print("[DEBUG] ✓ Обучение завершено")
        
        # Получаем лучшую точность на валидационном наборе
        best_val_accuracy = max(history.history['val_accuracy'])
        print(f"[DEBUG] ✓ Лучшая точность на валидации: {best_val_accuracy:.4f}")
        
        # Очищаем память
        print("\n[DEBUG] 4. Очистка после обучения...")
        print("[DEBUG] 4.1. Удаление модели...")
        del model
        print("[DEBUG] 4.2. Удаление истории обучения...")
        del history
        print("[DEBUG] 4.3. Удаление датасетов...")
        del train_dataset
        del val_dataset
        print("[DEBUG] 4.4. Финальная очистка памяти...")
        clear_memory()
        print("[DEBUG] ✓ Очистка завершена")
        
        print(f"\n[DEBUG] ===== Trial {trial.number} успешно завершен =====")
        return best_val_accuracy
        
    except Exception as e:
        print(f"\n[DEBUG] ✗ Ошибка в trial {trial.number}: {str(e)}")
        print("[DEBUG] Stack trace:", flush=True)
        import traceback
        traceback.print_exc()
        
        # Очищаем память в случае ошибки
        print("\n[DEBUG] Очистка памяти после ошибки...")
        try:
            if 'model' in locals():
                del model
            if 'history' in locals():
                del history
            if 'train_dataset' in locals():
                del train_dataset
            if 'val_dataset' in locals():
                del val_dataset
            clear_memory()
        except Exception as cleanup_error:
            print(f"[DEBUG] ✗ Ошибка при очистке после ошибки: {str(cleanup_error)}")
        
        # Возвращаем очень плохое значение вместо None
        return float('-inf')

def save_tuning_results(study, n_trials):
    """
    Сохранение результатов подбора гиперпараметров
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
    print("[DEBUG] Начало подбора гиперпараметров")
    
    if not device_available:
        print("Warning: Device setup failed. This will be very slow.")
    
    # Создание директории для сохранения результатов
    os.makedirs(os.path.join(Config.MODEL_SAVE_PATH, 'tuning'), exist_ok=True)
    
    # Очищаем память перед началом оптимизации
    clear_memory()
    
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
    
    try:
        study.optimize(objective, n_trials=n_trials)
    except Exception as e:
        print(f"\nError during optimization: {str(e)}")
        print(f"[DEBUG] Stack trace:", flush=True)
        import traceback
        traceback.print_exc()
        clear_memory()
    finally:
        clear_memory()
    
    total_time = time.time() - start_time
    
    # Проверяем, есть ли успешные trials
    successful_trials = [t for t in study.trials if t.value is not None and t.value != float('-inf')]
    if not successful_trials:
        print("\nNo successful trials completed. Check the error messages above.")
        return {
            'study': study,
            'best_params': None,
            'best_value': float('-inf')
        }
    
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