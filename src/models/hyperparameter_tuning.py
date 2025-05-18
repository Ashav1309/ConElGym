import os
import time
import json
import traceback
import numpy as np
from datetime import timedelta
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import optuna
from src.models.model import (
    create_model_with_params,
    create_mobilenetv3_model,
    create_mobilenetv4_model,
    f1_score_element
)
from src.data_proc.data_loader import VideoDataLoader
from src.config import Config
from src.models.losses import focal_loss
from src.models.metrics import get_tuning_metrics
from src.models.callbacks import get_tuning_callbacks
from src.utils.gpu_config import setup_gpu

# Настройка GPU
setup_gpu()

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import gc
import subprocess
import sys
import cv2
from optuna.trial import Trial
from src.utils.network_handler import NetworkErrorHandler, NetworkMonitor
from src.data_proc.data_validation import validate_data_pipeline, validate_training_data
from src.models.train import create_data_pipeline, create_tuning_data_pipeline  # Импортируем общую функцию и новую функцию

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
        
        # Уменьшаем max_videos для более частой смены видео
        max_videos = min(Config.MAX_VIDEOS, 3)  # Ограничиваем количество видео для лучшего баланса
        print(f"[DEBUG] Используем max_videos={max_videos} для лучшего баланса классов")
        
        train_loader = VideoDataLoader(Config.TRAIN_DATA_PATH, max_videos=max_videos)
        val_loader = VideoDataLoader(Config.VALID_DATA_PATH, max_videos=max_videos)
        print("[DEBUG] VideoDataLoader создан успешно")
        
        target_size = Config.INPUT_SIZE
        
        # Создание оптимизированных pipeline данных с балансировкой классов
        train_dataset = create_tuning_data_pipeline(
            train_loader, 
            Config.SEQUENCE_LENGTH, 
            batch_size, 
            Config.INPUT_SIZE, 
            force_positive=True  # Включаем принудительное использование положительных примеров для обучения
        )
        val_dataset = create_tuning_data_pipeline(
            val_loader, 
            Config.SEQUENCE_LENGTH, 
            batch_size, 
            Config.INPUT_SIZE, 
            force_positive=False  # Оставляем отключенным для валидации
        )
        
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
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-4, log=True)  # Уменьшаем диапазон learning rate
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        lstm_units = trial.suggest_int('lstm_units', 64, 256)
        model_type = Config.MODEL_TYPE
        # rnn_type = trial.suggest_categorical('rnn_type', ['lstm', 'bigru'])
        # temporal_block_type = trial.suggest_categorical('temporal_block_type', ['rnn', 'tcn', '3d_attention', 'transformer'])
        rnn_type = trial.suggest_categorical('rnn_type', ['lstm'])
        temporal_block_type = trial.suggest_categorical('temporal_block_type', ['tcn', '3d_attention', 'transformer'])
        clipnorm = trial.suggest_float('clipnorm', 0.1, 2.0)
        
        # Подбираем размер батча с шагом 16 для лучшей стабильности
        batch_size = trial.suggest_int('batch_size', 8, 64, step=8)
        print(f"[DEBUG] Выбран размер батча: {batch_size}")
        
        # Подбираем параметры аугментации
        augmentation_params = {
            'brightness_range': trial.suggest_float('brightness_range', 0.1, 0.3),
            'contrast_range': trial.suggest_float('contrast_range', 0.1, 0.3),
            'rotation_range': trial.suggest_int('rotation_range', 5, 15),
            'noise_std': trial.suggest_float('noise_std', 0.02, 0.08),
            'blur_sigma': trial.suggest_float('blur_sigma', 0.5, 1.5)
        }
        
        # Подбираем вероятности аугментации
        augmentation_probs = {
            'brightness_prob': trial.suggest_float('brightness_prob', 0.3, 0.7),
            'contrast_prob': trial.suggest_float('contrast_prob', 0.3, 0.7),
            'rotation_prob': trial.suggest_float('rotation_prob', 0.3, 0.7),
            'noise_prob': trial.suggest_float('noise_prob', 0.2, 0.4),
            'blur_prob': trial.suggest_float('blur_prob', 0.1, 0.3)
        }
        
        # Обновляем параметры аугментации в конфиге
        Config.AUGMENTATION.update(augmentation_params)
        Config.AUGMENTATION.update(augmentation_probs)
        
        print(f"[DEBUG] Подобранные параметры аугментации:")
        print(f"  - Параметры: {augmentation_params}")
        print(f"  - Вероятности: {augmentation_probs}")
        
        # Загружаем базовые веса из config_weights.json
        try:
            with open(Config.CONFIG_PATH, 'r') as f:
                config = json.load(f)
                base_weights = config['class_weights']
                print(f"[DEBUG] Загружены базовые веса из конфига: {base_weights}")
        except Exception as e:
            print(f"[WARNING] Не удалось загрузить веса классов из конфига: {str(e)}")
            print("[WARNING] Используем значения по умолчанию.")
            base_weights = {
                'background': 1.0,
                'action': 10 
            }
        
        # Подбираем веса с меньшим отклонением для более стабильного обучения
        weight_deviation = trial.suggest_float('weight_deviation', -0.2, 0.0)
        action_weight = base_weights['action'] * (1 + weight_deviation)
        
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
                'temporal_block_type': temporal_block_type,
                'clipnorm': clipnorm,
                'augmentation_params': augmentation_params,
                'augmentation_probs': augmentation_probs
            },
            input_shape=(Config.SEQUENCE_LENGTH, *Config.INPUT_SIZE, 3),
            num_classes=Config.NUM_CLASSES,
            class_weights=class_weights
        )
        
        # Создаем callbacks
        callbacks = get_tuning_callbacks(trial.number)
        
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
        
        # Возвращаем лучший F1-score на валидации
        return max(history.history['val_f1_score'])
        
    except Exception as e:
        print(f"[ERROR] Ошибка в триале #{trial.number}: {str(e)}")
        print("[DEBUG] Stack trace:", flush=True)
        traceback.print_exc()
        clear_memory()
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
            
            # Добавляем информацию о параметрах аугментации
            f.write("\nПараметры аугментации:\n")
            for key, value in Config.AUGMENTATION.items():
                f.write(f"  - {key}: {value}\n")
            
            # Добавляем информацию о лучших параметрах аугментации
            f.write("\nЛучшие параметры аугментации:\n")
            augmentation_params = {k: v for k, v in study.best_params.items() 
                                if k in ['brightness_range', 'contrast_range', 'rotation_range', 
                                       'noise_std', 'blur_sigma', 'brightness_prob', 'contrast_prob',
                                       'rotation_prob', 'noise_prob', 'blur_prob']}
            for key, value in augmentation_params.items():
                f.write(f"  - {key}: {value}\n")
        
        # Создаем визуализации
        try:
            # График оптимизации
            plt.figure(figsize=(10, 6))
            optuna.visualization.matplotlib.plot_optimization_history(study)
            plt.savefig(os.path.join(tuning_dir, 'optimization_history.png'))
            plt.close()
            
            # График важности параметров
            plt.figure(figsize=(12, 8))
            optuna.visualization.matplotlib.plot_param_importances(study)
            plt.savefig(os.path.join(tuning_dir, 'param_importances.png'))
            plt.close()
            
            # График параллельных координат
            plt.figure(figsize=(15, 10))
            optuna.visualization.matplotlib.plot_parallel_coordinate(study)
            plt.savefig(os.path.join(tuning_dir, 'parallel_coordinate.png'))
            plt.close()
            
        except Exception as e:
            print(f"[WARNING] Ошибка при создании визуализаций: {str(e)}")
        
        print(f"[DEBUG] Результаты сохранены в {tuning_dir}")
        
    except Exception as e:
        print(f"[ERROR] Ошибка при сохранении результатов: {str(e)}")
        print("[DEBUG] Stack trace:", flush=True)
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

def create_and_compile_model(params, input_shape, num_classes, class_weights):
    """
    Создание и компиляция модели с заданными параметрами
    """
    print("\n[DEBUG] Создание модели с параметрами:")
    print(f"  - Тип модели: {params['model_type']}")
    print(f"  - Dropout: {params['dropout_rate']}")
    print(f"  - LSTM units: {params['lstm_units']}")
    print(f"  - Тип RNN: {params['rnn_type']}")
    print(f"  - Тип временного блока: {params['temporal_block_type']}")
    print(f"  - Веса классов: {class_weights}")
    print(f"  - Параметры аугментации: {params.get('augmentation_params', {})}")
    print(f"  - Вероятности аугментации: {params.get('augmentation_probs', {})}")
    
    # Обновляем параметры аугментации в конфиге
    if 'augmentation_params' in params:
        Config.AUGMENTATION.update(params['augmentation_params'])
    if 'augmentation_probs' in params:
        Config.AUGMENTATION.update(params['augmentation_probs'])
    
    # Создаем модель
    model = create_model_with_params(
        model_type=params['model_type'],
        input_shape=input_shape,
        num_classes=num_classes,
        params={
            'dropout_rate': params['dropout_rate'],
            'lstm_units': params['lstm_units'],
            'rnn_type': params['rnn_type'],
            'temporal_block_type': params['temporal_block_type']
        },
        class_weights=class_weights
    )
    
    # Создаем оптимизатор
    optimizer = Adam(
        learning_rate=params['learning_rate'],
        clipnorm=params.get('clipnorm', 1.0)
    )
    
    # Компилируем модель с focal loss и метриками
    model.compile(
        optimizer=optimizer,
        loss=focal_loss(gamma=2.0, alpha=[class_weights['background'], class_weights['action']]),
        metrics=get_tuning_metrics()
    )
    
    return model
