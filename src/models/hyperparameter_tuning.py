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
    postprocess_predictions,
    indices_to_seconds,
    merge_classes
)
from src.data_proc.data_loader import VideoDataLoader
from src.config import Config
from src.models.losses import focal_loss
from src.models.metrics import get_tuning_metrics
from src.models.callbacks import get_tuning_callbacks
from src.utils.gpu_config import setup_gpu
from src.utils.network_handler import NetworkErrorHandler, NetworkMonitor
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import gc
import subprocess
import sys
import cv2
from optuna.trial import Trial
from src.data_proc.data_validation import validate_data_pipeline, validate_training_data
from src.models.train import create_data_pipeline, create_tuning_data_pipeline

# Настройка GPU
setup_gpu()

# Объявляем глобальные переменные в начале файла
train_loader = None
val_loader = None
train_data = None
val_data = None
# Добавляем глобальные переменные для кэширования данных
cached_train_sequences = None
cached_train_labels = None
cached_val_sequences = None
cached_val_labels = None

def clear_memory():
    """Очистка памяти"""
    global train_loader, val_loader, train_data, val_data
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
        
        # Очищаем загрузчики и датасеты, но сохраняем кэшированные данные
        train_loader = None
        val_loader = None
        train_data = None
        val_data = None
        
    except Exception as e:
        print(f"[DEBUG] ✗ Критическая ошибка при очистке памяти: {str(e)}")
        print("[DEBUG] Stack trace:", flush=True)
        traceback.print_exc()
    
    print("[DEBUG] ===== Очистка памяти завершена =====\n")

def load_and_prepare_data(batch_size):
    """
    Загрузка и подготовка данных для подбора гиперпараметров
    """
    global train_loader, val_loader, train_data, val_data
    global cached_train_sequences, cached_train_labels, cached_val_sequences, cached_val_labels
    
    try:
        print("\n[DEBUG] Загрузка данных для подбора гиперпараметров...")
        
        # Если данные уже закэшированы, используем их
        if (cached_train_sequences is not None and cached_train_labels is not None and
            cached_val_sequences is not None and cached_val_labels is not None):
            print("[DEBUG] Используем кэшированные данные...")
            
            # Создаем датасеты из кэшированных данных
            train_data = tf.data.Dataset.from_tensor_slices((cached_train_sequences, cached_train_labels))
            train_data = train_data.shuffle(len(cached_train_sequences))
            train_data = train_data.batch(batch_size)
            train_data = train_data.prefetch(Config.MEMORY_OPTIMIZATION['prefetch_buffer_size'])
            
            val_data = tf.data.Dataset.from_tensor_slices((cached_val_sequences, cached_val_labels))
            val_data = val_data.batch(batch_size)
            val_data = val_data.prefetch(Config.MEMORY_OPTIMIZATION['prefetch_buffer_size'])
            
            print(f"[DEBUG] Использовано {len(cached_train_sequences)} обучающих и {len(cached_val_sequences)} валидационных последовательностей из кэша")
            return train_data, val_data
        
        # Если данных нет в кэше, загружаем их
        print("[DEBUG] Загрузка новых данных...")
        
        # Создаем загрузчики данных
        train_loader = VideoDataLoader(
            data_path=Config.TRAIN_DATA_PATH,
            max_videos=Config.MAX_VIDEOS
        )
        
        val_loader = VideoDataLoader(
            data_path=Config.VALID_DATA_PATH,
            max_videos=Config.MAX_VIDEOS
        )
        
        # Кэшируем последовательности для обучения
        print("[DEBUG] Кэширование обучающих последовательностей...")
        cached_train_sequences = []
        cached_train_labels = []
        
        while len(train_loader.processed_video_paths) < len(train_loader.video_paths):
            X, y = train_loader._get_sequence(
                sequence_length=Config.SEQUENCE_LENGTH,
                target_size=Config.INPUT_SIZE,
                force_positive=True,
                is_validation=False
            )
            
            if X is not None and y is not None and len(X) > 0 and len(y) > 0 and np.any(y):
                y_one_hot = np.zeros((Config.SEQUENCE_LENGTH, 2), dtype=np.float32)
                for i in range(Config.SEQUENCE_LENGTH):
                    if isinstance(y[i], np.ndarray):
                        if y[i].size == 2:
                            label = np.argmax(y[i])
                        else:
                            label = int(y[i].item())
                    else:
                        label = int(y[i])
                    y_one_hot[i, label] = 1
                
                cached_train_sequences.append(X)
                cached_train_labels.append(y_one_hot)
        
        # Кэшируем последовательности для валидации
        print("[DEBUG] Кэширование валидационных последовательностей...")
        cached_val_sequences = []
        cached_val_labels = []
        
        while len(val_loader.processed_video_paths) < len(val_loader.video_paths):
            X, y = val_loader._get_sequence(
                sequence_length=Config.SEQUENCE_LENGTH,
                target_size=Config.INPUT_SIZE,
                force_positive=False,
                is_validation=True
            )
            
            if X is not None and y is not None and len(X) > 0 and len(y) > 0 and np.any(y):
                y_one_hot = np.zeros((Config.SEQUENCE_LENGTH, 2), dtype=np.float32)
                for i in range(Config.SEQUENCE_LENGTH):
                    if isinstance(y[i], np.ndarray):
                        if y[i].size == 2:
                            label = np.argmax(y[i])
                        else:
                            label = int(y[i].item())
                    else:
                        label = int(y[i])
                    y_one_hot[i, label] = 1
                
                cached_val_sequences.append(X)
                cached_val_labels.append(y_one_hot)
        
        # Создаем датасеты из кэшированных данных
        print("[DEBUG] Создание датасетов из кэшированных данных...")
        train_data = tf.data.Dataset.from_tensor_slices((cached_train_sequences, cached_train_labels))
        train_data = train_data.shuffle(len(cached_train_sequences))
        train_data = train_data.batch(batch_size)
        train_data = train_data.prefetch(Config.MEMORY_OPTIMIZATION['prefetch_buffer_size'])
        
        val_data = tf.data.Dataset.from_tensor_slices((cached_val_sequences, cached_val_labels))
        val_data = val_data.batch(batch_size)
        val_data = val_data.prefetch(Config.MEMORY_OPTIMIZATION['prefetch_buffer_size'])
        
        print(f"[DEBUG] Загружено {len(cached_train_sequences)} обучающих и {len(cached_val_sequences)} валидационных последовательностей")
        return train_data, val_data
        
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
        print(f"=======================================")
        print(f"\n[DEBUG] Начало триала #{trial.number}")
        print(f"[DEBUG] Время начала: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"=======================================")
        
        # Очищаем память перед началом триала
        clear_memory()
        
        # Получаем гиперпараметры
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-4, log=True)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        lstm_units = trial.suggest_int('lstm_units', 64, 256)
        model_type = Config.MODEL_TYPE
        rnn_type = trial.suggest_categorical('rnn_type', ['lstm'])
        temporal_block_type = trial.suggest_categorical('temporal_block_type', ['rnn','tcn','transformer'])
        clipnorm = trial.suggest_float('clipnorm', 0.1, 2.0)
        
        # Подбираем размер батча с шагом 16 для лучшей стабильности
        batch_size = trial.suggest_int('batch_size', 8, 64, step=16)
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
            'noise_prob': trial.suggest_float('noise_prob', 0.2, 0.5),
            'blur_prob': trial.suggest_float('blur_prob', 0.1, 0.3)
        }
        
        print(f"[DEBUG] Параметры триала #{trial.number}:")
        print(f"  - learning_rate: {learning_rate}")
        print(f"  - dropout_rate: {dropout_rate}")
        print(f"  - lstm_units: {lstm_units}")
        print(f"  - model_type: {model_type}")
        print(f"  - rnn_type: {rnn_type}")
        print(f"  - temporal_block_type: {temporal_block_type}")
        print(f"  - clipnorm: {clipnorm}")
        print(f"  - batch_size: {batch_size}")
        print(f"  - augmentation_params: {augmentation_params}")
        print(f"  - augmentation_probs: {augmentation_probs}")
        
        # Рассчитываем веса классов
        print(f"[DEBUG] Тип модели: {model_type}")
        print(f"[DEBUG] MODEL_PARAMS: {Config.MODEL_PARAMS}")

        try:
            # Загружаем веса из файла конфигурации
            if os.path.exists(Config.CONFIG_PATH):
                print(f"[DEBUG] Загрузка весов из {Config.CONFIG_PATH}")
                with open(Config.CONFIG_PATH, 'r') as f:
                    config = json.load(f)
                    base_weights = config.get('class_weights', {
                        'background': 1.0,
                        'action': 10.0
                    })
                    print(f"[DEBUG] Загружены веса из файла: {base_weights}")
            else:
                print("[WARNING] Файл конфигурации не найден, используем веса по умолчанию")
                base_weights = {
                    'background': 1.0,
                    'action': 10.0
                }
            
            # Создаем и компилируем модель
            print("[DEBUG] Создание и компиляция модели...")
            print(f"[DEBUG] Передаем тип модели в create_model_with_params: {model_type}")
            print(f"[DEBUG] Проверка типа модели: {model_type.lower()}")
            model = create_model_with_params(
                model_type=model_type,
                input_shape=(Config.SEQUENCE_LENGTH, *Config.INPUT_SIZE, 3),
                num_classes=2,
                params={
                    'dropout_rate': dropout_rate,
                    'lstm_units': lstm_units,
                    'rnn_type': rnn_type,
                    'temporal_block_type': temporal_block_type
                },
                class_weights=base_weights
            )
            
            optimizer = Adam(
                learning_rate=learning_rate,
                clipnorm=clipnorm
            )
            
            model.compile(
                optimizer=optimizer,
                loss=focal_loss(gamma=2.0, alpha=[base_weights['background'], base_weights['action']]),
                metrics=get_tuning_metrics()
            )
            
            print("[DEBUG] Модель успешно создана и скомпилирована")
            
            # Загружаем данные
            print("[DEBUG] Загрузка данных...")
            train_data, val_data = load_and_prepare_data(batch_size)
            print("[DEBUG] Данные успешно загружены")
            
            # Создаем колбэки
            print("[DEBUG] Создание колбэков...")
            callbacks = get_tuning_callbacks(trial.number)
            print("[DEBUG] Колбэки успешно созданы")
            
            # Обучаем модель
            print("[DEBUG] Начало обучения модели...")
            history = model.fit(
                train_data,
                epochs=Config.HYPERPARAM_TUNING['epochs'],
                validation_data=val_data,
                callbacks=callbacks,
                verbose=1
            )
            print("[DEBUG] Обучение модели завершено")
            
            # Получаем лучший F1-score
            val_f1_scores = history.history['val_scalar_f1_score']
            if isinstance(val_f1_scores, list):
                best_f1 = max(val_f1_scores)
            else:
                best_f1 = float(val_f1_scores)
            print(f"[DEBUG] Лучший F1-score: {best_f1}")
            
            # Очищаем память
            clear_memory()
            
            return best_f1
            
        except Exception as e:
            print(f"[ERROR] Ошибка в процессе обучения: {str(e)}")
            print("[DEBUG] Stack trace:", flush=True)
            traceback.print_exc()
            clear_memory()
            raise
            
    except Exception as e:
        print(f"[ERROR] Критическая ошибка в триале #{trial.number}: {str(e)}")
        print("[DEBUG] Stack trace:", flush=True)
        traceback.print_exc()
        clear_memory()
        raise

def save_tuning_results(study, total_time, n_trials):
    """
    Сохранение результатов подбора гиперпараметров + визуализация и подробный лог
    """
    try:
        model_type = Config.MODEL_TYPE
        tuning_dir = os.path.join(Config.MODEL_SAVE_PATH, 'tuning', model_type)
        print(f"\n[DEBUG] Сохранение результатов подбора гиперпараметров в {tuning_dir}...")
        
        # Очищаем старые результаты только из папки текущей модели
        if os.path.exists(tuning_dir):
            print(f"[DEBUG] Удаление старых результатов из {tuning_dir}...")
            for file in os.listdir(tuning_dir):
                file_path = os.path.join(tuning_dir, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"[WARNING] Ошибка при удалении файла {file_path}: {str(e)}")
        else:
            print(f"[DEBUG] Папка {tuning_dir} не существует, создаю...")
            os.makedirs(tuning_dir, exist_ok=True)
        
        # Сохраняем результаты в текстовый файл
        results_file = os.path.join(tuning_dir, 'optuna_results.txt')
        with open(results_file, 'w') as f:
            f.write(f"=== Результаты оптимизации гиперпараметров ===\n")
            f.write(f"Модель: {model_type}\n")
            f.write(f"Время выполнения: {timedelta(seconds=int(total_time))}\n")
            f.write(f"Количество trials: {n_trials}\n")
            f.write(f"Лучшее значение: {study.best_value}\n")
            f.write("\nЛучшие параметры:\n")
            for key, value in study.best_params.items():
                f.write(f"{key}: {value}\n")
        
        # Сохраняем все параметры в JSON файл
        params_file = os.path.join(tuning_dir, 'best_params.json')
        params = {
            'model_params': {
                'learning_rate': study.best_params['learning_rate'],
                'dropout_rate': study.best_params['dropout_rate'],
                'lstm_units': study.best_params['lstm_units'],
                'rnn_type': study.best_params['rnn_type'],
                'temporal_block_type': study.best_params['temporal_block_type'],
                'clipnorm': study.best_params['clipnorm'],
                'batch_size': study.best_params['batch_size']
            },
            'augmentation_params': {
                'brightness_range': study.best_params['brightness_range'],
                'contrast_range': study.best_params['contrast_range'],
                'rotation_range': study.best_params['rotation_range'],
                'noise_std': study.best_params['noise_std'],
                'blur_sigma': study.best_params['blur_sigma'],
                'brightness_prob': study.best_params['brightness_prob'],
                'contrast_prob': study.best_params['contrast_prob'],
                'rotation_prob': study.best_params['rotation_prob'],
                'noise_prob': study.best_params['noise_prob'],
                'blur_prob': study.best_params['blur_prob']
            },
            'model_type': model_type,
            'best_value': study.best_value,
            'total_time': total_time,
            'n_trials': n_trials
        }
        with open(params_file, 'w') as f:
            json.dump(params, f, indent=4)
        
        # Создаем визуализации
        try:
            # График оптимизации
            plt.figure(figsize=(10, 6))
            optuna.visualization.matplotlib.plot_optimization_history(study)
            plt.savefig(os.path.join(tuning_dir, 'optimization_history.png'))
            plt.close()
            
            # График важности параметров
            ax = optuna.visualization.matplotlib.plot_param_importances(study)
            ax.set_title('Важность гиперпараметров', fontsize=18, pad=20)
            ax.figure.savefig(os.path.join(tuning_dir, 'param_importances.png'), bbox_inches='tight')
            plt.close(ax.figure)
            
            # График параллельных координат
            plt.figure(figsize=(15, 10))
            optuna.visualization.matplotlib.plot_parallel_coordinate(study)
            plt.savefig(os.path.join(tuning_dir, 'parallel_coordinate.png'))
            plt.close()
            
        except Exception as e:
            print(f"[WARNING] Ошибка при создании визуализаций: {str(e)}")
            traceback.print_exc()
        
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
        
        # Извлекаем лучшие параметры и значение
        best_params = study.best_params
        best_value = study.best_value
        
        print("\n[INFO] Лучшие параметры:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        print(f"[INFO] Лучшее значение метрики: {best_value:.4f}")
        
        return best_params, best_value
        
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
