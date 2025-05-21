import os
import time
import json
import traceback
import numpy as np
from datetime import timedelta, datetime
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import optuna
from src.models.model import (
    create_model_with_params,
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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.mixed_precision import Policy
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

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

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Включаем mixed precision
        policy = Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("Mixed precision policy set:", policy.name)
        
        # Проверяем, что GPU действительно используется
        with tf.device('/GPU:0'):
            a = tf.random.normal([1000, 1000])
            b = tf.random.normal([1000, 1000])
            c = tf.matmul(a, b)
            print("GPU test successful")
    except RuntimeError as e:
        print(f"Error setting up GPU: {e}")
else:
    print("\nNo GPU devices found")

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
        
        # Сначала загружаем обучающий набор
        print("[DEBUG] Загрузка обучающего набора...")
        train_loader = VideoDataLoader(
            data_path=Config.TRAIN_DATA_PATH,
            max_videos=Config.MAX_VIDEOS
        )
        
        # Кэшируем последовательности для обучения
        print("[DEBUG] Кэширование обучающих последовательностей...")
        cached_train_sequences = []
        cached_train_labels = []
        
        while len(train_loader.processed_video_paths) < train_loader.total_videos:
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
        
        print(f"[DEBUG] Загружено {len(cached_train_sequences)} обучающих последовательностей")
        print(f"[DEBUG] Обработано видео: {len(train_loader.processed_video_paths)}/{train_loader.total_videos} ({len(train_loader.processed_video_paths)/train_loader.total_videos*100:.1f}%)")
        
        # Очищаем память после обработки обучающего набора
        train_loader.clear_cache()
        del train_loader
        gc.collect()
        
        # Теперь загружаем валидационный набор
        print("\n[DEBUG] Загрузка валидационного набора...")
        val_loader = VideoDataLoader(
            data_path=Config.VALID_DATA_PATH,
            max_videos=Config.MAX_VIDEOS
        )
        
        # Кэшируем последовательности для валидации
        print("[DEBUG] Кэширование валидационных последовательностей...")
        cached_val_sequences = []
        cached_val_labels = []
        
        while len(val_loader.processed_video_paths) < val_loader.total_videos:
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
        
        print(f"[DEBUG] Загружено {len(cached_val_sequences)} валидационных последовательностей")
        
        # Очищаем память после обработки валидационного набора
        val_loader.clear_cache()
        del val_loader
        gc.collect()
        
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
    """
    Целевая функция для оптимизации гиперпараметров
    """
    try:
        print(f"\n[DEBUG] ===== ============================ ======")
        print(f"\n[DEBUG] ===== Начало триала #{trial.number} =====")
        print(f"\n[DEBUG] ===== ============================ ======")
        
        # Получаем гиперпараметры из trial
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        lstm_units = trial.suggest_int('lstm_units', 32, 256)
        model_type = Config.MODEL_TYPE
        rnn_type = trial.suggest_categorical('rnn_type', ['lstm', 'bigru'])
        temporal_block_type = trial.suggest_categorical('temporal_block_type', ['rnn', 'hybrid', '3d_attention', 'transformer'])
        clipnorm = trial.suggest_float('clipnorm', 0.1, 2.0)
        
        print(f"[DEBUG] Параметры триала #{trial.number}:")
        print(f"  - learning_rate: {learning_rate}")
        print(f"  - dropout_rate: {dropout_rate}")
        print(f"  - lstm_units: {lstm_units}")
        print(f"  - rnn_type: {rnn_type}")
        print(f"  - temporal_block_type: {temporal_block_type}")
        print(f"  - clipnorm: {clipnorm}")
        
        # Создаем модель с текущими гиперпараметрами
        model = create_model_with_params(
            model_type=model_type,
            input_shape=(Config.SEQUENCE_LENGTH, *Config.INPUT_SIZE, 3),
            num_classes=2,
            params={
                'dropout_rate': dropout_rate,
                'lstm_units': lstm_units,
                'rnn_type': rnn_type,
                'temporal_block_type': temporal_block_type,
                'clipnorm': clipnorm
            },
            class_weights={
                'background': 1.0,
                'action': 1.0
            }
        )
        
        # Загружаем данные
        data_loader = VideoDataLoader(
            data_path=Config.TRAIN_DATA_PATH,
            max_videos=Config.MAX_VIDEOS
        )
        
        train_data, val_data = load_and_prepare_data(Config.BATCH_SIZE)
        
        # Создаем директорию для сохранения результатов
        trial_dir = os.path.join(Config.MODEL_SAVE_PATH, 'tuning', f'trial_{trial.number}')
        os.makedirs(trial_dir, exist_ok=True)
        
        # Создаем callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=os.path.join(trial_dir, 'best_model.h5'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Обучаем модель
        history = model.fit(
            train_data,
            epochs=Config.EPOCHS,
            steps_per_epoch=Config.STEPS_PER_EPOCH,
            validation_data=val_data,
            validation_steps=Config.VALIDATION_STEPS,
            callbacks=callbacks,
            verbose=1
        )
        
        # Сохраняем историю обучения
        history_dict = {
            'loss': [float(x) for x in history.history['loss']],
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']]
        }
        
        with open(os.path.join(trial_dir, 'history.json'), 'w') as f:
            json.dump(history_dict, f)
        
        # Визуализируем результаты
        plot_training_history(history, trial_dir)
        
        # Оцениваем модель на валидационном наборе
        val_metrics = model.evaluate(val_data, steps=Config.VALIDATION_STEPS, verbose=1)
        val_loss, val_accuracy = val_metrics
        
        # Сохраняем метрики
        metrics = {
            'val_loss': float(val_loss),
            'val_accuracy': float(val_accuracy)
        }
        
        with open(os.path.join(trial_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f)
        
        # Очищаем память
        tf.keras.backend.clear_session()
        gc.collect()
        
        return val_accuracy
        
    except Exception as e:
        print(f"\n[ERROR] Ошибка в триале #{trial.number}: {str(e)}")
        return None

def tune_hyperparameters():
    """
    Функция для оптимизации гиперпараметров модели
    """
    try:
        print("\n[DEBUG] ===== Начало оптимизации гиперпараметров =====")
        
        # Создаем директорию для сохранения результатов
        tuning_dir = os.path.join(Config.MODEL_SAVE_PATH, 'tuning')
        os.makedirs(tuning_dir, exist_ok=True)
        
        # Создаем study
        study = optuna.create_study(
            direction='maximize',
            study_name=f'hyperparameter_tuning_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        )
        
        # Запускаем оптимизацию
        study.optimize(objective, n_trials=Config.N_TRIALS)
        
        # Сохраняем результаты
        results = []
        for trial in study.trials:
            if trial.value is not None:
                results.append({
                    'trial_number': trial.number,
                    'value': trial.value,
                    'params': trial.params
                })
        
        # Сортируем результаты по значению метрики
        results.sort(key=lambda x: x['value'], reverse=True)
        
        # Сохраняем результаты в файл
        with open(os.path.join(tuning_dir, 'optuna_results.txt'), 'w') as f:
            f.write("Results of hyperparameter optimization:\n\n")
            for result in results:
                f.write(f"Trial #{result['trial_number']}\n")
                f.write(f"Value: {result['value']:.4f}\n")
                f.write("Parameters:\n")
                for param, value in result['params'].items():
                    f.write(f"  {param}: {value}\n")
                f.write("\n")
        
        # Визуализируем результаты
        plot_optimization_history(study, tuning_dir)
        plot_param_importances(study, tuning_dir)
        
        print("\n[DEBUG] ===== Оптимизация гиперпараметров завершена =====")
        
        return study.best_params, study.best_value
        
    except Exception as e:
        print(f"\n[ERROR] Ошибка при оптимизации гиперпараметров: {str(e)}")
        return None, None

def plot_optimization_history(study, save_dir):
    """
    Визуализация истории оптимизации
    """
    try:
        # Создаем DataFrame с результатами
        trials_df = pd.DataFrame([
            {
                'trial': t.number,
                'value': t.value,
                'datetime_start': t.datetime_start,
                'datetime_complete': t.datetime_complete
            }
            for t in study.trials
            if t.value is not None
        ])
        
        # Сортируем по номеру триала
        trials_df = trials_df.sort_values('trial')
        
        # Создаем график
        fig = go.Figure()
        
        # Добавляем линию значений
        fig.add_trace(go.Scatter(
            x=trials_df['trial'],
            y=trials_df['value'],
            mode='lines+markers',
            name='Значение метрики',
            line=dict(color='blue'),
            marker=dict(size=8)
        ))
        
        # Добавляем линию лучшего значения
        best_values = [study.best_value] * len(trials_df)
        fig.add_trace(go.Scatter(
            x=trials_df['trial'],
            y=best_values,
            mode='lines',
            name='Лучшее значение',
            line=dict(color='red', dash='dash')
        ))
        
        # Настраиваем layout
        fig.update_layout(
            title='История оптимизации гиперпараметров',
            xaxis_title='Номер триала',
            yaxis_title='Значение метрики',
            showlegend=True,
            template='plotly_white'
        )
        
        # Сохраняем график
        fig.write_image(os.path.join(save_dir, 'optimization_history.png'))
        fig.write_html(os.path.join(save_dir, 'optimization_history.html'))
        
    except Exception as e:
        print(f"\n[ERROR] Ошибка при создании графика истории оптимизации: {str(e)}")

def plot_param_importances(study, save_dir):
    """
    Визуализация важности параметров
    """
    try:
        # Получаем важность параметров
        param_importances = optuna.importance.get_param_importances(study)
        
        # Создаем DataFrame
        importances_df = pd.DataFrame({
            'parameter': list(param_importances.keys()),
            'importance': list(param_importances.values())
        })
        
        # Сортируем по важности
        importances_df = importances_df.sort_values('importance', ascending=True)
        
        # Создаем график
        fig = go.Figure()
        
        # Добавляем горизонтальные бары
        fig.add_trace(go.Bar(
            y=importances_df['parameter'],
            x=importances_df['importance'],
            orientation='h',
            marker=dict(color='blue')
        ))
        
        # Настраиваем layout
        fig.update_layout(
            title='Важность гиперпараметров',
            xaxis_title='Важность',
            yaxis_title='Параметр',
            showlegend=False,
            template='plotly_white'
        )
        
        # Сохраняем график
        fig.write_image(os.path.join(save_dir, 'param_importances.png'))
        fig.write_html(os.path.join(save_dir, 'param_importances.html'))
        
    except Exception as e:
        print(f"\n[ERROR] Ошибка при создании графика важности параметров: {str(e)}")

if __name__ == "__main__":
    try:
        print("[DEBUG] ===== Запуск оптимизации гиперпараметров =====")
        best_params, best_value = tune_hyperparameters()
        
        if best_params is not None:
            print("\n[INFO] Лучшие параметры:")
            for param, value in best_params.items():
                print(f"  {param}: {value}")
            print(f"[INFO] Лучшее значение метрики: {best_value:.4f}")
            
            # Сохраняем лучшие параметры в файл
            with open('best_params.txt', 'w') as f:
                f.write("Best parameters:\n")
                for param, value in best_params.items():
                    f.write(f"{param}: {value}\n")
                f.write(f"\nBest validation accuracy: {best_value:.4f}\n")
        else:
            print("\n[ERROR] Не удалось найти оптимальные параметры")
            
    except Exception as e:
        print(f"\n[ERROR] Критическая ошибка при запуске оптимизации: {str(e)}")
        sys.exit(1)
