<<<<<<< HEAD
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
from src.models.model import create_mobilenetv3_model, create_mobilenetv4_model, create_model, focal_loss
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
from tensorflow.keras.metrics import Precision, Recall, F1Score
import subprocess
import sys
import json
import cv2
from tensorflow.keras.optimizers import Adam
import psutil
from src.data_proc.data_augmentation import VideoAugmenter
from optuna.trial import Trial
from src.utils.network_handler import NetworkErrorHandler, NetworkMonitor
=======
import cv2
import numpy as np
from typing import Tuple, List, Generator
import os
import json
from src.data_proc.annotation import VideoAnnotation
from src.config import Config
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor
import threading
from src.utils.network_handler import NetworkErrorHandler, NetworkMonitor
import logging
import gc
>>>>>>> 14f4b73b718e50c32cc5e9db2625586f89c8a60c

logger = logging.getLogger(__name__)

class VideoDataLoader:
    def __init__(self, data_path, max_videos=Config.MAX_VIDEOS):
        """
        Инициализация загрузчика данных
        Args:
            data_path: путь к директории с данными
            max_videos: максимальное количество видео для загрузки (None для загрузки всех видео)
        """
        self.positive_indices_cache = {}  # Кэш для индексов положительных кадров (инициализация первой!)
        self.data_path = data_path
        self.max_videos = max_videos
        self.video_paths = []
        self.labels = []
        self.video_count = 0
        self.batch_size = 32
        self.current_video_index = 0
        self.current_frame_index = 0
        self.current_batch = 0
        self.total_batches = 0
        self.network_handler = NetworkErrorHandler()
        self.network_monitor = NetworkMonitor()
        self.cache = {}
        self.cache_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Инициализация параметров из конфигурации
        self.sequence_length = Config.SEQUENCE_LENGTH
        self.max_sequences_per_video = Config.MAX_SEQUENCES_PER_VIDEO
        
        # Загружаем видео
        self._load_videos()
        
        # Рассчитываем общее количество батчей
        self._calculate_total_batches()
        
        print(f"[DEBUG] Загружено {self.video_count} видео")
        if self.max_videos is not None and self.video_count > self.max_videos:
            print(f"[WARNING] Загружено слишком много видео: {self.video_count} > {self.max_videos}")
            self.video_paths = self.video_paths[:self.max_videos]
            self.labels = self.labels[:self.max_videos]
            self.video_count = self.max_videos
            print(f"[DEBUG] Оставлено {self.video_count} видео")
    
    def _load_videos(self):
        """
        Загрузка путей к видео и соответствующих аннотаций.
        
        Raises:
            FileNotFoundError: Если директория с данными не найдена
            ValueError: Если нет видео файлов в директории
        """
        try:
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"Директория с данными не найдена: {self.data_path}")
            
            # Определяем путь к аннотациям в зависимости от типа данных (train/valid)
            if 'train' in self.data_path:
                annotation_dir = Config.TRAIN_ANNOTATION_PATH
            else:
                annotation_dir = Config.VALID_ANNOTATION_PATH
            
            if not os.path.exists(annotation_dir):
                print(f"[DEBUG] Создание директории для аннотаций: {annotation_dir}")
                os.makedirs(annotation_dir, exist_ok=True)
            
            print(f"[DEBUG] Поиск видео в {self.data_path}, аннотаций в {annotation_dir}")
            
            self.video_paths = []
            self.labels = []
            self.video_count = 0  # Добавляем счетчик
            
            for file_name in os.listdir(self.data_path):
                if self.max_videos is not None and self.video_count >= self.max_videos:
                    break
                
<<<<<<< HEAD
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

def create_data_pipeline(data_loader, sequence_length, batch_size, input_size, is_training=True, force_positive=False):
    """
    Создание оптимизированного пайплайна данных.
    Args:
        data_loader: Загрузчик данных
        sequence_length: Длина последовательности
        batch_size: Размер батча
        input_size: Размер входного изображения
        is_training: Флаг обучения
        force_positive: Флаг принудительного включения положительных примеров
    Returns:
        tf.data.Dataset: Оптимизированный датасет
    """
    try:
        print("\n[DEBUG] ===== Создание пайплайна данных =====")
        print(f"[DEBUG] Параметры:")
        print(f"  - batch_size: {batch_size}")
        print(f"  - sequence_length: {sequence_length}")
        print(f"  - input_size: {input_size}")
        print(f"  - is_training: {is_training}")
        print(f"  - force_positive: {force_positive}")
        
        # Проверяем количество загруженных видео
        if hasattr(data_loader, 'video_count'):
            print(f"[DEBUG] Количество загруженных видео: {data_loader.video_count}")
            if data_loader.video_count > Config.MAX_VIDEOS:
                print(f"[WARNING] Загружено слишком много видео: {data_loader.video_count} > {Config.MAX_VIDEOS}")
        
        # Устанавливаем размер батча в загрузчике данных
        data_loader.batch_size = batch_size
        
        # Создаем датасет из генератора
        dataset = tf.data.Dataset.from_generator(
            data_loader.data_generator,
            output_signature=(
                tf.TensorSpec(shape=(batch_size, sequence_length, *input_size, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(batch_size, sequence_length, Config.NUM_CLASSES), dtype=tf.float32)
            )
        )
        
        # Применяем оптимизации
        if Config.MEMORY_OPTIMIZATION['cache_dataset']:
            dataset = dataset.cache()
        
        if is_training:
            dataset = dataset.shuffle(buffer_size=1000)
        
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        print("[DEBUG] Pipeline данных успешно создан")
        return dataset
        
    except Exception as e:
        print(f"[ERROR] Ошибка при создании пайплайна данных: {str(e)}")
        print("[DEBUG] Stack trace:", flush=True)
        traceback.print_exc()
        raise

def create_and_compile_model(input_shape, num_classes, learning_rate, dropout_rate, lstm_units=None, model_type='v3', positive_class_weight=None):
    """
    Создание и компиляция модели с заданными параметрами
    Args:
        input_shape: форма входных данных
        num_classes: количество классов
        learning_rate: скорость обучения
        dropout_rate: коэффициент dropout
        lstm_units: количество юнитов в LSTM слоях (только для v3)
        model_type: тип модели ('v3' или 'v4')
        positive_class_weight: вес положительного класса (если None, будет загружен из конфига)
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
    
    # Если positive_class_weight не указан, загружаем из конфига
    if positive_class_weight is None:
        if os.path.exists(Config.CONFIG_PATH):
            with open(Config.CONFIG_PATH, 'r') as f:
                config = json.load(f)
                positive_class_weight = config['MODEL_PARAMS'][model_type]['positive_class_weight']
        else:
            raise ValueError("Конфигурационный файл не найден. Сначала запустите calculate_weights.py")
    
    print(f"  - Positive class weight: {positive_class_weight}")
    
    # Проверяем и корректируем input_shape
    if len(input_shape) == 3:  # Если это (height, width, channels)
        full_input_shape = (Config.SEQUENCE_LENGTH,) + input_shape
    elif len(input_shape) == 4:  # Если это (sequence_length, height, width, channels)
        full_input_shape = input_shape
    elif len(input_shape) == 5:  # Если есть лишняя размерность
        full_input_shape = tuple(s for i, s in enumerate(input_shape) if i != 1 or s != 1)
    else:
        raise ValueError(f"Неверная форма входных данных: {input_shape}")
    
    print(f"[DEBUG] Исправленный input_shape: {full_input_shape}")
    
    model, class_weights = create_model(
        input_shape=full_input_shape,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        lstm_units=lstm_units,
        model_type=model_type,
        positive_class_weight=positive_class_weight
    )
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Включаем mixed precision если используется GPU
    if Config.DEVICE_CONFIG['use_gpu'] and Config.MEMORY_OPTIMIZATION['use_mixed_precision']:
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    
    # Создаем метрики
    print("[DEBUG] Создание метрик...")
    metrics = [
        'accuracy',
        tf.keras.metrics.Precision(name='precision_element', class_id=1, thresholds=0.5),
        tf.keras.metrics.Recall(name='recall_element', class_id=1, thresholds=0.5)
    ]

    print("[DEBUG] Добавление F1Score...")
    try:
        # Создаем адаптер для F1Score
        class F1ScoreAdapter(tf.keras.metrics.F1Score):
            def update_state(self, y_true, y_pred, sample_weight=None):
                y_true = tf.argmax(y_true, axis=-1)
                y_pred = tf.argmax(y_pred, axis=-1)
                y_true = tf.reshape(y_true, [-1])
                y_pred = tf.reshape(y_pred, [-1])
                y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=2)
                y_pred = tf.one_hot(tf.cast(y_pred, tf.int32), depth=2)
                return super().update_state(y_true, y_pred, sample_weight)
            
            def result(self):
                # Получаем результат от родительского класса
                result = super().result()
                # Возвращаем среднее значение по всем классам
                return tf.reduce_mean(result)
        
        f1_metric = F1ScoreAdapter(name='f1_score_element', threshold=0.5)
        print(f"[DEBUG] F1Score создан успешно: {f1_metric}")
        metrics.append(f1_metric)
    except Exception as e:
        print(f"[ERROR] Ошибка при создании F1Score: {str(e)}")
        print("[DEBUG] Stack trace:", flush=True)
        import traceback
        traceback.print_exc()

    print(f"[DEBUG] Итоговый список метрик: {metrics}")

    # Компилируем модель
    model.compile(
        optimizer=optimizer,
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=metrics
    )
    
    print("[DEBUG] Модель успешно создана и скомпилирована")
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
        
        train_loader = VideoDataLoader(
            Config.TRAIN_DATA_PATH,
            max_videos=None
        )
        val_loader = VideoDataLoader(
            Config.VALID_DATA_PATH,
            max_videos=None
        )
        print("[DEBUG] VideoDataLoader создан успешно")
        
        target_size = Config.INPUT_SIZE
        
        # Создание оптимизированных pipeline данных
        train_dataset = create_data_pipeline(train_loader, Config.SEQUENCE_LENGTH, Config.BATCH_SIZE, Config.INPUT_SIZE, True, True)
        val_dataset = create_data_pipeline(val_loader, Config.SEQUENCE_LENGTH, Config.BATCH_SIZE, Config.INPUT_SIZE, False, True)
        
        return train_dataset, val_dataset
    except Exception as e:
        print(f"[ERROR] Ошибка при загрузке данных: {str(e)}")
        print("[DEBUG] Stack trace:", flush=True)
        traceback.print_exc()
        clear_memory()
        raise

def objective(trial):
    """
    Функция для оптимизации гиперпараметров
    """
    try:
        # Очищаем память перед каждым испытанием
        clear_memory()
        
        # Определяем гиперпараметры для поиска
        learning_rate = trial.suggest_float('learning_rate', 1e-7, 1e-2, log=True)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.8)
        lstm_units = trial.suggest_int('lstm_units', 128, 256)  # Увеличиваем диапазон для первого LSTM слоя
        positive_class_weight = trial.suggest_float('positive_class_weight', 
                                                  Config.FOCAL_LOSS['class_weights'][1] * 0.7,
                                                  Config.FOCAL_LOSS['class_weights'][1] * 1.3)
        
        # Добавляем гиперпараметры аугментации
        augment_probability = trial.suggest_float('augment_probability', 0.3, 0.7)
        rotation_range = trial.suggest_int('rotation_range', 5, 20)
        width_shift_range = trial.suggest_float('width_shift_range', 0.05, 0.2)
        height_shift_range = trial.suggest_float('height_shift_range', 0.05, 0.2)
        brightness_range = trial.suggest_float('brightness_range', 0.8, 1.2)
        contrast_range = trial.suggest_float('contrast_range', 0.8, 1.2)
        saturation_range = trial.suggest_float('saturation_range', 0.8, 1.2)
        hue_range = trial.suggest_float('hue_range', 0.0, 0.1)
        zoom_range = trial.suggest_float('zoom_range', 0.8, 1.2)
        horizontal_flip_prob = trial.suggest_float('horizontal_flip_prob', 0.0, 0.5)
        vertical_flip_prob = trial.suggest_float('vertical_flip_prob', 0.0, 0.3)
        
        # Параметры focal loss
        gamma = trial.suggest_float('gamma', 0.5, 5.0)
        alpha = trial.suggest_float('alpha', 0.05, 0.5)
        
        # Создаем загрузчики данных
        train_loader = VideoDataLoader(
            Config.TRAIN_DATA_PATH,
            max_videos=None
        )
        
        val_loader = VideoDataLoader(
            Config.VALID_DATA_PATH,
            max_videos=None
        )
        
        # Создаем аугментатор с оптимизированными параметрами
        augmenter = VideoAugmenter(
            augment_probability=augment_probability,
            rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            brightness_range=brightness_range,
            contrast_range=contrast_range,
            saturation_range=saturation_range,
            hue_range=hue_range,
            zoom_range=zoom_range,
            horizontal_flip=horizontal_flip_prob,
            vertical_flip=vertical_flip_prob
        )
        
        # Создаем пайплайны данных
        train_data = create_data_pipeline(
            train_loader,
            Config.SEQUENCE_LENGTH,
            Config.BATCH_SIZE,
            Config.INPUT_SIZE,
            one_hot=True,
            infinite_loop=True,
            max_sequences_per_video=None,
            is_train=True,
            force_positive=True,
            augmenter=augmenter
        )
        
        val_data = create_data_pipeline(
            val_loader,
            Config.SEQUENCE_LENGTH,
            Config.BATCH_SIZE,
            Config.INPUT_SIZE,
            one_hot=True,
            infinite_loop=False,
            max_sequences_per_video=None,
            is_train=False,
            force_positive=True
        )
        
        # Создаем и компилируем модель
        model = create_mobilenetv3_model(
            input_shape=(Config.SEQUENCE_LENGTH, *Config.INPUT_SIZE, 3),
            num_classes=Config.NUM_CLASSES,
            dropout_rate=dropout_rate,
            lstm_units=lstm_units
        )
        
        # Создаем метрики
        metrics = [
            'accuracy',
            tf.keras.metrics.Precision(name='precision_element', class_id=1, thresholds=0.5),
            tf.keras.metrics.Recall(name='recall_element', class_id=1, thresholds=0.5),
            tf.keras.metrics.AUC(name='auc')
        ]
        
        # Создаем адаптер для F1Score
        class F1ScoreAdapter(tf.keras.metrics.F1Score):
            def update_state(self, y_true, y_pred, sample_weight=None):
                y_true = tf.argmax(y_true, axis=-1)
                y_pred = tf.argmax(y_pred, axis=-1)
                y_true = tf.reshape(y_true, [-1])
                y_pred = tf.reshape(y_pred, [-1])
                y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=2)
                y_pred = tf.one_hot(tf.cast(y_pred, tf.int32), depth=2)
                return super().update_state(y_true, y_pred, sample_weight)
            
            def result(self):
                result = super().result()
                return tf.reduce_mean(result)
        
        # Добавляем F1Score в метрики
        metrics.append(F1ScoreAdapter(name='f1_score_element', threshold=0.5))
        
        # Компилируем модель
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=focal_loss(gamma=gamma, alpha=alpha),
            metrics=metrics
        )
        
        # Создаем колбэки
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_f1_score_element',
                patience=Config.OVERFITTING_PREVENTION['early_stopping_patience'],
                restore_best_weights=True,
                mode='max'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_f1_score_element',
                factor=Config.OVERFITTING_PREVENTION['reduce_lr_factor'],
                patience=Config.OVERFITTING_PREVENTION['reduce_lr_patience'],
                min_lr=Config.OVERFITTING_PREVENTION['min_lr'],
                mode='max'
            ),
            AdaptiveThresholdCallback(validation_data=(val_data[0], val_data[1]))
        ]
        
        # Обучаем модель
        history = model.fit(
            train_data,
            epochs=Config.EPOCHS,
            validation_data=val_data,
            callbacks=callbacks,
            verbose=1
        )
        
        # Получаем лучший F1-score
        best_f1 = max(history.history['val_f1_score_element'])
        
        # Очищаем память после обучения
        clear_memory()
        
        return best_f1
        
    except Exception as e:
        print(f"[ERROR] Ошибка в испытании {trial.number}: {str(e)}")
        print("[DEBUG] Полный стек ошибки:")
        import traceback
        traceback.print_exc()
        return float('-inf')

def save_tuning_results(study, total_time, n_trials):
    """
    Сохранение результатов подбора гиперпараметров
    """
    try:
        print("\n[DEBUG] Сохранение результатов подбора гиперпараметров...")
        
        # Создаем директорию для результатов
        tuning_dir = os.path.join(Config.MODEL_SAVE_PATH, 'tuning')
        os.makedirs(tuning_dir, exist_ok=True)
        
        # Загружаем базовые веса из конфигурации
        if os.path.exists(Config.CONFIG_PATH):
            with open(Config.CONFIG_PATH, 'r') as f:
                config = json.load(f)
                base_weight = config['MODEL_PARAMS'][Config.MODEL_TYPE]['positive_class_weight']
        else:
            base_weight = None
        
        # Сохраняем результаты в текстовый файл
        with open(os.path.join(tuning_dir, 'optuna_results.txt'), 'w') as f:
            f.write(f"Время выполнения: {total_time:.2f} секунд\n")
            f.write(f"Количество триалов: {n_trials}\n")
            if base_weight:
                f.write(f"Базовый вес положительного класса: {base_weight}\n")
            f.write("\n")
            
            # Получаем лучший триал
            best_trial = study.best_trial
            f.write(f"Лучший триал: {best_trial.number}\n")
            f.write(f"Лучшее значение: {best_trial.value}\n")
            f.write("\nПараметры лучшего триала:\n")
            for key, value in best_trial.params.items():
                f.write(f"{key}: {value}\n")
            
            # Сохраняем историю всех триалов
            f.write("\nИстория всех триалов:\n")
            for trial in study.trials:
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    f.write(f"\nТриал {trial.number}:\n")
                    f.write(f"Значение: {trial.value}\n")
                    f.write("Параметры:\n")
                    for key, value in trial.params.items():
                        f.write(f"{key}: {value}\n")
=======
                file_path = os.path.join(self.data_path, file_name)
                if file_name.endswith('.mp4') and os.path.isfile(file_path):
                    self.video_paths.append(file_path)
                    base = os.path.splitext(file_name)[0]
                    ann_path = os.path.join(annotation_dir, base + '.json')
>>>>>>> 14f4b73b718e50c32cc5e9db2625586f89c8a60c
                    
                    if os.path.exists(ann_path):
                        pass
                        # print(f"[DEBUG] Найдена аннотация для {file_name}")
                    else:
                        print(f"[DEBUG] Аннотация для {file_name} не найдена")
                    
                    self.labels.append(ann_path if os.path.exists(ann_path) else None)
                    self.video_count += 1  # Увеличиваем счетчик
            
            self.video_count = len(self.video_paths)
            
            print(f"[DEBUG] Загружено {self.video_count} видео файлов")
            
            # Ограничиваем количество видео до Config.MAX_VIDEOS
            if hasattr(Config, "MAX_VIDEOS") and len(self.video_paths) > Config.MAX_VIDEOS:
                print(f"[DEBUG] Ограничиваем количество видео до {Config.MAX_VIDEOS}")
                self.video_paths = self.video_paths[:Config.MAX_VIDEOS]
                self.labels = self.labels[:Config.MAX_VIDEOS]
            
        except Exception as e:
            print(f"[ERROR] Ошибка при загрузке данных: {str(e)}")
            print("[DEBUG] Stack trace:", flush=True)
            import traceback
            traceback.print_exc()
            raise
    
    def load_video(self, video_path):
        """Загрузка видео с оптимизацией памяти и подробным логированием"""
        try:
            print(f"[DEBUG] Загрузка видео: {os.path.basename(video_path)}")
            
            # Проверяем существование файла
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Видеофайл не найден: {video_path}")
            
            # Проверяем размер файла
            file_size = os.path.getsize(video_path)
            print(f"[DEBUG] Размер файла: {file_size / (1024*1024):.2f} MB")
            
            # Открываем видео с таймаутом
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Не удалось открыть видео: {video_path}")
            
            print("[DEBUG] Видео успешно открыто")
            
            # Получаем информацию о видео с проверкой каждого свойства
            try:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                print(f"[DEBUG] Ширина: {width}")
            except Exception as e:
                print(f"[ERROR] Ошибка при получении ширины: {str(e)}")
                width = 0
            
            try:
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"[DEBUG] Высота: {height}")
            except Exception as e:
                print(f"[ERROR] Ошибка при получении высоты: {str(e)}")
                height = 0
            
            try:
                fps = cap.get(cv2.CAP_PROP_FPS)
                print(f"[DEBUG] FPS: {fps}")
            except Exception as e:
                print(f"[ERROR] Ошибка при получении FPS: {str(e)}")
                fps = 0
            
            try:
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                print(f"[DEBUG] Количество кадров: {total_frames}")
            except Exception as e:
                print(f"[ERROR] Ошибка при получении количества кадров: {str(e)}")
                total_frames = 0
            
            # Проверяем корректность полученных данных
            if width <= 0 or height <= 0 or fps <= 0 or total_frames <= 0:
                raise ValueError(f"Некорректные параметры видео: width={width}, height={height}, fps={fps}, frames={total_frames}")
            
            print(f"[DEBUG] Видео успешно загружено:")
            print(f"  - Размер: {width}x{height}")
            print(f"  - FPS: {fps}")
            print(f"  - Количество кадров: {total_frames}")
            
            return cap, total_frames
            
        except Exception as e:
            print(f"[ERROR] Ошибка при загрузке видео: {str(e)}")
            if 'cap' in locals():
                cap.release()
            raise
    
    def get_batch(self, batch_size, sequence_length, target_size, one_hot=True, max_sequences_per_video=None, force_positive=False):
        """Получение батча данных с опциональным sampling положительных примеров и подробным debug-логом"""
        try:
            if self.current_video_index >= len(self.video_paths):
                print(f"[DEBUG] get_batch: current_video_index >= len(video_paths), сбрасываем индекс")
                self.current_video_index = 0
                self.current_frame_index = 0
                return None
            
            video_path = self.video_paths[self.current_video_index]
            print(f"[DEBUG] get_batch: Начинаем обработку видео {video_path}")
            cap, total_frames = self.load_video(video_path)
            
            # Проверяем, не достигли ли мы конца видео
            if self.current_frame_index >= total_frames:
                print(f"[DEBUG] get_batch: Достигнут конец видео {video_path}")
                self.current_video_index += 1
                self.current_frame_index = 0
                cap.release()
                return None
            
            # Устанавливаем текущую позицию в видео
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_index)
            
            # Загружаем аннотации
            annotations = self.labels[self.current_video_index]
            if annotations is not None:
                with open(annotations, 'r') as f:
                    ann_data = json.load(f)
                    frame_labels = np.zeros((total_frames, Config.NUM_CLASSES), dtype=np.float32)
                    for annotation in ann_data['annotations']:
                        start_frame = annotation['start_frame']
                        end_frame = annotation['end_frame']
                        for frame_idx in range(start_frame, end_frame + 1):
                            if frame_idx < len(frame_labels):
                                if frame_idx == start_frame:
                                    frame_labels[frame_idx] = [1, 0]
                                elif frame_idx == end_frame:
                                    frame_labels[frame_idx] = [0, 1]
                                else:
                                    frame_labels[frame_idx] = [0, 0]
            else:
                frame_labels = np.zeros((total_frames, Config.NUM_CLASSES), dtype=np.float32)
            
            batch_sequences = []
            batch_labels = []
            used_indices = set()
            batches_for_this_video = 0
            
            # --- Кэширование индексов положительных кадров ---
            if video_path not in self.positive_indices_cache:
                positive_indices = np.where(np.any(frame_labels == 1, axis=1))[0]
                self.positive_indices_cache[video_path] = positive_indices
            else:
                positive_indices = self.positive_indices_cache[video_path]
            
            # --- Новый sampling: гарантированное наличие положительных примеров ---
            if force_positive and len(positive_indices) > 0:
                num_positive = max(1, batch_size // 4)
                print(f"[DEBUG] get_batch: Добавляем {num_positive} положительных последовательностей")
                
                selected_pos_indices = np.random.choice(positive_indices, size=min(num_positive, len(positive_indices)), replace=False)
                
                for pos_idx in selected_pos_indices:
                    start_idx = max(0, pos_idx - sequence_length // 2)
                    end_idx = min(total_frames, start_idx + sequence_length)
                    start_idx = end_idx - sequence_length
                    
                    if start_idx >= 0 and end_idx <= total_frames:
                        print(f"[DEBUG] get_batch: Добавляем положительную последовательность с кадра {start_idx} по {end_idx} (pos_idx={pos_idx})")
                        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
                        frames = []
                        labels = []
                        for i in range(start_idx, end_idx):
                            ret, frame = cap.read()
                            if not ret:
                                print(f"[DEBUG] get_batch: Не удалось прочитать кадр {i}")
                                break
                            frame = cv2.resize(frame, target_size)
                            frames.append(frame)
                            labels.append(frame_labels[i])
                        if len(frames) == sequence_length:
                            batch_sequences.append(frames)
                            batch_labels.append(labels)
                            used_indices.update(range(start_idx, end_idx))
                            batches_for_this_video += 1
                            # Очищаем память после каждой последовательности
                            del frames
                            del labels
                            gc.collect()
            
            # --- Добавляем обычные последовательности ---
            unreadable_frames_count = 0  # Счетчик нечитаемых кадров
            while len(batch_sequences) < batch_size:
                if self.current_frame_index + sequence_length > total_frames:
                    print(f"[DEBUG] get_batch: Достигнут конец видео {video_path}")
                    self.current_video_index += 1
                    self.current_frame_index = 0
                    cap.release()
                    if len(batch_sequences) > 0:
                        print(f"[WARNING] Не удалось собрать полный батч. Получено последовательностей: {len(batch_sequences)}")
                        return None
                    return None
                
                if any(idx in used_indices for idx in range(self.current_frame_index, self.current_frame_index + sequence_length)):
                    self.current_frame_index += 1
                    continue
                
                frames = []
                labels = []
                start_frame = self.current_frame_index
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                
                # Читаем кадры до достижения нужной длины последовательности
                while len(frames) < sequence_length and self.current_frame_index < total_frames:
                    ret, frame = cap.read()
                    if not ret:
                        print(f"[DEBUG] get_batch: Не удалось прочитать кадр {self.current_frame_index}")
                        unreadable_frames_count += 1
                        self.current_frame_index += 1
                        
                        # Если слишком много нечитаемых кадров, пропускаем видео
                        if unreadable_frames_count > 120:
                            print(f"[WARNING] Слишком много нечитаемых кадров ({unreadable_frames_count}), пропускаем видео")
                            self.current_video_index += 1
                            self.current_frame_index = 0
                            cap.release()
                            return None
                        continue
                    
                    frame = cv2.resize(frame, target_size)
                    frames.append(frame)
                    labels.append(frame_labels[self.current_frame_index])
                    self.current_frame_index += 1
                
                # Если удалось собрать последовательность нужной длины
                if len(frames) == sequence_length:
                    # Проверяем наличие положительных примеров в последовательности
                    sequence_labels = np.array(labels)
                    has_positive = np.any(sequence_labels[:, 1] == 1)
                    
                    if has_positive:
                        print(f"[DEBUG] get_batch: Найдена положительная последовательность в обычных примерах")
                    
                    batch_sequences.append(frames)
                    batch_labels.append(labels)
                    batches_for_this_video += 1
                    # Очищаем память после каждой последовательности
                    del frames
                    del labels
                    gc.collect()
                else:
                    print(f"[DEBUG] get_batch: Не удалось собрать последовательность нужной длины. Получено кадров: {len(frames)}")
                    # Очищаем память
                    del frames
                    del labels
                    gc.collect()
            
            if len(batch_sequences) != batch_size:
                print(f"[WARNING] Не удалось собрать полный батч. Получено последовательностей: {len(batch_sequences)}")
                print(f"[DEBUG] get_batch: Для видео {video_path} собрано {batches_for_this_video} батчей")
                return None
            
            # После успешного формирования батча обновляем индекс кадра
            if len(batch_sequences) == batch_size:
                print(f"[DEBUG] get_batch: Батч успешно собран. batch_sequences={len(batch_sequences)}")
                print(f"[DEBUG] get_batch: Для видео {video_path} собрано {batches_for_this_video} батчей")
                print(f"[DEBUG] get_batch: Текущий индекс видео: {self.current_video_index}, текущий индекс кадра: {self.current_frame_index}")
                
                # Проверяем наличие положительных примеров в батче
                positive_in_batch = [np.any(np.array(lbl)[:,1] == 1) for lbl in batch_labels]
                num_positive = sum(positive_in_batch)
                positive_indices = [i for i, v in enumerate(positive_in_batch) if v]
                print(f"[DEBUG] В батче положительных примеров (class 1): {num_positive}")
                print(f"[DEBUG] Индексы последовательностей с положительным примером в батче: {positive_indices}")
                if num_positive > 0:
                    print(f"[DEBUG] Распределение положительных примеров по кадрам:")
                    for idx in positive_indices:
                        positive_frames = np.where(np.array(batch_labels[idx])[:,1] == 1)[0]
                        print(f"  - Последовательность {idx}: кадры {positive_frames.tolist()}")
                
                # Конвертируем в numpy массивы с оптимизированным типом данных
                X = np.array(batch_sequences, dtype=np.float32) / 255.0
                y = np.array(batch_labels, dtype=np.float32)
                
                # Очищаем память
                del batch_sequences
                del batch_labels
                gc.collect()
                
                # Обновляем индекс кадра после формирования батча
                self.current_frame_index += sequence_length
                
                print(f"[DEBUG] get_batch: Прогресс обработки видео: {self.current_frame_index}/{total_frames} кадров")
                
                if self.current_frame_index >= total_frames:
                    print(f"[DEBUG] get_batch: Достигнут конец видео {video_path}, переходим к следующему")
                    self.current_video_index += 1
                    self.current_frame_index = 0
                    cap.release()
                    if self.current_video_index >= len(self.video_paths):
                        print(f"[DEBUG] get_batch: Обработаны все видео")
                        self.current_video_index = 0
                        return None
                
                return X, y
            else:
                print(f"[WARNING] Не удалось собрать полный батч. Получено последовательностей: {len(batch_sequences)}")
                print(f"[DEBUG] get_batch: Для видео {video_path} собрано {batches_for_this_video} батчей")
                return None
            
        except Exception as e:
            print(f"[ERROR] Ошибка при получении батча: {str(e)}")
            print("[DEBUG] Stack trace:", flush=True)
            import traceback
            traceback.print_exc()
            return None
    
    def create_sequences(self, frames, annotations):
        """Создание последовательностей с оптимизацией памяти"""
        try:
            sequences = []
            labels = []
            
            # Очищаем память перед созданием последовательностей
            gc.collect()
            
            # Проверяем, что аннотации существуют
            if annotations is None:
                print("[WARNING] Аннотации не найдены, создаем пустые метки")
                annotations = np.zeros((len(frames), Config.NUM_CLASSES), dtype=np.float32)
            else:
                # Загружаем аннотации из JSON файла
                try:
                    with open(annotations, 'r') as f:
                        ann_data = json.load(f)
                        # Создаем массив меток для каждого кадра
                        frame_labels = np.zeros((len(frames), Config.NUM_CLASSES), dtype=np.float32)
                        
                        # Обрабатываем каждую аннотацию
                        for annotation in ann_data['annotations']:
                            start_frame = annotation['start_frame']
                            end_frame = annotation['end_frame']
                            
                            print(f"[DEBUG] Аннотация: начало кадра {start_frame}, конец кадра {end_frame}")
                            
                            # Устанавливаем метки для кадров в пределах аннотации
                            for frame_idx in range(start_frame, end_frame + 1):
                                if frame_idx < len(frame_labels):
                                    # [1, 0] для начала элемента
                                    if frame_idx == start_frame:
                                        frame_labels[frame_idx] = [1, 0]
                                    # [0, 1] для конца элемента
                                    elif frame_idx == end_frame:
                                        frame_labels[frame_idx] = [0, 1]
                                    # [0, 0] для промежуточных кадров
                                    else:
                                        frame_labels[frame_idx] = [0, 0]
                        
                        annotations = frame_labels
                        print(f"[DEBUG] Загружены аннотации формы: {annotations.shape}")
                except Exception as e:
                    print(f"[ERROR] Ошибка при загрузке аннотаций: {str(e)}")
                    print("[WARNING] Создаем пустые метки")
                    annotations = np.zeros((len(frames), Config.NUM_CLASSES), dtype=np.float32)
            
            # Проверяем размерности
            if len(annotations) != len(frames):
                print(f"[WARNING] Несоответствие размерностей: frames={len(frames)}, annotations={len(annotations)}")
                # Обрезаем до минимальной длины
                min_len = min(len(frames), len(annotations))
                frames = frames[:min_len]
                annotations = annotations[:min_len]
            
            # Создаем последовательности
            for i in range(0, len(frames) - self.sequence_length + 1, self.sequence_length // 2):
                sequence = frames[i:i + self.sequence_length]
                sequence_labels = annotations[i:i + self.sequence_length]
                
                # Проверяем размерности последовательности
                if len(sequence) == self.sequence_length and len(sequence_labels) == self.sequence_length:
                    sequences.append(sequence)
                    labels.append(sequence_labels)
                
                # Очищаем память каждые 10 последовательностей
                if len(sequences) % 10 == 0:
                    gc.collect()
            
            # Преобразуем в numpy массивы с оптимизированным типом данных
            sequences = np.array(sequences, dtype=np.float32)
            labels = np.array(labels, dtype=np.float32)
            
            print(f"[DEBUG] Создано {len(sequences)} последовательностей")
            print(f"[DEBUG] Форма последовательностей: {sequences.shape}")
            print(f"[DEBUG] Форма меток: {labels.shape}")
            
            return sequences, labels
            
        except Exception as e:
            print(f"[ERROR] Ошибка при создании последовательностей: {str(e)}")
            print("[DEBUG] Stack trace:", flush=True)
            import traceback
            traceback.print_exc()
            raise
    
    def preload_video(self, video_path, target_size):
        """
        Предварительная загрузка видео в отдельном потоке.
        """
        self.load_video(video_path)
    
    def data_generator(self, force_positive=True):
        """Генератор данных с sampling положительных примеров"""
        try:
            print("\n[DEBUG] ===== Запуск генератора данных =====")
            print(f"[DEBUG] Количество видео для обработки: {len(self.video_paths)}")
            while True:
                batch_data = self.get_batch(
                    batch_size=self.batch_size,
                    sequence_length=self.sequence_length,
                    target_size=Config.INPUT_SIZE,
                    one_hot=True,
                    max_sequences_per_video=self.max_sequences_per_video,
                    force_positive=force_positive
                )
                if batch_data is None:
                    print("[DEBUG] Достигнут конец эпохи")
                    break
                
                X, y = batch_data
                if X is None or y is None or X.shape[0] == 0 or y.shape[0] == 0:
                    print("[WARNING] Получен пустой батч")
                    continue
                
                try:
                    num_positive = int((y[...,1] == 1).sum())
                    print(f"[DEBUG] В батче положительных примеров (class 1): {num_positive}")
                    
                    # Конвертируем в тензоры с оптимизацией памяти
                    x = tf.convert_to_tensor(X, dtype=tf.float32)
                    y_tensor = tf.convert_to_tensor(y, dtype=tf.float32)
                    
                    # Очищаем память
                    del X
                    del y
                    gc.collect()
                    
                    yield (x, y_tensor)
                    
                except Exception as e:
                    print(f"[ERROR] Ошибка при обработке батча: {str(e)}")
                    print("[DEBUG] Stack trace:", flush=True)
                    import traceback
                    traceback.print_exc()
                    continue
                
        except Exception as e:
            print(f"[ERROR] Ошибка в генераторе данных: {str(e)}")
            print("[DEBUG] Stack trace:", flush=True)
            import traceback
            traceback.print_exc()
            raise
    
    def load_data(self, sequence_length, batch_size, target_size=None, one_hot=False, infinite_loop=False, max_sequences_per_video=10):
        """
        Загрузка данных для обучения.
        
        Args:
            sequence_length (int): Длина последовательности
            batch_size (int): Размер батча
            target_size (tuple): Размер изображения (ширина, высота)
            one_hot (bool): Использовать one-hot encoding для меток
            infinite_loop (bool): Бесконечный цикл генерации данных
            
        Returns:
            generator: Генератор данных
        """
        return self.data_generator()
    
    def _calculate_total_batches(self):
        """
        Рассчитывает общее количество батчей для данных.
        """
        try:
            print("[DEBUG] Начало расчета общего количества батчей")
            batch_count = 0
            for _ in self.data_generator():
                batch_count += 1
            self.total_batches = batch_count
            print(f"[DEBUG] Рассчитано батчей: {self.total_batches}")
        except Exception as e:
            print(f"[ERROR] Ошибка при расчете количества батчей: {str(e)}")
            print("[DEBUG] Stack trace:", flush=True)
            import traceback
            traceback.print_exc()
            self.total_batches = 0
    
    def get_video_info(self, video_path):
        """
        Получение информации о видео
        
        Args:
            video_path: путь к видео файлу
            
        Returns:
            dict: словарь с информацией о видео (total_frames, fps, width, height)
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Не удалось открыть видео: {video_path}")
            
            # Получаем информацию о видео
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            cap.release()
            
            return {
                'total_frames': total_frames,
                'fps': fps,
                'width': width,
                'height': height
            }
            
        except Exception as e:
            print(f"[ERROR] Ошибка при получении информации о видео {video_path}: {str(e)}")
            raise 
