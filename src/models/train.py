import tensorflow as tf
from src.models.model import create_model_with_params
from src.data_proc.data_loader import VideoDataLoader
from src.config import Config
import os
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    Callback
)
import numpy as np
import gc
from tqdm.auto import tqdm
from tqdm.keras import TqdmCallback
from tensorflow.keras.metrics import Precision, Recall
import json
import re
import psutil
from src.data_proc.data_augmentation import VideoAugmenter
from src.models.losses import focal_loss, F1ScoreAdapter
from src.models.metrics import f1_score_element, get_training_metrics
from src.models.callbacks import AdaptiveThresholdCallback, get_training_callbacks

# Включаем eager execution
tf.config.run_functions_eagerly(True)

# Настройка GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Сначала включаем динамический рост памяти
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth enabled for GPUs")
        
        # Затем настраиваем ограничение памяти
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(
                memory_limit=Config.DEVICE_CONFIG['gpu_memory_limit']
            )]
        )
    except RuntimeError as e:
        print(f"Error setting GPU configuration: {e}")

# Включение mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("GPU Device: ", tf.test.gpu_device_name())

def clear_memory():
    """Очистка памяти"""
    # Очищаем все сессии TensorFlow
    tf.keras.backend.clear_session()
    
    # Очистка Python garbage collector
    gc.collect()
    
    # Очистка CUDA кэша если используется GPU
    if Config.DEVICE_CONFIG['use_gpu']:
        try:
            import numba
            numba.cuda.close()
        except:
            pass

def create_data_pipeline(loader, sequence_length, batch_size, target_size, is_training=True, force_positive=False, cache_dataset=False):
    """
    Создание оптимизированного pipeline данных для обучения и подбора гиперпараметров
    
    Args:
        loader: загрузчик данных
        sequence_length: длина последовательности
        batch_size: размер батча
        target_size: размер кадра
        is_training: флаг обучения
        force_positive: принудительно использовать положительные примеры
        cache_dataset: кэшировать датасет (используется только для небольших наборов данных)
    """
    try:
        print("\n[DEBUG] Создание pipeline данных...")
        print(f"[DEBUG] Параметры:")
        print(f"  - sequence_length: {sequence_length}")
        print(f"  - batch_size: {batch_size}")
        print(f"  - target_size: {target_size}")
        print(f"  - is_training: {is_training}")
        print(f"  - force_positive: {force_positive}")
        print(f"  - cache_dataset: {cache_dataset}")
        print(f"[DEBUG] RAM до создания датасета: {psutil.virtual_memory().used / 1024**3:.2f} GB")

        def generator():
            while True:
                # Получаем следующую последовательность
                X, y = loader._get_sequence(
                    sequence_length=sequence_length,
                    target_size=target_size,
                    force_positive=force_positive
                )
                if X is not None and y is not None:
                    print(f"[DEBUG] Форма входных данных X: {X.shape}")
                    print(f"[DEBUG] Форма меток y: {y.shape}")
                    print(f"[DEBUG] Тип меток y: {type(y)}")
                    print(f"[DEBUG] Тип первого элемента y: {type(y[0])}")
                    if isinstance(y[0], np.ndarray):
                        print(f"[DEBUG] Форма первого элемента y: {y[0].shape}")
                    
                    # Преобразуем метки в one-hot encoding для 2 классов
                    y_one_hot = np.zeros((sequence_length, 2), dtype=np.float32)
                    
                    # Используем правильную индексацию для создания one-hot encoding
                    for i in range(sequence_length):
                        try:
                            # Проверяем, является ли y[i] массивом
                            if isinstance(y[i], np.ndarray):
                                if y[i].size == 1:
                                    label = int(y[i].item())
                                else:
                                    # Если это массив с несколькими элементами, берем первый
                                    label = int(y[i][0])
                            else:
                                label = int(y[i])
                            
                            y_one_hot[i, label] = 1
                            print(f"[DEBUG] Обработка метки {i}: исходное значение = {y[i]}, преобразованное = {label}")
                        except Exception as e:
                            print(f"[ERROR] Ошибка при обработке метки {i}: {str(e)}")
                            print(f"[DEBUG] Значение y[{i}]: {y[i]}")
                            print(f"[DEBUG] Тип y[{i}]: {type(y[i])}")
                            if isinstance(y[i], np.ndarray):
                                print(f"[DEBUG] Форма y[{i}]: {y[i].shape}")
                            raise
                    
                    print(f"[DEBUG] Итоговая форма one-hot encoding: {y_one_hot.shape}")
                    print(f"[DEBUG] Сумма меток в one-hot encoding: {np.sum(y_one_hot)}")
                    yield X, y_one_hot

        # Создаем dataset напрямую из генератора
        output_signature = (
            tf.TensorSpec(shape=(sequence_length, *target_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(sequence_length, 2), dtype=tf.float32)  # two-hot encoding для 2 классов
        )

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=output_signature
        )

        # Оптимизация производительности
        if cache_dataset and (not hasattr(loader, 'video_count') or loader.video_count <= 50):
            dataset = dataset.cache()
        if is_training:
            dataset = dataset.shuffle(64)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        print(f"[DEBUG] RAM после создания датасета: {psutil.virtual_memory().used / 1024**3:.2f} GB")
        print("[DEBUG] Pipeline данных успешно создан")
        return dataset

    except Exception as e:
        print(f"[ERROR] Ошибка при создании pipeline данных: {str(e)}")
        print("[DEBUG] Stack trace:", flush=True)
        import traceback
        traceback.print_exc()
        raise

class OverfittingMonitor(Callback):
    """Мониторинг переобучения"""
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold
        self.overfitting_epochs = 0
        
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
            
        train_acc = logs.get('accuracy')
        val_acc = logs.get('val_accuracy')
        
        if train_acc is not None and val_acc is not None:
            diff = train_acc - val_acc
            if diff > self.threshold:
                self.overfitting_epochs += 1
                print(f"\nWarning: Possible overfitting detected! "
                      f"Train-Val accuracy difference: {diff:.4f}")
            else:
                self.overfitting_epochs = 0

class TrainingPlotter(Callback):
    def __init__(self, save_path):
        super().__init__()
        self.save_path = save_path
        self.epochs = []
        self.losses = []
        self.val_losses = []
        self.accuracies = []
        self.val_accuracies = []
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
    def on_epoch_end(self, epoch, logs=None):
        self.epochs.append(epoch)
        self.losses.append(logs['loss'])
        self.val_losses.append(logs['val_loss'])
        self.accuracies.append(logs['accuracy'])
        self.val_accuracies.append(logs['val_accuracy'])
        
        # Очищаем предыдущие графики
        self.ax1.clear()
        self.ax2.clear()
        
        # График потерь
        self.ax1.plot(self.epochs, self.losses, label='Training Loss')
        self.ax1.plot(self.epochs, self.val_losses, label='Validation Loss')
        self.ax1.set_title('Loss')
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Loss')
        self.ax1.legend()
        self.ax1.grid(True)
        
        # График точности
        self.ax2.plot(self.epochs, self.accuracies, label='Training Accuracy')
        self.ax2.plot(self.epochs, self.val_accuracies, label='Validation Accuracy')
        self.ax2.set_title('Accuracy')
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('Accuracy')
        self.ax2.legend()
        self.ax2.grid(True)
        
        # Сохраняем графики
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, 'training_plot.png'))
        plt.close()

def plot_confusion_matrix(y_true, y_pred, save_path):
    """Визуализация матрицы ошибок"""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))
    plt.close()

def load_best_params(model_type=None):
    """
    Загрузка лучших параметров из файла optuna_results.txt и весов из config_weights.json
    Args:
        model_type: тип модели ('v3' или 'v4'). Если None, используется Config.MODEL_TYPE
    """
    try:
        model_type = model_type or Config.MODEL_TYPE
        results_dir = os.path.join(Config.MODEL_SAVE_PATH, 'tuning')
        results_path = os.path.join(results_dir, 'optuna_results.txt')
        
        # Проверяем существование директории
        if not os.path.exists(results_dir):
            print(f"[DEBUG] Создание директории для результатов: {results_dir}")
            os.makedirs(results_dir, exist_ok=True)
        
        # Загружаем веса из конфигурационного файла
        if os.path.exists(Config.CONFIG_PATH):
            print(f"[DEBUG] Загрузка весов из {Config.CONFIG_PATH}")
            with open(Config.CONFIG_PATH, 'r') as f:
                config = json.load(f)
                positive_class_weight = config['MODEL_PARAMS'][model_type]['positive_class_weight']
                print(f"[DEBUG] Загружен вес положительного класса: {positive_class_weight}")
        else:
            print(f"[WARNING] Конфигурационный файл не найден: {Config.CONFIG_PATH}")
            positive_class_weight = None
        
        if not os.path.exists(results_path):
            print(f"[DEBUG] Файл с результатами подбора гиперпараметров не найден. Используем параметры по умолчанию для {model_type}.")
            default_params = {
                'learning_rate': 1e-4,
                'dropout_rate': Config.MODEL_PARAMS[model_type]['dropout_rate'],
                'batch_size': Config.BATCH_SIZE,  # Используем значение из конфига
                'positive_class_weight': positive_class_weight
            }
            if model_type == 'v3':
                default_params['lstm_units'] = Config.MODEL_PARAMS[model_type]['lstm_units']
            return default_params
        
        print(f"[DEBUG] Загрузка параметров из {results_path}")
        with open(results_path, 'r') as f:
            content = f.read()
            
            # Ищем лучшие результаты для каждого типа модели
            v3_best = None
            v4_best = None
            
            # Ищем все trials
            trials = re.finditer(r"Trial \d+:\s+Value: ([\d.-]+)\s+Params: ({[^}]+})", content)
            for trial in trials:
                value = float(trial.group(1))
                params_str = trial.group(2).replace("'", '"')
                params = json.loads(params_str)
                
                if params.get('model_type') == 'v3' and (v3_best is None or value > v3_best[0]):
                    v3_best = (value, params)
                elif params.get('model_type') == 'v4' and (v4_best is None or value > v4_best[0]):
                    v4_best = (value, params)
            
            # Выбираем лучшие параметры для запрошенного типа модели
            if model_type == 'v3' and v3_best:
                best_params = v3_best[1]
            elif model_type == 'v4' and v4_best:
                best_params = v4_best[1]
            else:
                print(f"[DEBUG] Не найдены результаты для модели {model_type}. Используем параметры по умолчанию.")
                best_params = {
                    'learning_rate': 1e-4,
                    'dropout_rate': Config.MODEL_PARAMS[model_type]['dropout_rate'],
                    'batch_size': Config.BATCH_SIZE,  # Используем значение из конфига
                    'positive_class_weight': positive_class_weight
                }
                if model_type == 'v3':
                    best_params['lstm_units'] = Config.MODEL_PARAMS[model_type]['lstm_units']
            
            # Добавляем вес положительного класса из конфигурации
            best_params['positive_class_weight'] = positive_class_weight
            
            print(f"[DEBUG] Загружены лучшие параметры для {model_type}: {best_params}")
            return best_params
            
    except Exception as e:
        print(f"[ERROR] Ошибка при загрузке параметров: {str(e)}")
        print("[DEBUG] Stack trace:", flush=True)
        import traceback
        traceback.print_exc()
    
    print(f"[DEBUG] Не удалось загрузить параметры для {model_type}. Используем параметры по умолчанию.")
    default_params = {
        'learning_rate': 1e-4,
        'dropout_rate': Config.MODEL_PARAMS[model_type]['dropout_rate'],
        'batch_size': Config.BATCH_SIZE,  # Используем значение из конфига
        'positive_class_weight': positive_class_weight
    }
    if model_type == 'v3':
        default_params['lstm_units'] = Config.MODEL_PARAMS[model_type]['lstm_units']
    return default_params

def train(model_type: str = 'v4', epochs: int = 50, batch_size: int = Config.BATCH_SIZE):
    """
    Обучение модели
    Args:
        model_type: тип модели ('v3' или 'v4')
        epochs: количество эпох
        batch_size: размер батча (по умолчанию берется из конфига)
    """
    try:
        print("\n[DEBUG] ===== Начало обучения =====")
        print(f"[DEBUG] Тип модели: {model_type}")
        print(f"[DEBUG] Количество эпох: {epochs}")
        print(f"[DEBUG] Размер батча: {batch_size}")

        # Загрузка лучших параметров из подбора гиперпараметров
        best_params = load_best_params(model_type)
        print(f"[DEBUG] Загружены лучшие параметры: {best_params}")

        # Загрузка весов классов из config_weights.json
        with open(Config.CONFIG_PATH, 'r') as f:
            config = json.load(f)
            class_weights = config['class_weights']
        tf_class_weights = {
            0: class_weights['background'],
            1: class_weights['action']
        }

        # Создаем загрузчики данных без ограничения на количество видео
        train_loader = VideoDataLoader(
            Config.TRAIN_DATA_PATH,
            max_videos=None  # Убираем ограничение
        )
        val_loader = VideoDataLoader(
            Config.VALID_DATA_PATH,
            max_videos=None  # Убираем ограничение
        )
        
        # Создаем пайплайны данных
        train_data = create_data_pipeline(
            train_loader,
            Config.SEQUENCE_LENGTH,
            batch_size,
            Config.TARGET_SIZE,
            is_training=True,
            force_positive=True,
            cache_dataset=True
        )
        val_data = create_data_pipeline(
            val_loader,
            Config.SEQUENCE_LENGTH,
            batch_size,
            Config.TARGET_SIZE,
            is_training=False,
            force_positive=False,
            cache_dataset=True
        )
        
        # Создаем и компилируем модель с лучшими параметрами
        model = create_model_with_params(
            model_type=model_type,
            input_shape=(Config.SEQUENCE_LENGTH, *Config.TARGET_SIZE, 3),
            num_classes=Config.NUM_CLASSES,
            params={
                'dropout_rate': best_params['dropout_rate'],
                'lstm_units': best_params.get('lstm_units', Config.MODEL_PARAMS[model_type].get('lstm_units', 128)),
                'rnn_type': best_params.get('rnn_type', 'lstm'),
                'temporal_block_type': best_params.get('temporal_block_type', 'rnn')
            },
            class_weights=class_weights
        )
        
        # Компилируем модель с лучшими параметрами
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=best_params['learning_rate'],
                clipnorm=best_params.get('clipnorm', 1.0)
            ),
            loss=focal_loss(gamma=Config.FOCAL_LOSS['gamma'], alpha=Config.FOCAL_LOSS['alpha']),
            metrics=get_training_metrics()
        )
        
        # Создаем callbacks
        callbacks = get_training_callbacks(val_data)
        
        # Обучаем модель
        history = model.fit(
            train_data,
            epochs=epochs,
            validation_data=val_data,
            callbacks=callbacks,
            class_weight=tf_class_weights,
            verbose=1
        )
        return model, history
    except Exception as e:
        print(f"[ERROR] Ошибка при обучении модели: {str(e)}")
        print("[DEBUG] Stack trace:", flush=True)
        import traceback
        traceback.print_exc()
        raise

def plot_training_results(history, save_path):
    """
    Визуализация результатов обучения
    """
    try:
        print("\n[DEBUG] Визуализация результатов обучения...")
        
        # Валидация входных параметров
        if not isinstance(history, tf.keras.callbacks.History):
            raise ValueError("history должен быть экземпляром tf.keras.callbacks.History")
            
        if not os.path.exists(save_path):
            raise ValueError(f"Директория не существует: {save_path}")
        
        # Создаем директорию для графиков
        plot_path = os.path.join(save_path, 'plots')
        os.makedirs(plot_path, exist_ok=True)
        
        # Графики потерь и точности
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_path, 'training_metrics.png'))
        plt.close()
        
        # График F1-score
        plt.figure(figsize=(6, 4))
        plt.plot(history.history['f1_score_element'], label='Training F1-score')
        plt.plot(history.history['val_f1_score_element'], label='Validation F1-score')
        plt.title('F1-score')
        plt.xlabel('Epoch')
        plt.ylabel('F1-score')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_path, 'f1_score.png'))
        plt.close()
        
        print("[DEBUG] Результаты обучения успешно визуализированы")
        
    except Exception as e:
        print(f"[ERROR] Ошибка при визуализации результатов: {str(e)}")
        print("[DEBUG] Stack trace:", flush=True)
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    # Обучаем обе модели
    print("Обучение MobileNetV3...")
    model_v3, history_v3 = train('v3')
    
    print("\nОбучение MobileNetV4...")
    model_v4, history_v4 = train('v4') 