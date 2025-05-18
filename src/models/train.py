import tensorflow as tf
from src.models.model import create_model
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
from src.data_proc.data_augmentation import VideoAugmenter, focal_loss, AdaptiveThresholdCallback

# Включаем eager execution
tf.config.run_functions_eagerly(True)

# Настройка GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Ограничиваем память GPU
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(
                memory_limit=Config.DEVICE_CONFIG['gpu_memory_limit']
            )]
        )
        
        # Включаем динамический рост памяти
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth enabled for GPUs")
    except RuntimeError as e:
        print(f"Error setting memory growth: {e}")

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

def create_data_pipeline(loader, sequence_length, batch_size, target_size, one_hot=True, infinite_loop=True, max_sequences_per_video=None, is_train=True, force_positive=True):
    """
    Создание pipeline данных для обучения
    
    Args:
        loader: загрузчик данных
        sequence_length: длина последовательности
        batch_size: размер батча
        target_size: размер кадра
        one_hot: использовать one-hot encoding
        infinite_loop: бесконечный цикл
        max_sequences_per_video: максимальное количество последовательностей на видео
        is_train: флаг обучения
        force_positive: принудительно использовать положительные примеры
    """
    try:
        print("[DEBUG] Создание pipeline данных")
        print(f"[DEBUG] RAM до создания датасета: {psutil.virtual_memory().used / 1024**3:.2f} GB")
        
        # Создаем dataset напрямую из генератора
        output_signature = (
            tf.TensorSpec(shape=(None, sequence_length, *target_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, sequence_length, 3), dtype=tf.float32)  # three-hot encoding
        )
        
        dataset = tf.data.Dataset.from_generator(
            lambda: loader.data_generator(force_positive=force_positive, is_validation=not is_train),
            output_signature=output_signature
        )
        
        # Оптимизация производительности
        if Config.MEMORY_OPTIMIZATION['cache_dataset'] and (not hasattr(loader, 'video_count') or loader.video_count <= 50):
            dataset = dataset.cache()
        if is_train:
            dataset = dataset.shuffle(64)
            dataset = dataset.batch(batch_size, drop_remainder=True)
        else:
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

def f1_score_element(y_true, y_pred):
    """
    Вычисление F1-score для элемента с учетом временной размерности и three-hot encoded меток
    """
    # Объединяем классы действия и перехода в один положительный класс
    y_true_bin = tf.reduce_any(y_true[:, :, 1:], axis=-1)  # Объединяем действие и переход
    y_pred_bin = tf.reduce_any(y_pred[:, :, 1:], axis=-1)  # Объединяем действие и переход
    
    true_positives = tf.reduce_sum(tf.cast((y_true_bin == 1) & (y_pred_bin == 1), tf.float32))
    predicted_positives = tf.reduce_sum(tf.cast(y_pred_bin == 1, tf.float32))
    possible_positives = tf.reduce_sum(tf.cast(y_true_bin == 1, tf.float32))
    
    # Добавляем epsilon для предотвращения деления на ноль
    epsilon = tf.keras.backend.epsilon()
    
    # Вычисляем precision и recall
    precision = true_positives / (predicted_positives + epsilon)
    recall = true_positives / (possible_positives + epsilon)
    
    # Вычисляем F1-score
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    
    return f1

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

def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)
        
        # Загружаем веса классов
        with open(Config.CONFIG_PATH, 'r') as f:
            config = json.load(f)
            class_weights = config['MODEL_PARAMS'][Config.MODEL_TYPE]['class_weights']
        
        # Применяем веса к каждому классу
        weights = tf.constant([
            class_weights['background'],
            class_weights['action'],
            class_weights['transition']
        ])
        
        # Вычисляем focal loss с весами
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        # Вычисляем веса для каждого класса
        alpha_weight = alpha * y_true + (1 - alpha) * (1 - y_true)
        
        # Вычисляем фокусный вес
        pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = tf.pow(1 - pt, gamma)
        
        # Вычисляем кросс-энтропию
        cross_entropy = -y_true * tf.math.log(y_pred)
        
        # Применяем веса классов и фокусный вес
        loss = alpha_weight * focal_weight * cross_entropy * weights
        
        return tf.reduce_mean(loss)
    return focal_loss_fixed

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

        # Загрузка весов классов из config_weights.json
        with open(Config.CONFIG_PATH, 'r') as f:
            config = json.load(f)
            class_weights = config['class_weights']
        tf_class_weights = {
            0: class_weights['background'],
            1: class_weights['action'],
            2: class_weights['transition']
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
            one_hot=True,
            infinite_loop=True,
            max_sequences_per_video=None,  # Убираем ограничение
            is_train=True,
            force_positive=True
        )
        val_data = create_data_pipeline(
            val_loader,
            Config.SEQUENCE_LENGTH,
            batch_size,
            Config.TARGET_SIZE,
            one_hot=True,
            infinite_loop=True,
            max_sequences_per_video=None,  # Убираем ограничение
            is_train=False,
            force_positive=False
        )
        # Создаем аугментатор
        augmenter = VideoAugmenter(augment_probability=0.5)
        # Создаем и компилируем модель
        if model_type == 'v3':
            model, _ = create_mobilenetv3_model(
                input_shape=(Config.SEQUENCE_LENGTH, *Config.TARGET_SIZE, 3),
                num_classes=Config.NUM_CLASSES,
                dropout_rate=best_params['dropout_rate'],
                lstm_units=best_params['lstm_units'],
                class_weights=class_weights
            )
        else:
            model, _ = create_mobilenetv4_model(
                input_shape=(Config.SEQUENCE_LENGTH, *Config.TARGET_SIZE, 3),
                num_classes=Config.NUM_CLASSES,
                dropout_rate=best_params['dropout_rate'],
                class_weights=class_weights
            )
        # Создаем метрики
        metrics = [
            'accuracy',
            tf.keras.metrics.Precision(name='precision_element', class_id=1, thresholds=0.5),
            tf.keras.metrics.Recall(name='recall_element', class_id=1, thresholds=0.5),
            tf.keras.metrics.AUC(name='auc')  # Добавляем AUC
        ]
        
        # Создаем адаптер для F1Score
        class F1ScoreAdapter(tf.keras.metrics.F1Score):
            def update_state(self, y_true, y_pred, sample_weight=None):
                y_true = tf.argmax(y_true, axis=-1)
                y_pred = tf.argmax(y_pred, axis=-1)
                y_true = tf.reshape(y_true, [-1])
                y_pred = tf.reshape(y_pred, [-1])
                y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=3)  # 3 класса
                y_pred = tf.one_hot(tf.cast(y_pred, tf.int32), depth=3)  # 3 класса
                return super().update_state(y_true, y_pred, sample_weight)
            
            def result(self):
                # Получаем результат от родительского класса
                result = super().result()
                # Возвращаем среднее значение по всем классам
                return tf.reduce_mean(result)
        
        # Добавляем F1Score в метрики
        metrics.append(F1ScoreAdapter(name='f1_score_element', threshold=0.5))
        
        # Компилируем модель с focal loss
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=focal_loss(gamma=Config.FOCAL_LOSS['gamma'], alpha=Config.FOCAL_LOSS['alpha']),
            metrics=metrics
        )
        
        # Создаем колбэки
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_f1_score_element',
                patience=Config.OVERFITTING_PREVENTION['early_stopping_patience'],
                restore_best_weights=True,
                mode='max'  # Явно указываем режим максимизации
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_f1_score_element',
                factor=Config.OVERFITTING_PREVENTION['reduce_lr_factor'],
                patience=Config.OVERFITTING_PREVENTION['reduce_lr_patience'],
                min_lr=Config.OVERFITTING_PREVENTION['min_lr'],
                mode='max'  # Явно указываем режим максимизации
            ),
            AdaptiveThresholdCallback(validation_data=(val_data[0], val_data[1]))  # Добавляем адаптивный порог
        ]
        
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