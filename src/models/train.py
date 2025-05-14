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

def create_data_pipeline(loader, sequence_length, batch_size, target_size, one_hot=True, infinite_loop=True, max_sequences_per_video=None, is_train=True):
    """
    Создание оптимизированного pipeline данных
    """
    try:
        print("\n[DEBUG] Создание pipeline данных...")
        print(f"[DEBUG] RAM до создания датасета: {psutil.virtual_memory().used / 1024**3:.2f} GB")
        # Проверка VideoDataLoader на загрузку всех видео
        if hasattr(loader, 'video_count') and loader.video_count > 50:
            print(f"[WARNING] VideoDataLoader содержит {loader.video_count} видео. Проверьте, не загружаются ли все видео в память!")
        # Валидация входных параметров
        if not isinstance(loader, VideoDataLoader):
            raise ValueError("loader должен быть экземпляром VideoDataLoader")
            
        if sequence_length <= 0:
            raise ValueError(f"Длина последовательности должна быть положительной: {sequence_length}")
            
        if batch_size <= 0:
            raise ValueError(f"Размер батча должен быть положительным: {batch_size}")
            
        if not isinstance(target_size, tuple) or len(target_size) != 2:
            raise ValueError(f"Неверный формат target_size: {target_size}. Ожидается (height, width)")
            
        if max_sequences_per_video is not None and max_sequences_per_video <= 0:
            raise ValueError(f"Максимальное количество последовательностей должно быть положительным: {max_sequences_per_video}")
        
        # Создаем генератор данных
        def data_generator():
            while True:
                try:
                    # Получаем батч данных
                    batch_data = loader.get_batch(
                        batch_size=batch_size,
                        sequence_length=sequence_length,
                        target_size=target_size,
                        one_hot=one_hot,
                        max_sequences_per_video=max_sequences_per_video
                    )
                    
                    if batch_data is None:
                        print("[WARNING] Получен пустой батч данных")
                        continue
                        
                    X, y = batch_data
                    
                    # Проверяем размерности
                    if X.shape[0] == 0 or y.shape[0] == 0:
                        print("[WARNING] Получен батч с нулевой размерностью")
                        continue
                        
                    yield X, y
                    
                except Exception as e:
                    print(f"[ERROR] Ошибка в генераторе данных: {str(e)}")
                    print("[DEBUG] Stack trace:", flush=True)
                    import traceback
                    traceback.print_exc()
                    continue
                
                if not infinite_loop:
                    break
        
        # Создаем dataset
        output_signature = (
            tf.TensorSpec(shape=(None, sequence_length, *target_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, sequence_length, 2 if one_hot else 1), dtype=tf.float32)
        )
        
        dataset = tf.data.Dataset.from_generator(
            data_generator,
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
    y_true = tf.argmax(y_true, axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1)
    true_positives = tf.reduce_sum(tf.cast((y_true == 1) & (y_pred == 1), tf.float32))
    predicted_positives = tf.reduce_sum(tf.cast(y_pred == 1, tf.float32))
    possible_positives = tf.reduce_sum(tf.cast(y_true == 1, tf.float32))
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
    return 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())

def load_best_params(model_type=None):
    """
    Загрузка лучших параметров из файла optuna_results.txt
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
        
        if not os.path.exists(results_path):
            print(f"[DEBUG] Файл с результатами подбора гиперпараметров не найден. Используем параметры по умолчанию для {model_type}.")
            default_params = {
                'learning_rate': 1e-4,
                'dropout_rate': Config.MODEL_PARAMS[model_type]['dropout_rate'],
                'batch_size': 16
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
                    'batch_size': 16
                }
                if model_type == 'v3':
                    best_params['lstm_units'] = Config.MODEL_PARAMS[model_type]['lstm_units']
            
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
        'batch_size': 16
    }
    if model_type == 'v3':
        default_params['lstm_units'] = Config.MODEL_PARAMS[model_type]['lstm_units']
    return default_params

def train(model_type=None):
    """
    Обучение модели с оптимизацией памяти
    """
    try:
        print("\n[DEBUG] Начало обучения...")
        
        # Валидация конфигурации
        Config.validate()
        
        # Определение типа модели
        model_type = model_type or Config.MODEL_TYPE
        print(f"[DEBUG] Тип модели: {model_type}")
        
        # Создание директории для сохранения модели
        model_save_path = os.path.join('models', f'model_{model_type}')
        os.makedirs(model_save_path, exist_ok=True)
        
        # Получение лучших параметров из Optuna (если есть)
        best_params = load_best_params(model_type)
        print(f"[DEBUG] Используем параметры для обучения: {best_params}")
        
        # Создание модели с учетом лучших параметров
        input_shape = (Config.SEQUENCE_LENGTH,) + Config.INPUT_SIZE + (3,)
        model, class_weights = create_model(
            input_shape=input_shape,
            num_classes=Config.NUM_CLASSES,
            dropout_rate=best_params.get('dropout_rate', Config.MODEL_PARAMS[model_type]['dropout_rate']),
            lstm_units=best_params.get('lstm_units', Config.MODEL_PARAMS[model_type].get('lstm_units', 64)),
            model_type=model_type,
            positive_class_weight=best_params.get('positive_class_weight', 200.0)
        )
        
        # Настройка оптимизатора с mixed precision
        optimizer = tf.keras.optimizers.Adam(learning_rate=best_params.get('learning_rate', Config.LEARNING_RATE))
        if Config.MEMORY_OPTIMIZATION['use_mixed_precision']:
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        
        # Создаем метрики
        print("[DEBUG] Создание метрик...")
        metrics = [
            'accuracy',
            tf.keras.metrics.Precision(name='precision_element', class_id=1),
            tf.keras.metrics.Recall(name='recall_element', class_id=1)
        ]

        print("[DEBUG] Добавление F1Score...")
        try:
            f1_metric = tf.keras.metrics.F1Score(name='f1_score_element', threshold=0.5)
            print(f"[DEBUG] F1Score создан успешно: {f1_metric}")
            metrics.append(f1_metric)
        except Exception as e:
            print(f"[ERROR] Ошибка при создании F1Score: {str(e)}")
            print("[DEBUG] Stack trace:", flush=True)
            import traceback
            traceback.print_exc()

        print(f"[DEBUG] Итоговый список метрик: {metrics}")

        # Компиляция модели
        print("[DEBUG] Компиляция модели...")
        model.compile(
            optimizer=optimizer,
            loss=focal_loss(gamma=2., alpha=0.25),
            metrics=metrics
        )
        print("[DEBUG] Модель успешно скомпилирована")
        
        # Создаем callbacks
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(model_save_path, 'best_model.h5'),
                monitor='val_f1_score_element',
                save_best_only=True,
                mode='max'
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_f1_score_element',
                patience=5,
                restore_best_weights=True,
                mode='max'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_f1_score_element',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                mode='max'
            )
        ]
        
        # Загрузка данных
        train_loader = VideoDataLoader(Config.TRAIN_DATA_PATH)
        val_loader = VideoDataLoader(Config.VALID_DATA_PATH)
        
        # Создание оптимизированных pipeline данных
        train_dataset = create_data_pipeline(
            loader=train_loader,
            sequence_length=Config.SEQUENCE_LENGTH,
            batch_size=best_params.get('batch_size', Config.BATCH_SIZE),
            target_size=Config.INPUT_SIZE,
            one_hot=True,
            infinite_loop=True,
            max_sequences_per_video=Config.MAX_SEQUENCES_PER_VIDEO,
            is_train=True
        )
        
        val_dataset = create_data_pipeline(
            loader=val_loader,
            sequence_length=Config.SEQUENCE_LENGTH,
            batch_size=best_params.get('batch_size', Config.BATCH_SIZE),
            target_size=Config.INPUT_SIZE,
            one_hot=True,
            infinite_loop=False,
            max_sequences_per_video=Config.MAX_SEQUENCES_PER_VIDEO
        )
        
        # Обучение
        history = model.fit(
            train_dataset,
            epochs=Config.EPOCHS,
            validation_data=val_dataset,
            callbacks=callbacks,
            steps_per_epoch=Config.STEPS_PER_EPOCH,
            validation_steps=Config.VALIDATION_STEPS,
            class_weight=class_weights,
            verbose=1
        )
        
        # Сохранение финальной модели
        model.save(os.path.join(model_save_path, 'final_model.h5'))
        
        # Визуализация результатов
        plot_training_results(history, model_save_path)
        
        return model, history
        
    except Exception as e:
        print(f"[ERROR] Ошибка при обучении модели: {str(e)}")
        print("[DEBUG] Stack trace:", flush=True)
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Очистка памяти
        if Config.MEMORY_OPTIMIZATION['clear_memory_after_epoch']:
            clear_memory()

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