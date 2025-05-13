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

def create_data_pipeline(loader, sequence_length, batch_size, target_size, one_hot, infinite_loop, max_sequences_per_video):
    """
    Создает оптимизированный pipeline данных с использованием tf.data.Dataset
    """
    def generator():
        return loader.data_generator(
            sequence_length=sequence_length,
            batch_size=batch_size,
            target_size=target_size,
            one_hot=one_hot,
            infinite_loop=infinite_loop,
            max_sequences_per_video=max_sequences_per_video
        )
    
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(Config.SEQUENCE_LENGTH, *Config.INPUT_SIZE, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(Config.SEQUENCE_LENGTH, 2), dtype=tf.float32)
        )
    )
    
    # Оптимизация загрузки данных
    if Config.MEMORY_OPTIMIZATION['cache_dataset']:
        dataset = dataset.cache()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    
    return dataset

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
    model_type = model_type or Config.MODEL_TYPE
    results_path = os.path.join(Config.MODEL_SAVE_PATH, 'tuning', 'optuna_results.txt')
    
    if not os.path.exists(results_path):
        print(f"Файл с результатами подбора гиперпараметров не найден. Используем параметры по умолчанию для {model_type}.")
        default_params = {
            'learning_rate': 1e-4,
            'dropout_rate': Config.MODEL_PARAMS[model_type]['dropout_rate'],
            'batch_size': 16
        }
        if model_type == 'v3':
            default_params['lstm_units'] = Config.MODEL_PARAMS[model_type]['lstm_units']
        return default_params
    
    try:
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
                print(f"Не найдены результаты для модели {model_type}. Используем параметры по умолчанию.")
                best_params = {
                    'learning_rate': 1e-4,
                    'dropout_rate': Config.MODEL_PARAMS[model_type]['dropout_rate'],
                    'batch_size': 16
                }
                if model_type == 'v3':
                    best_params['lstm_units'] = Config.MODEL_PARAMS[model_type]['lstm_units']
            
            print(f"Загружены лучшие параметры для {model_type}: {best_params}")
            return best_params
            
    except Exception as e:
        print(f"Ошибка при загрузке параметров: {e}")
    
    print(f"Не удалось загрузить параметры для {model_type}. Используем параметры по умолчанию.")
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
    Обучение модели
    Args:
        model_type: тип модели ('v3' или 'v4'). Если None, используется Config.MODEL_TYPE
    """
    # Очищаем память перед началом обучения
    clear_memory()
    
    # Определяем тип модели
    model_type = model_type or Config.MODEL_TYPE
    
    # Загружаем лучшие параметры
    best_params = load_best_params(model_type)
    
    # Создание директорий
    model_save_path = os.path.join(Config.MODEL_SAVE_PATH, model_type)
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(os.path.join(model_save_path, 'plots'), exist_ok=True)
    
    # Создание модели с лучшими параметрами
    input_shape = (Config.SEQUENCE_LENGTH, *Config.INPUT_SIZE, 3)
    
    # Получаем параметры модели в зависимости от типа
    model_params = Config.MODEL_PARAMS[model_type]
    if model_type == 'v3':
        model = create_model(
            input_shape=input_shape,
            num_classes=Config.NUM_CLASSES,
            dropout_rate=best_params.get('dropout_rate', model_params['dropout_rate']),
            lstm_units=best_params.get('lstm_units', model_params['lstm_units']),
            model_type=model_type
        )
    else:  # v4
        model = create_model(
            input_shape=input_shape,
            num_classes=Config.NUM_CLASSES,
            dropout_rate=best_params.get('dropout_rate', model_params['dropout_rate']),
            model_type=model_type
        )
    
    # Компиляция модели с mixed precision и лучшим learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=best_params['learning_rate'])
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
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(model_save_path, 'best_model.h5'),
            monitor='val_loss',
            save_best_only=True
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=Config.OVERFITTING_PREVENTION['early_stopping_patience'],
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=Config.OVERFITTING_PREVENTION['reduce_lr_factor'],
            patience=Config.OVERFITTING_PREVENTION['reduce_lr_patience'],
            min_lr=Config.OVERFITTING_PREVENTION['min_lr']
        ),
        OverfittingMonitor(
            threshold=Config.OVERFITTING_PREVENTION['max_overfitting_threshold']
        ),
        TrainingPlotter(os.path.join(model_save_path, 'plots')),
        TqdmCallback(verbose=1)
    ]
    
    try:
        # Загрузка данных
        train_loader = VideoDataLoader(Config.TRAIN_DATA_PATH)
        val_loader = VideoDataLoader(Config.VALID_DATA_PATH)
        
        # Создание оптимизированных pipeline данных
        train_dataset = create_data_pipeline(
            loader=train_loader,
            sequence_length=Config.SEQUENCE_LENGTH,
            batch_size=best_params['batch_size'],
            target_size=Config.INPUT_SIZE,
            one_hot=True,
            infinite_loop=True,
            max_sequences_per_video=Config.MAX_SEQUENCES_PER_VIDEO
        )
        
        val_dataset = create_data_pipeline(
            loader=val_loader,
            sequence_length=Config.SEQUENCE_LENGTH,
            batch_size=best_params['batch_size'],
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
            verbose=0
        )
        
        # Визуализация результатов
        plot_path = os.path.join(model_save_path, 'plots')
        
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
        plt.savefig(os.path.join(plot_path, 'final_training_plot.png'))
        plt.close()
        
        # Сохраняем параметры модели
        model_params = {
            'model_type': model_type,
            'best_params': best_params,
            'input_shape': input_shape,
            'num_classes': Config.NUM_CLASSES,
            'batch_size': best_params['batch_size'],
            'sequence_length': Config.SEQUENCE_LENGTH,
            'max_sequences_per_video': Config.MAX_SEQUENCES_PER_VIDEO
        }
        
        with open(os.path.join(model_save_path, 'model_params.json'), 'w') as f:
            json.dump(model_params, f, indent=4)
        
        return model, history
        
    except Exception as e:
        print(f"Ошибка при обучении модели: {e}")
        raise
    finally:
        clear_memory()

if __name__ == "__main__":
    # Обучаем обе модели
    print("Обучение MobileNetV3...")
    model_v3, history_v3 = train('v3')
    
    print("\nОбучение MobileNetV4...")
    model_v4, history_v4 = train('v4') 