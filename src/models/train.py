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

def create_data_pipeline(generator, batch_size):
    """
    Создает оптимизированный pipeline данных с использованием tf.data.Dataset
    """
    dataset = tf.data.Dataset.from_generator(
        lambda: generator,
        output_signature=(
            tf.TensorSpec(shape=(None, Config.SEQUENCE_LENGTH, *Config.INPUT_SIZE, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, Config.NUM_CLASSES), dtype=tf.float32)
        )
    )
    
    # Оптимизация загрузки данных
    if Config.MEMORY_OPTIMIZATION['cache_dataset']:
        dataset = dataset.cache()
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    
    # Исправление размерности данных
    def reshape_data(x, y):
        x = tf.squeeze(x, axis=1)
        y = tf.squeeze(y, axis=1)
        return x, y
    
    dataset = dataset.map(reshape_data, num_parallel_calls=tf.data.AUTOTUNE)
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

def train():
    # Очищаем память перед началом обучения
    clear_memory()
    
    # Создание директорий
    os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(os.path.join(Config.MODEL_SAVE_PATH, 'plots'), exist_ok=True)
    
    # Создание модели
    input_shape = (Config.SEQUENCE_LENGTH, *Config.INPUT_SIZE, 3)
    model = create_model(
        input_shape=input_shape,
        num_classes=Config.NUM_CLASSES,
        dropout_rate=0.5,
        lstm_units=32  # Увеличиваем количество LSTM units
    )
    
    # Компиляция модели с mixed precision
    optimizer = tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE)
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(Config.MODEL_SAVE_PATH, 'best_model.h5'),
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
        TrainingPlotter(os.path.join(Config.MODEL_SAVE_PATH, 'plots')),
        TqdmCallback(verbose=1)  # Добавляем прогресс-бар
    ]
    
    try:
        # Загрузка данных
        train_loader = VideoDataLoader(Config.TRAIN_DATA_PATH)
        val_loader = VideoDataLoader(Config.VALID_DATA_PATH)
        
        # Создание генераторов данных
        train_generator = train_loader.load_data(
            Config.SEQUENCE_LENGTH, 
            Config.BATCH_SIZE, 
            target_size=Config.INPUT_SIZE,
            one_hot=True,
            infinite_loop=True
        )
        val_generator = val_loader.load_data(
            Config.SEQUENCE_LENGTH, 
            Config.BATCH_SIZE, 
            target_size=Config.INPUT_SIZE,
            one_hot=True,
            infinite_loop=True
        )
        
        # Создание оптимизированных pipeline данных
        train_dataset = create_data_pipeline(train_generator, Config.BATCH_SIZE)
        val_dataset = create_data_pipeline(val_generator, Config.BATCH_SIZE)
        
        # Обучение
        history = model.fit(
            train_dataset,
            epochs=Config.EPOCHS,
            validation_data=val_dataset,
            callbacks=callbacks,
            steps_per_epoch=Config.STEPS_PER_EPOCH,
            validation_steps=Config.VALIDATION_STEPS,
            verbose=0  # Отключаем стандартный вывод, так как используем tqdm
        )
        
        # Визуализация результатов
        plot_path = os.path.join(Config.MODEL_SAVE_PATH, 'plots')
        
        # Графики потерь и точности
        plt.figure(figsize=(12, 4))
        
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
        
        # Матрица ошибок
        print("\nВычисление матрицы ошибок...")
        val_generator = val_loader.load_data(
            Config.SEQUENCE_LENGTH, 
            Config.BATCH_SIZE, 
            target_size=Config.INPUT_SIZE, 
            one_hot=True
        )
        val_dataset = create_data_pipeline(val_generator, Config.BATCH_SIZE)
        y_true = []
        y_pred = []
        
        # Добавляем прогресс-бар для вычисления матрицы ошибок
        with tqdm(total=Config.VALIDATION_STEPS, desc="Computing Confusion Matrix") as pbar:
            for _ in range(Config.VALIDATION_STEPS):
                X_val, y_val = next(val_generator)
                pred = model.predict(X_val, verbose=0)
                y_true.extend(y_val)
                y_pred.extend(np.round(pred))
                pbar.update(1)
        
        plot_confusion_matrix(np.array(y_true), np.array(y_pred), plot_path)
        
        return history
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        clear_memory()
        raise e
    finally:
        # Очищаем память после обучения
        clear_memory()

if __name__ == "__main__":
    train() 