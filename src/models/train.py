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
    # Создание директорий
    os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(os.path.join(Config.MODEL_SAVE_PATH, 'plots'), exist_ok=True)
    
    # Создание модели
    model = create_model(
        input_shape=(Config.SEQUENCE_LENGTH, *Config.INPUT_SHAPE),
        num_classes=Config.NUM_CLASSES
    )
    
    # Компиляция модели
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE),
        loss='binary_crossentropy',
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
            patience=10,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6
        ),
        TrainingPlotter(os.path.join(Config.MODEL_SAVE_PATH, 'plots'))
    ]
    
    # Загрузка данных
    train_loader = VideoDataLoader(Config.TRAIN_DATA_PATH)
    val_loader = VideoDataLoader(Config.VALID_DATA_PATH)
    
    # Создание генераторов данных
    train_generator = train_loader.load_data(Config.SEQUENCE_LENGTH, Config.BATCH_SIZE)
    val_generator = val_loader.load_data(Config.SEQUENCE_LENGTH, Config.BATCH_SIZE)
    
    # Обучение
    history = model.fit(
        train_generator,
        epochs=Config.EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks,
        steps_per_epoch=100,  # Количество батчей в эпохе
        validation_steps=20   # Количество батчей для валидации
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
    val_generator = val_loader.load_data(Config.SEQUENCE_LENGTH, Config.BATCH_SIZE)
    y_true = []
    y_pred = []
    
    for _ in range(20):  # Используем 20 батчей для оценки
        X_val, y_val = next(val_generator)
        pred = model.predict(X_val)
        y_true.extend(y_val)
        y_pred.extend(np.round(pred))
    
    plot_confusion_matrix(np.array(y_true), np.array(y_pred), plot_path)
    
    return history

if __name__ == "__main__":
    train() 