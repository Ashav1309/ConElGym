import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import Callback
from typing import Tuple
from src.config import Config
import pickle
import os
import time
from tensorflow.keras.metrics import F1Score

class ScalarF1Score(tf.keras.metrics.Metric):
    """
    Метрика F1-score, которая возвращает скалярное значение
    """
    def __init__(self, name='scalar_f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.f1 = tf.keras.metrics.F1Score(threshold=0.5)
        
    def update_state(self, y_true, y_pred, sample_weight=None):
         # Преобразуем входные данные в 2D
        y_true = tf.reshape(y_true, [-1, y_true.shape[-1]])
        y_pred = tf.reshape(y_pred, [-1, y_pred.shape[-1]])
        # БИНАРИЗАЦИЯ!
        y_pred = tf.cast(y_pred > 0.5, tf.float32)
        if sample_weight is not None:
           sample_weight = tf.reshape(sample_weight, [-1])
        self.f1.update_state(y_true, y_pred, sample_weight)
        
    def result(self):
        return tf.reduce_mean(self.f1.result())
        
    def reset_states(self):
        self.f1.reset_states()

class PickleModelCheckpoint(Callback):
    """
    Callback для сохранения модели в формате pickle
    """
    def __init__(self, filepath, monitor='val_f1_action', save_best_only=True, mode='max'):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.best = -np.inf if mode == 'max' else np.inf
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return
            
        if self.mode == 'min':
            improved = current < self.best
        else:
            improved = current > self.best
            
        if improved or not self.save_best_only:
            self.best = current
            # Сохраняем модель и метаданные
            model_data = {
                'model': self.model,
                'epoch': epoch,
                'best_metric': self.best,
                'monitor': self.monitor,
                'mode': self.mode,
                'logs': logs
            }
            with open(self.filepath, 'wb') as f:
                pickle.dump(model_data, f)
            print(f'\n[INFO] Сохранена модель в {self.filepath} (метрика: {self.best:.4f})')

def get_training_callbacks(val_data, config=None):
    """
    Получение callbacks для обучения модели
    Args:
        val_data: валидационные данные
        config: конфигурация с параметрами callbacks
    """
    if config is None:
        config = Config.OVERFITTING_PREVENTION
    
    # Создаем директорию для сохранения модели с временной меткой
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(Config.MODEL_SAVE_PATH, f'model_{timestamp}')
    os.makedirs(model_dir, exist_ok=True)
    
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_f1_action',
            patience=config['early_stopping_patience'],
            restore_best_weights=True,
            mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_f1_action',
            factor=config['reduce_lr_factor'],
            patience=config['reduce_lr_patience'],
            min_lr=config['min_lr'],
            mode='max'
        ),
        PickleModelCheckpoint(
            os.path.join(model_dir, 'model.pkl'),
            monitor='val_f1_action',
            save_best_only=True,
            mode='max'
        )
    ]

def get_tuning_callbacks(trial_number):
    """
    Получение колбэков для подбора гиперпараметров
    """
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_f1_action',
            patience=5,  # Фиксированное значение для early stopping
            restore_best_weights=True,
            mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_f1_action',
            factor=0.5,
            patience=3,  # Фиксированное значение для reduce lr
            min_lr=1e-6,
            mode='max'
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(Config.MODEL_SAVE_PATH, f'trial_{trial_number}_best_model.h5'),
            monitor='val_f1_action',
            save_best_only=True,
            mode='max'
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(Config.MODEL_SAVE_PATH, f'trial_{trial_number}_history.csv')
        )
    ]
    
    return callbacks

def get_training_metrics():
    """
    Получение метрик для обучения модели
    """
    return [
        'accuracy',
        tf.keras.metrics.Precision(name='precision_action', class_id=1, thresholds=0.5),
        tf.keras.metrics.Recall(name='recall_action', class_id=1, thresholds=0.5),
        F1Score(name='f1_action', class_id=1, threshold=0.5)
    ] 