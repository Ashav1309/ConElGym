import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import Callback
from typing import Tuple
from src.config import Config

class ScalarF1Score(tf.keras.metrics.Metric):
    """
    Метрика F1-score, которая возвращает скалярное значение
    """
    def __init__(self, name='scalar_f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.f1 = tf.keras.metrics.F1Score(threshold=0.5)
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.f1.update_state(y_true, y_pred, sample_weight)
        
    def result(self):
        return tf.reduce_mean(self.f1.result())
        
    def reset_states(self):
        self.f1.reset_states()

class AdaptiveThresholdCallback(Callback):
    """
    Callback для поиска оптимального порога классификации
    """
    def __init__(self, validation_data: Tuple[np.ndarray, np.ndarray]):
        super().__init__()
        self.validation_data = validation_data
        self.best_threshold = 0.5
        self.best_f1 = 0.0
    
    def on_epoch_end(self, epoch: int, logs: dict = None):
        # Получаем предсказания на валидационном наборе
        val_pred = self.model.predict(self.validation_data[0])
        val_true = self.validation_data[1]
        
        # Ищем оптимальный порог
        thresholds = np.arange(0.1, 1.0, 0.05)
        f1_scores = []
        
        for threshold in thresholds:
            val_pred_binary = (val_pred >= threshold).astype(int)
            f1 = tf.keras.metrics.F1Score()(val_true, val_pred_binary)
            f1_scores.append(f1)
        
        best_idx = np.argmax(f1_scores)
        current_f1 = f1_scores[best_idx]
        current_threshold = thresholds[best_idx]
        
        # Обновляем лучший порог, если нашли лучше
        if current_f1 > self.best_f1:
            self.best_f1 = current_f1
            self.best_threshold = current_threshold
            print(f"\n[INFO] Новый лучший порог: {self.best_threshold:.3f} (F1: {self.best_f1:.3f})")
        
        # Добавляем метрики в логи
        logs['val_threshold'] = self.best_threshold
        logs['val_f1'] = self.best_f1

def get_training_callbacks(val_data, config=None):
    """
    Получение callbacks для обучения модели
    
    Args:
        val_data: валидационные данные
        config: конфигурация с параметрами callbacks
    """
    if config is None:
        config = Config.OVERFITTING_PREVENTION
    
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_f1_score',
            patience=config['early_stopping_patience'],
            restore_best_weights=True,
            mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_f1_score',
            factor=config['reduce_lr_factor'],
            patience=config['reduce_lr_patience'],
            min_lr=config['min_lr'],
            mode='max'
        ),
        AdaptiveThresholdCallback(validation_data=val_data)
    ]

def get_tuning_callbacks(trial_number):
    """
    Получение callbacks для подбора гиперпараметров
    
    Args:
        trial_number: номер текущего trial
    """
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_scalar_f1_score',
            patience=5,
            restore_best_weights=True,
            mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_scalar_f1_score',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            mode='max'
        ),
        tf.keras.callbacks.ModelCheckpoint(
            f'best_model_trial_{trial_number}.h5',
            monitor='val_scalar_f1_score',
            save_best_only=True,
            mode='max'
        ),
        tf.keras.callbacks.CSVLogger(f'trial_{trial_number}_history.csv')
    ] 