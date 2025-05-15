import numpy as np
import cv2
import tensorflow as tf
from typing import Tuple, List

def augment_frame(frame: np.ndarray) -> np.ndarray:
    """
    Аугментация одного кадра
    """
    # Случайное изменение яркости
    if np.random.random() < 0.5:
        brightness = np.random.uniform(0.8, 1.2)
        frame = cv2.convertScaleAbs(frame, alpha=brightness, beta=0)
    
    # Случайное изменение контраста
    if np.random.random() < 0.5:
        contrast = np.random.uniform(0.8, 1.2)
        frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=0)
    
    # Случайное горизонтальное отражение
    if np.random.random() < 0.5:
        frame = cv2.flip(frame, 1)
    
    # Случайный поворот
    if np.random.random() < 0.5:
        angle = np.random.uniform(-10, 10)
        h, w = frame.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        frame = cv2.warpAffine(frame, M, (w, h))
    
    return frame

def augment_sequence(frames: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Аугментация последовательности кадров
    """
    augmented_frames = []
    for frame in frames:
        augmented_frames.append(augment_frame(frame))
    
    return np.array(augmented_frames), labels

class VideoAugmenter:
    """
    Класс для аугментации видео последовательностей
    """
    def __init__(self, augment_probability: float = 0.5):
        self.augment_probability = augment_probability
    
    def augment_batch(self, frames_batch: np.ndarray, labels_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Аугментация батча последовательностей
        """
        augmented_frames = []
        augmented_labels = []
        
        for frames, labels in zip(frames_batch, labels_batch):
            # Аугментируем только положительные последовательности
            if np.any(labels[:, 1] == 1) and np.random.random() < self.augment_probability:
                aug_frames, aug_labels = augment_sequence(frames, labels)
                augmented_frames.append(aug_frames)
                augmented_labels.append(aug_labels)
            else:
                augmented_frames.append(frames)
                augmented_labels.append(labels)
        
        return np.array(augmented_frames), np.array(augmented_labels)

def focal_loss(gamma: float = 2.0, alpha: float = 0.25) -> tf.keras.losses.Loss:
    """
    Focal Loss для бинарной классификации
    """
    def focal_loss_fixed(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        # Преобразуем предсказания в вероятности
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        # Вычисляем focal loss
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        focal_loss = -alpha * tf.pow(1 - pt, gamma) * tf.math.log(pt)
        
        return tf.reduce_mean(focal_loss)
    
    return focal_loss_fixed

class AdaptiveThresholdCallback(tf.keras.callbacks.Callback):
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