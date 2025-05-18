import numpy as np
import cv2
import tensorflow as tf
from typing import Tuple, List
from src.models.losses import focal_loss

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
    def __init__(self,
                 augment_probability: float = 0.5,
                 rotation_range: int = 10,
                 width_shift_range: float = 0.1,
                 height_shift_range: float = 0.1,
                 brightness_range: float = 0.2,
                 contrast_range: float = 0.2,
                 saturation_range: float = 0.2,
                 hue_range: float = 0.1,
                 zoom_range: float = 0.2,
                 horizontal_flip: float = 0.5,
                 vertical_flip: float = 0.0):
        self.augment_probability = augment_probability
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_range = hue_range
        self.zoom_range = zoom_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
    
    def augment_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Аугментация одного кадра
        """
        # Горизонтальное отражение
        if np.random.random() < self.horizontal_flip:
            frame = cv2.flip(frame, 1)
        
        # Вертикальное отражение
        if np.random.random() < self.vertical_flip:
            frame = cv2.flip(frame, 0)
        
        # Поворот
        if self.rotation_range > 0:
            angle = np.random.uniform(-self.rotation_range, self.rotation_range)
            h, w = frame.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            frame = cv2.warpAffine(frame, M, (w, h))
        
        # Сдвиг
        if self.width_shift_range > 0 or self.height_shift_range > 0:
            tx = np.random.uniform(-self.width_shift_range, self.width_shift_range) * frame.shape[1]
            ty = np.random.uniform(-self.height_shift_range, self.height_shift_range) * frame.shape[0]
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
        
        # Масштабирование
        if self.zoom_range > 0:
            scale = np.random.uniform(1 - self.zoom_range, 1 + self.zoom_range)
            h, w = frame.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), 0, scale)
            frame = cv2.warpAffine(frame, M, (w, h))
        
        # Цветовые преобразования
        if self.brightness_range > 0:
            brightness = np.random.uniform(1 - self.brightness_range, 1 + self.brightness_range)
            frame = cv2.convertScaleAbs(frame, alpha=brightness, beta=0)
        
        if self.contrast_range > 0:
            contrast = np.random.uniform(1 - self.contrast_range, 1 + self.contrast_range)
            frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=0)
        
        if self.saturation_range > 0 or self.hue_range > 0:
            frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            if self.saturation_range > 0:
                saturation = np.random.uniform(1 - self.saturation_range, 1 + self.saturation_range)
                frame_hsv[:,:,1] = cv2.multiply(frame_hsv[:,:,1], saturation)
            
            if self.hue_range > 0:
                hue = np.random.uniform(-self.hue_range, self.hue_range)
                frame_hsv[:,:,0] = cv2.add(frame_hsv[:,:,0], hue)
            
            frame = cv2.cvtColor(frame_hsv, cv2.COLOR_HSV2BGR)
        
        return frame
    
    def augment_batch(self, frames_batch: np.ndarray, labels_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Аугментация батча последовательностей
        """
        augmented_frames = []
        augmented_labels = []
        
        for frames, labels in zip(frames_batch, labels_batch):
            if np.random.random() < self.augment_probability:
                aug_frames = []
                for frame in frames:
                    aug_frames.append(self.augment_frame(frame))
                augmented_frames.append(np.array(aug_frames))
                augmented_labels.append(labels)
            else:
                augmented_frames.append(frames)
                augmented_labels.append(labels)
        
        return np.array(augmented_frames), np.array(augmented_labels)

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
