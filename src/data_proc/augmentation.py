import tensorflow as tf
import numpy as np
import cv2
from typing import Tuple, List
from src.config import Config

def apply_augmentations(image):
    """
    Применяет аугментации к изображению используя настройки из конфигурации
    """
    if not Config.AUGMENTATION['enabled']:
        return image
        
    # Яркость
    if np.random.random() < Config.AUGMENTATION['probability']:
        delta = np.random.uniform(*Config.AUGMENTATION['brightness_range'])
        image = tf.image.adjust_brightness(image, delta)
    
    # Контраст
    if np.random.random() < Config.AUGMENTATION['probability']:
        delta = np.random.uniform(*Config.AUGMENTATION['contrast_range'])
        image = tf.image.adjust_contrast(image, delta)
    
    # Поворот
    if np.random.random() < Config.AUGMENTATION['probability']:
        angle = np.random.uniform(*Config.AUGMENTATION['rotation_range'])
        image = tf.image.rot90(image, k=int(angle/90))
    
    # Отражение
    if np.random.random() < Config.AUGMENTATION['flip_probability']:
        image = tf.image.flip_left_right(image)
        
    # Масштабирование
    if np.random.random() < Config.AUGMENTATION['probability']:
        zoom = np.random.uniform(*Config.AUGMENTATION['zoom_range'])
        image = tf.image.resize_with_crop_or_pad(image, 
            int(image.shape[0] * zoom), 
            int(image.shape[1] * zoom))
        image = tf.image.resize(image, [image.shape[0], image.shape[1]])
    
    # Сдвиг
    if np.random.random() < Config.AUGMENTATION['probability']:
        shear = np.random.uniform(*Config.AUGMENTATION['shear_range'])
        image = tf.image.translate(image, [shear * image.shape[1], 0])
    
    # Шум
    if np.random.random() < Config.AUGMENTATION['probability']:
        noise = tf.random.normal(shape=tf.shape(image), 
                               mean=0.0, 
                               stddev=Config.AUGMENTATION['noise_factor'])
        image = tf.clip_by_value(image + noise, 0.0, 1.0)
    
    # Размытие
    if np.random.random() < Config.AUGMENTATION['blur_probability']:
        kernel_size = Config.AUGMENTATION['blur_kernel_size']
        image = tf.image.gaussian_blur(image, kernel_size)
    
    return image

def augment_sequence(frames: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Аугментация последовательности кадров
    """
    if not Config.AUGMENTATION['enabled']:
        return frames, labels
        
    augmented_frames = []
    for frame in frames:
        augmented_frames.append(apply_augmentations(frame))
    
    return np.array(augmented_frames), labels

def augment_rare_classes(images, labels):
    """
    Аугментирует только редкие классы (действия)
    """
    if not Config.AUGMENTATION['enabled']:
        return images, labels
        
    # Находим редкие классы (действия)
    rare_class_indices = np.where(np.any(labels[:, 1:], axis=1))[0]
    
    # Аугментируем только редкие классы
    augmented_images = []
    augmented_labels = []
    
    for idx in rare_class_indices:
        # Применяем аугментации
        aug_img = apply_augmentations(images[idx])
        augmented_images.append(aug_img)
        augmented_labels.append(labels[idx])
        
        # Если включено увеличение положительных примеров
        if Config.DATA_BALANCING['oversample_positive']:
            for _ in range(int(Config.DATA_BALANCING['oversample_factor'] - 1)):
                aug_img = apply_augmentations(images[idx])
                augmented_images.append(aug_img)
                augmented_labels.append(labels[idx])
    
    # Объединяем с оригинальными данными
    if len(augmented_images) > 0:
        images = np.concatenate([images, np.array(augmented_images)])
        labels = np.concatenate([labels, np.array(augmented_labels)])
    
    return images, labels

def create_balanced_batches(dataset, batch_size):
    """
    Создает сбалансированные батчи с равным количеством примеров каждого класса
    """
    if not Config.DATA_BALANCING['enabled']:
        return list(range(len(dataset)))
        
    # Группируем данные по классам
    class_indices = {
        0: [],  # фон
        1: []   # действие
    }
    
    for i, label in enumerate(dataset.labels):
        class_idx = np.argmax(label)
        class_indices[class_idx].append(i)
    
    # Создаем сбалансированные батчи
    batches = []
    while True:
        batch_indices = []
        for class_idx in class_indices:
            # Берем равное количество примеров каждого класса
            samples = np.random.choice(class_indices[class_idx], 
                                     size=int(batch_size * Config.DATA_BALANCING['class_ratio']), 
                                     replace=True)
            batch_indices.extend(samples)
        
        np.random.shuffle(batch_indices)
        batches.append(batch_indices)
        
        if len(batches) * batch_size >= len(dataset):
            break
    
    return batches

class VideoAugmenter:
    """
    Класс для аугментации видео
    """
    def __init__(self):
        self.config = Config.AUGMENTATION
    
    def augment(self, video, labels):
        """
        Применяет аугментацию к видео и меткам
        """
        if not self.config['enabled']:
            return video, labels
            
        # Проверяем, нужно ли аугментировать только положительные примеры
        if Config.AUGMENT_POSITIVE_ONLY:
            # Проверяем, есть ли хотя бы один кадр с действием
            has_action = np.any(np.argmax(labels, axis=1) == 1)
            if not has_action:
                return video, labels
        
        # Применяем аугментации к каждому кадру
        augmented_video = []
        for frame in video:
            augmented_video.append(apply_augmentations(frame))
        
        return np.array(augmented_video), labels

class BalancedDataGenerator(tf.keras.utils.Sequence):
    """
    Генератор данных с балансировкой классов
    """
    def __init__(self, images, labels, batch_size, augment=True):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.augment = augment and Config.AUGMENTATION['enabled']
        
        # Создаем сбалансированные батчи
        self.batches = create_balanced_batches(self, batch_size)
        
    def __len__(self):
        return len(self.batches)
    
    def __getitem__(self, idx):
        batch_indices = self.batches[idx]
        
        # Получаем данные для батча
        batch_images = self.images[batch_indices]
        batch_labels = self.labels[batch_indices]
        
        # Применяем аугментацию если нужно
        if self.augment:
            batch_images, batch_labels = augment_rare_classes(batch_images, batch_labels)
        
        return batch_images, batch_labels
    
    def on_epoch_end(self):
        # Перемешиваем батчи в конце каждой эпохи
        np.random.shuffle(self.batches) 