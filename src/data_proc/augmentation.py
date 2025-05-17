import tensorflow as tf
import numpy as np
from src.config import Config

def apply_augmentations(image):
    """
    Применяет аугментации к изображению
    """
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
    
    return image

def augment_rare_classes(images, labels):
    """
    Аугментирует только редкие классы
    """
    # Находим редкие классы (действия и переходы)
    rare_class_indices = np.where(np.any(labels[:, 1:], axis=1))[0]
    
    # Аугментируем только редкие классы
    augmented_images = []
    augmented_labels = []
    
    for idx in rare_class_indices:
        # Применяем аугментации
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
    # Группируем данные по классам
    class_indices = {
        0: [],  # фон
        1: [],  # действие
        2: []   # переход
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
                                     size=batch_size//3, 
                                     replace=True)
            batch_indices.extend(samples)
        
        np.random.shuffle(batch_indices)
        batches.append(batch_indices)
        
        if len(batches) * batch_size >= len(dataset):
            break
    
    return batches

class BalancedDataGenerator(tf.keras.utils.Sequence):
    """
    Генератор данных с балансировкой классов
    """
    def __init__(self, images, labels, batch_size, augment=True):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.augment = augment
        
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