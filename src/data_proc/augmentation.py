import tensorflow as tf
import numpy as np
import cv2
from typing import Tuple, List
from src.config import Config
from sklearn.neighbors import NearestNeighbors
import logging

def apply_augmentations(image):
    """
    Применяет аугментации к изображению на основе настроек из конфига
    """
    if image is None:
        raise ValueError("Входное изображение не может быть None")
    if not isinstance(image, (tf.Tensor, np.ndarray)):
        raise TypeError("Входное изображение должно быть tf.Tensor или np.ndarray")
        
    # Проверяем размеры изображения
    if isinstance(image, tf.Tensor):
        shape = image.shape
        if len(shape) < 3 or shape[0] == 0 or shape[1] == 0:
            logging.error(f"Некорректные размеры изображения: {shape}")
            return image
    else:
        if len(image.shape) < 3 or image.shape[0] == 0 or image.shape[1] == 0:
            logging.error(f"Некорректные размеры изображения: {image.shape}")
            return image
        
    if not Config.AUGMENTATION['enabled']:
        return image
        
    # ... (остальной код без изменений до блока сдвига)
    
    # Сдвиг
    if np.random.random() < Config.AUGMENTATION['shift_prob']:
        shift = np.random.uniform(
            -Config.AUGMENTATION['shift_range'],
            Config.AUGMENTATION['shift_range']
        )
        # Преобразуем тензор в numpy массив для cv2
        image_np = image.numpy() if isinstance(image, tf.Tensor) else image
        # Проверяем размеры перед сдвигом
        if image_np.shape[0] == 0 or image_np.shape[1] == 0:
            logging.error(f"Некорректные размеры изображения перед сдвигом: {image_np.shape}")
            return image
        # Создаем матрицу преобразования
        rows, cols = image_np.shape[:2]
        M = np.float32([[1, 0, shift], [0, 1, shift]])
        # Применяем сдвиг
        try:
            shifted = cv2.warpAffine(image_np, M, (cols, rows))
            # Преобразуем обратно в тензор
            image = tf.convert_to_tensor(shifted, dtype=tf.float32)
        except cv2.error as e:
            logging.error(f"Ошибка при применении сдвига: {e}")
            return image
    
    # ... (остальной код без изменений до блока размытия)
    
    # Размытие
    if np.random.random() < Config.AUGMENTATION['blur_prob']:
        kernel_size = np.random.choice([3, 5])
        # Преобразуем тензор в numpy массив для cv2
        image_np = image.numpy() if isinstance(image, tf.Tensor) else image
        # Проверяем размеры перед размытием
        if image_np.shape[0] == 0 or image_np.shape[1] == 0:
            logging.error(f"Некорректные размеры изображения перед размытием: {image_np.shape}")
            return image
        # Применяем размытие
        try:
            blurred = cv2.GaussianBlur(
                image_np,
                (kernel_size, kernel_size),
                Config.AUGMENTATION['blur_sigma']
            )
            # Преобразуем обратно в тензор
            image = tf.convert_to_tensor(blurred, dtype=tf.float32)
        except cv2.error as e:
            logging.error(f"Ошибка при применении размытия: {e}")
            return image
    
    return image

def augment_sequence(frames: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Аугментация последовательности кадров
    """
    if not Config.AUGMENTATION['enabled']:
        return frames, labels
        
    if len(frames) != len(labels):
        raise ValueError("Количество кадров не совпадает с количеством меток")
        
    augmented_frames = []
    for frame in frames:
        augmented_frames.append(apply_augmentations(frame))
    
    return np.array(augmented_frames), labels

def augment_rare_classes(images, labels, is_training=True):
    """
    Аугментирует только редкие классы (действия)
    is_training: флаг, указывающий, что это режим обучения (аугментация применяется только в этом режиме)
    """
    if not Config.AUGMENTATION['enabled'] or not is_training:
        return images, labels
        
    # Находим редкие классы (действия)
    rare_class_indices = np.where(np.any(labels[:, 1:], axis=1))[0]
    
    # Аугментируем только редкие классы
    augmented_images = []
    augmented_labels = []
    
    for idx in rare_class_indices:
        aug_img = apply_augmentations(images[idx])
        augmented_images.append(aug_img)
        augmented_labels.append(labels[idx])
        
        if Config.DATA_BALANCING['oversample_positive']:
            for _ in range(int(Config.DATA_BALANCING['oversample_factor'] - 1)):
                aug_img = apply_augmentations(images[idx])
                augmented_images.append(aug_img)
                augmented_labels.append(labels[idx])
    
    # Объединяем с оригинальными данными
    if len(augmented_images) > 0:
        images = np.concatenate([images, np.array(augmented_images)])
        labels = np.concatenate([labels, np.array(augmented_labels)])
    
    # Применяем SMOTE если включено
    if Config.DATA_BALANCING['use_smote']:
        images, labels = apply_smote(images, labels, k_neighbors=Config.DATA_BALANCING['smote_k_neighbors'])
    
    return images, labels

def apply_smote(images, labels, k_neighbors=5):
    """
    Применяет SMOTE для генерации синтетических примеров положительного класса
    """
    if not Config.DATA_BALANCING['use_smote']:
        return images, labels
        
    # Находим положительные примеры (действия)
    positive_indices = np.where(np.argmax(labels, axis=1) == 1)[0]
    if len(positive_indices) < 2:
        return images, labels
        
    logging.debug(f"SMOTE: найдено {len(positive_indices)} положительных примеров")
    
    # Подготавливаем данные для SMOTE
    positive_images = images[positive_indices]
    positive_labels = labels[positive_indices]
    
    # Вычисляем количество синтетических примеров
    n_samples = int(len(positive_images) * (Config.DATA_BALANCING['oversample_factor'] - 1))
    
    # Ограничиваем количество синтетических примеров для экономии памяти
    if len(positive_images) * n_samples > 10000:
        n_samples = 10000 // len(positive_images)
        logging.warning(f"SMOTE: ограничено количество синтетических примеров до {n_samples}")
    
    # Находим k ближайших соседей
    nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(positive_images.reshape(len(positive_images), -1))
    distances, indices = nbrs.kneighbors(positive_images.reshape(len(positive_images), -1))
    
    # Генерируем синтетические примеры
    synthetic_images = []
    synthetic_labels = []
    
    for i in range(n_samples):
        idx = np.random.randint(0, len(positive_images))
        neighbor_idx = np.random.choice(indices[idx][1:])
        alpha = np.random.random()
        synthetic_image = tf.cast(alpha * positive_images[idx] + (1 - alpha) * positive_images[neighbor_idx], tf.float32)
        synthetic_images.append(synthetic_image)
        synthetic_labels.append(positive_labels[idx])
    
    # Объединяем с оригинальными данными
    if len(synthetic_images) > 0:
        images = np.concatenate([images, np.array(synthetic_images)])
        labels = np.concatenate([labels, np.array(synthetic_labels)])
    
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
            if not class_indices[class_idx]:  # Пропускаем пустые классы
                continue
            # Берем равное количество примеров каждого класса
            samples = np.random.choice(class_indices[class_idx], 
                                     size=int(batch_size * Config.DATA_BALANCING['class_ratio']), 
                                     replace=True)
            batch_indices.extend(samples)
        
        if not batch_indices:  # Если все классы пустые
            break
            
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