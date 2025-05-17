import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import Callback
from src.config import Config
import json
import os

def focal_loss(gamma=2., alpha=0.25, beta=0.999):
    """
    Улучшенный focal loss с дополнительной балансировкой через beta
    """
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)
        
        # Загружаем веса классов
        with open(Config.CONFIG_PATH, 'r') as f:
            config = json.load(f)
            class_weights = config['MODEL_PARAMS'][Config.MODEL_TYPE]['class_weights']
        
        # Применяем веса к каждому классу
        weights = tf.constant([
            class_weights['background'],
            class_weights['action'],
            class_weights['transition']
        ])
        
        # Добавляем beta для дополнительной балансировки
        beta_weight = tf.pow(beta, tf.reduce_sum(y_true, axis=-1))
        
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        alpha_weight = alpha * y_true + (1 - alpha) * (1 - y_true)
        pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = tf.pow(1 - pt, gamma)
        
        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        
        # Применяем все веса
        loss = alpha_weight * focal_weight * cross_entropy * weights * beta_weight
        
        return tf.reduce_mean(loss)
    return focal_loss_fixed

class DynamicClassWeights(Callback):
    """
    Callback для динамического обновления весов классов
    """
    def __init__(self, validation_data, update_frequency=5):
        super().__init__()
        self.validation_data = validation_data
        self.update_frequency = update_frequency
        self.best_weights = None
        self.best_f1 = 0.0
        
    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.update_frequency != 0:
            return
            
        # Получаем предсказания на валидации
        y_pred = self.model.predict(self.validation_data[0])
        y_true = self.validation_data[1]
        
        # Рассчитываем F1-score для каждого класса
        f1_scores = []
        for i in range(3):  # 3 класса
            f1 = tf.keras.metrics.F1Score()(y_true[:, i], y_pred[:, i])
            f1_scores.append(f1)
        
        # Обновляем веса на основе F1-score
        current_weights = self.model.class_weights
        new_weights = {}
        
        for i, (class_name, f1) in enumerate(zip(['background', 'action', 'transition'], f1_scores)):
            # Увеличиваем вес для классов с низким F1-score
            if f1 < 0.5:  # Порог для "плохих" классов
                new_weights[class_name] = current_weights[class_name] * 1.1
            else:
                new_weights[class_name] = current_weights[class_name]
        
        # Обновляем веса в модели
        self.model.class_weights = new_weights
        
        # Сохраняем лучшие веса
        avg_f1 = np.mean(f1_scores)
        if avg_f1 > self.best_f1:
            self.best_f1 = avg_f1
            self.best_weights = new_weights.copy()
            
            # Сохраняем лучшие веса в конфиг
            with open(Config.CONFIG_PATH, 'r') as f:
                config = json.load(f)
            
            config['MODEL_PARAMS'][Config.MODEL_TYPE]['class_weights'] = self.best_weights
            
            with open(Config.CONFIG_PATH, 'w') as f:
                json.dump(config, f, indent=4)

class AdaptiveLearningRate(Callback):
    """
    Callback для адаптивного изменения learning rate и весов классов
    """
    def __init__(self, class_metrics, patience=3):
        super().__init__()
        self.class_metrics = class_metrics
        self.patience = patience
        self.bad_epochs = 0
        self.best_metrics = {class_name: 0.0 for class_name in class_metrics}
        
    def on_epoch_end(self, epoch, logs=None):
        # Анализируем метрики по классам
        worst_class = min(self.class_metrics.items(), 
                         key=lambda x: x[1]['f1_score'])[0]
        
        # Проверяем, улучшились ли метрики
        improved = False
        for class_name, metrics in self.class_metrics.items():
            if metrics['f1_score'] > self.best_metrics[class_name]:
                self.best_metrics[class_name] = metrics['f1_score']
                improved = True
        
        if improved:
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            
            if self.bad_epochs >= self.patience:
                # Увеличиваем вес худшего класса
                current_weights = self.model.class_weights
                current_weights[worst_class] *= 1.1
                
                # Уменьшаем learning rate
                current_lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
                new_lr = current_lr * 0.5
                tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                
                print(f"\n[INFO] Адаптация параметров:")
                print(f"  - Увеличен вес класса {worst_class}")
                print(f"  - Уменьшен learning rate до {new_lr}")
                
                self.bad_epochs = 0 