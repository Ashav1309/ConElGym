import tensorflow as tf
from src.models.losses import F1ScoreAdapter
from src.models.callbacks import ScalarF1Score

def get_training_metrics():
    """
    Получение метрик для обучения модели.
    Использует те же метрики, что и для подбора гиперпараметров.
    """
    class SequenceMetrics(tf.keras.metrics.Metric):
        def __init__(self, name='sequence_metrics', **kwargs):
            super().__init__(name=name, **kwargs)
            self.precision = tf.keras.metrics.Precision(thresholds=0.5)
            self.recall = tf.keras.metrics.Recall(thresholds=0.5)
            self.f1 = tf.keras.metrics.F1Score(threshold=0.5)
            
        def update_state(self, y_true, y_pred, sample_weight=None):
            # Преобразуем входные данные в 2D
            batch_size = tf.shape(y_true)[0]
            y_true = tf.reshape(y_true, [-1, y_true.shape[-1]])
            y_pred = tf.reshape(y_pred, [-1, y_pred.shape[-1]])
            
            if sample_weight is not None:
                sample_weight = tf.reshape(sample_weight, [-1])
            
            # Обновляем метрики
            self.precision.update_state(y_true, y_pred, sample_weight)
            self.recall.update_state(y_true, y_pred, sample_weight)
            self.f1.update_state(y_true, y_pred, sample_weight)
            
        def result(self):
            return {
                'precision': self.precision.result(),
                'recall': self.recall.result(),
                'f1_score': self.f1.result()
            }
            
        def reset_state(self):
            self.precision.reset_state()
            self.recall.reset_state()
            self.f1.reset_state()
    
    return [
        'accuracy',
        SequenceMetrics(name='sequence_metrics'),
        ScalarF1Score(name='scalar_f1_score')
    ]

def get_tuning_metrics():
    """
    Получение метрик для подбора гиперпараметров
    """
    class SequenceMetrics(tf.keras.metrics.Metric):
        def __init__(self, name='sequence_metrics', **kwargs):
            super().__init__(name=name, **kwargs)
            self.precision = tf.keras.metrics.Precision(thresholds=0.5)
            self.recall = tf.keras.metrics.Recall(thresholds=0.5)
            self.f1 = tf.keras.metrics.F1Score(threshold=0.5)
            
        def update_state(self, y_true, y_pred, sample_weight=None):
            # Преобразуем входные данные в 2D
            batch_size = tf.shape(y_true)[0]
            y_true = tf.reshape(y_true, [-1, y_true.shape[-1]])
            y_pred = tf.reshape(y_pred, [-1, y_pred.shape[-1]])
            
            if sample_weight is not None:
                sample_weight = tf.reshape(sample_weight, [-1])
            
            # Обновляем метрики
            self.precision.update_state(y_true, y_pred, sample_weight)
            self.recall.update_state(y_true, y_pred, sample_weight)
            self.f1.update_state(y_true, y_pred, sample_weight)
            
        def result(self):
            return {
                'precision': self.precision.result(),
                'recall': self.recall.result(),
                'f1_score': self.f1.result()
            }
            
        def reset_state(self):
            self.precision.reset_state()
            self.recall.reset_state()
            self.f1.reset_state()
    
    return [
        'accuracy',
        SequenceMetrics(name='sequence_metrics'),
        ScalarF1Score(name='scalar_f1_score')
    ] 