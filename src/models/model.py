import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, Dense, Bidirectional, LSTM, GRU, 
    GlobalAveragePooling2D, Reshape,
    Multiply, Conv2D, BatchNormalization,
    Activation, Dropout, TimeDistributed,
    GlobalAveragePooling1D, LayerNormalization, Add, DepthwiseConv2D, ReLU,
    Layer, MultiHeadAttention, GlobalAveragePooling3D,
    Conv3D, MaxPooling3D, Flatten, Permute,
    Concatenate, Lambda
)
from tensorflow.keras.applications import MobileNetV3Small
from src.utils.network_handler import NetworkErrorHandler, NetworkMonitor
import logging
import gc
from tensorflow.keras.optimizers import Adam
import traceback
from tensorflow.keras.metrics import Precision, Recall, F1Score
from tensorflow.keras.callbacks import Callback
from src.config import Config
from src.models.losses import focal_loss, DynamicClassWeights, AdaptiveLearningRate, F1ScoreAdapter
from src.data_proc.augmentation import BalancedDataGenerator
from tensorflow.keras.regularizers import l1_l2
import numpy as np
import json
import pickle
from src.models.metrics import get_training_metrics, get_tuning_metrics

logger = logging.getLogger(__name__)

__all__ = [
    'create_model_with_params',
    'create_mobilenetv3_model',
    'create_mobilenetv4_model',
    'postprocess_predictions',
    'indices_to_seconds',
    'merge_classes',
    'load_model_from_pickle'
]

def merge_classes(y):
    """
    Объединяет классы [1, 0] (фон) и [0, 1] (действие) в бинарные метки.
    y: shape (..., 2)
    Возвращает: shape (..., 1)
    """
    # Берем второй элемент (индекс 1) как класс действия
    return tf.cast(y[..., 1], tf.int32)

class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.kernel_size = kernel_size
        self.conv = None
        self.bn = None
        
    def build(self, input_shape):
        self.conv = Conv2D(1, self.kernel_size, padding='same')
        self.bn = BatchNormalization()
        super(SpatialAttention, self).build(input_shape)
        
    def compute_output_shape(self, input_shape):
        return input_shape
        
    def call(self, x):
        avg_out = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_out = tf.reduce_max(x, axis=-1, keepdims=True)
        x = tf.concat([avg_out, max_out], axis=-1)
        x = self.conv(x)
        x = self.bn(x)
        return tf.sigmoid(x)

class TemporalAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(TemporalAttention, self).__init__()
        self.units = units
        self.W1 = None
        self.W2 = None
        self.V = None
        
    def build(self, input_shape):
        self.W1 = Dense(self.units)
        self.W2 = Dense(self.units)
        self.V = Dense(1)
        super(TemporalAttention, self).build(input_shape)
        
    def compute_output_shape(self, input_shape):
        return input_shape
        
    def call(self, x):
        # Сохраняем входную форму
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        features = tf.shape(x)[2]
        
        # Вычисляем attention scores
        score = tf.nn.tanh(self.W1(x) + self.W2(x))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        
        # Применяем веса к значениям
        context_vector = attention_weights * x
        
        # Убеждаемся, что выходная форма совпадает с входной
        context_vector = tf.reshape(context_vector, [batch_size, seq_len, features])
        context_vector = tf.keras.layers.ReLU()(context_vector)
        
        return context_vector

class UniversalInvertedBottleneck(tf.keras.layers.Layer):
    """
    Универсальный инвертированный бутылочный слой
    """
    def __init__(self, filters, kernel_size=3, strides=1, expansion_factor=4, se_ratio=0.25, **kwargs):
        super(UniversalInvertedBottleneck, self).__init__(**kwargs)
        
        # Валидация входных параметров
        if filters <= 0:
            raise ValueError(f"Количество фильтров должно быть положительным: {filters}")
            
        if kernel_size <= 0:
            raise ValueError(f"Размер ядра должен быть положительным: {kernel_size}")
            
        if strides <= 0:
            raise ValueError(f"Шаг должен быть положительным: {strides}")
            
        if expansion_factor <= 0:
            raise ValueError(f"Фактор расширения должен быть положительным: {expansion_factor}")
            
        if not 0 <= se_ratio <= 1:
            raise ValueError(f"Коэффициент SE должен быть в диапазоне [0, 1]: {se_ratio}")
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.expansion_factor = expansion_factor
        self.se_ratio = se_ratio
        
        # Вычисляем количество каналов после расширения
        self.expanded_filters = int(filters * expansion_factor)
        
        # Создаем слои
        self.expand_conv = tf.keras.layers.Conv2D(
            self.expanded_filters,
            kernel_size=1,
            padding='same',
            use_bias=False,
            kernel_regularizer=l1_l2(l1=0.01, l2=0.01)
        )
        self.expand_bn = tf.keras.layers.BatchNormalization()
        self.expand_activation = tf.keras.layers.ReLU(max_value=6.0)
        
        self.depthwise_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            use_bias=False
        )
        self.depthwise_bn = tf.keras.layers.BatchNormalization()
        self.depthwise_activation = tf.keras.layers.ReLU(max_value=6.0)
        
        # Squeeze-and-Excitation блок
        self.se_reduce = tf.keras.layers.Conv2D(
            max(1, int(self.expanded_filters * se_ratio)),
            kernel_size=1,
            padding='same',
            kernel_regularizer=l1_l2(l1=0.01, l2=0.01)
        )
        self.se_expand = tf.keras.layers.Conv2D(
            self.expanded_filters,
            kernel_size=1,
            padding='same',
            activation='sigmoid',
            kernel_regularizer=l1_l2(l1=0.01, l2=0.01)
        )
        
        self.project_conv = tf.keras.layers.Conv2D(
            filters,
            kernel_size=1,
            padding='same',
            use_bias=False,
            kernel_regularizer=l1_l2(l1=0.01, l2=0.01)
        )
        self.project_bn = tf.keras.layers.BatchNormalization()
    
    def call(self, inputs, training=None):
        # Проверка входных размерностей
        if len(inputs.shape) != 4:
            raise ValueError(f"Неверная размерность входных данных: {inputs.shape}. Ожидается (batch_size, height, width, channels)")
        
        # Расширение каналов
        x = self.expand_conv(inputs)
        x = self.expand_bn(x, training=training)
        x = self.expand_activation(x)
        
        # Depthwise свертка
        x = self.depthwise_conv(x)
        x = self.depthwise_bn(x, training=training)
        x = self.depthwise_activation(x)
        
        # Squeeze-and-Excitation
        se = tf.keras.layers.GlobalAveragePooling2D()(x)
        se = self.se_reduce(se)
        se = tf.keras.layers.ReLU(max_value=6.0)(se)
        se = self.se_expand(se)
        se = tf.reshape(se, [-1, 1, 1, self.expanded_filters])
        x = tf.multiply(x, se)
        
        # Проекция
        x = self.project_conv(x)
        x = self.project_bn(x, training=training)
        
        # Добавляем skip connection если размерности совпадают
        if self.strides == 1 and inputs.shape[-1] == self.filters:
            x = tf.keras.layers.Add()([x, inputs])
        
        # Увеличиваем dropout
        x = Dropout(0.5)(x)  # Увеличиваем dropout для предотвращения переобучения
        
        return x
    
    def get_config(self):
        config = super(UniversalInvertedBottleneck, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'expansion_factor': self.expansion_factor,
            'se_ratio': self.se_ratio
        })
        return config

    def compute_output_shape(self, input_shape):
        # input_shape: (batch, height, width, channels)
        return (input_shape[0], input_shape[1], input_shape[2], self.filters)

class MemoryClearCallback(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        gc.collect()
        print("[DEBUG] Очистка памяти перед эпохой")

class ModelTrainer:
    def __init__(self, model, data_loader):
        self.model = model
        self.data_loader = data_loader
        self.network_handler = NetworkErrorHandler()
        self.network_monitor = NetworkMonitor()
        
    def train(self, epochs, batch_size):
        """
        Обучение модели с обработкой сетевых ошибок
        """
        def _train_operation():
            try:
                print("[DEBUG] Начало обучения модели...")
                
                # Получаем генераторы данных для обучения и валидации
                train_gen = self.data_loader.data_generator(force_positive=True, is_validation=False)
                val_gen = self.data_loader.data_generator(force_positive=True, is_validation=True)
                
                # Обрабатываем первый батч для проверки размерностей
                try:
                    x_batch, y_batch = next(train_gen)
                    print(f"[DEBUG] Размерность входных данных из генератора: {x_batch.shape}")
                    
                    # Функция для корректировки размерностей
                    def correct_dimensions(x):
                        # Преобразуем в тензор, если это еще не тензор
                        if not isinstance(x, tf.Tensor):
                            x = tf.convert_to_tensor(x)
                        
                        # Получаем текущую форму
                        shape = x.shape.as_list()
                        print(f"[DEBUG] Текущая форма данных: {shape}")
                        
                        # Проверяем количество измерений
                        if len(shape) == 6:
                            print(f"[DEBUG] Обнаружена лишняя размерность: {shape}")
                            # Находим ось с размером 1
                            squeeze_axis = [i for i, s in enumerate(shape) if s == 1]
                            if squeeze_axis:
                                print(f"[DEBUG] Удаляем оси: {squeeze_axis}")
                                x = tf.squeeze(x, axis=squeeze_axis)
                            else:
                                # Если нет осей с размером 1, используем reshape
                                target_shape = [shape[0], shape[2], shape[3], shape[4], shape[5]]
                                print(f"[DEBUG] Преобразуем в форму: {target_shape}")
                                x = tf.reshape(x, target_shape)
                        
                        print(f"[DEBUG] Итоговая форма: {x.shape}")
                        return x
                    
                    # Создаем новые генераторы с исправленными размерностями
                    def corrected_generator(gen):
                        for x, y in gen:
                            try:
                                x = correct_dimensions(x)
                                # Проверяем форму после коррекции
                                if len(x.shape) != 5:
                                    print(f"[ERROR] Неверная размерность после коррекции: {x.shape}")
                                    print(f"[ERROR] Ожидаемая размерность: (batch_size, sequence_length, height, width, channels)")
                                    raise ValueError(f"Неверная размерность после коррекции: {x.shape}")
                                yield x, y
                            except Exception as e:
                                print(f"[ERROR] Ошибка при обработке батча: {str(e)}")
                                print(f"[DEBUG] Stack trace: {traceback.format_exc()}")
                                raise
                    
                    # Проверяем первый батч после коррекции
                    train_gen_corrected = corrected_generator(train_gen)
                    val_gen_corrected = corrected_generator(val_gen)
                    test_x, test_y = next(train_gen_corrected)
                    print(f"[DEBUG] Размерность после коррекции: {test_x.shape}")
                    
                    history = self.model.fit(
                        train_gen_corrected,
                        validation_data=val_gen_corrected,
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[
                            tf.keras.callbacks.EarlyStopping(
                                monitor='val_loss',
                                patience=5,
                                restore_best_weights=True
                            ),
                            tf.keras.callbacks.ModelCheckpoint(
                                'best_model.h5',
                                save_best_only=True
                            ),
                            MemoryClearCallback()
                        ]
                    )
                    return history
                    
                except Exception as e:
                    print(f"[ERROR] Ошибка при обработке данных: {str(e)}")
                    print(f"[DEBUG] Stack trace: {traceback.format_exc()}")
                    raise
                
            except tf.errors.ResourceExhaustedError as e:
                logger.error(f"Недостаточно памяти GPU: {str(e)}")
                tf.keras.backend.clear_session()
                gc.collect()
                raise
                
            except Exception as e:
                logger.error(f"Ошибка при обучении: {str(e)}")
                print(f"[DEBUG] Stack trace: {traceback.format_exc()}")
                raise
                
        return self.network_handler.handle_network_operation(_train_operation)

class GradientAccumulationModel(tf.keras.Model):
    """
    Модель с поддержкой градиентной аккумуляции
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradient_accumulation_steps = Config.GRADIENT_ACCUMULATION['steps']
        self._accumulated_gradients = None
        self._train_counter = tf.Variable(0, trainable=False, dtype=tf.int32)
        
    @tf.function
    def train_step(self, data):
        """
        Стандартный шаг обучения без градиентной аккумуляции.
        
        Args:
            data: Кортеж (x, y) или (x, y, sample_weight)
            
        Returns:
            dict: Словарь с метриками
        """
        try:
            print("\n[DEBUG] ===== Шаг обучения =====")
            print(f"[DEBUG] Тип входных данных: {type(data)}")
            
            # Распаковываем данные
            if len(data) == 2:
                x, y = data
                sample_weight = None
                print("[DEBUG] Распакованы x и y")
            elif len(data) == 3:
                x, y, sample_weight = data
                print("[DEBUG] Распакованы x, y и sample_weight")
            else:
                raise ValueError(f"Неожиданный формат данных: {len(data)} элементов")
            
            print(f"[DEBUG] Форма x: {x.shape}")
            print(f"[DEBUG] Форма y: {y.shape}")
            if sample_weight is not None:
                print(f"[DEBUG] Форма sample_weight: {sample_weight.shape}")
            
            # Вычисляем градиенты
            with tf.GradientTape() as tape:
                predictions = self(x, training=True)
                print(f"[DEBUG] Форма predictions: {predictions.shape}")
                loss = self.compute_loss(x, y, predictions, sample_weight)
                print(f"[DEBUG] Значение loss: {loss}")
            
            # Вычисляем и применяем градиенты
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            print("[DEBUG] Градиенты применены")
            
            # Обновляем метрики
            print("[DEBUG] Обновление метрик:")
            for metric in self.metrics:
                print(f"[DEBUG] Обновление метрики: {metric.name}")
                try:
                    if metric.name == 'loss':
                        # Для метрики loss используем только значение loss
                        metric.update_state(loss)
                    else:
                        # Для остальных метрик используем y и predictions
                        if sample_weight is None:
                            metric.update_state(y, predictions)
                        else:
                            metric.update_state(y, predictions, sample_weight)
                    print(f"[DEBUG] Метрика {metric.name} обновлена")
                except Exception as e:
                    print(f"[ERROR] Ошибка при обновлении метрики {metric.name}: {str(e)}")
                    raise
            
            # Возвращаем метрики
            metrics_dict = {m.name: m.result() for m in self.metrics}
            print(f"[DEBUG] Результаты метрик: {metrics_dict}")
            return metrics_dict
            
        except Exception as e:
            print(f"[ERROR] Ошибка при вычислении градиента: {str(e)}")
            print("[DEBUG] Stack trace:", flush=True)
            import traceback
            traceback.print_exc()
            raise

class SpatioTemporal3DAttention(tf.keras.layers.Layer):
    """
    Кастомный слой 3D Spatio-Temporal Attention для видео:
    - Внимание одновременно по времени и пространству (sequence, height, width)
    - Работает с входом формы (batch, seq, h, w, c)
    """
    def __init__(self, num_heads=4, key_dim=32, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.attn = None
        self.norm = None
    def build(self, input_shape):
        # Объединяем seq, h, w в одну ось для attention
        self.attn = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dim)
        self.norm = LayerNormalization()
        super().build(input_shape)
    def call(self, x):
        # x: (batch, seq, h, w, c)
        b, t, h, w, c = tf.unstack(tf.shape(x))
        x_flat = tf.reshape(x, [b, t * h * w, c])
        attn_out = self.attn(x_flat, x_flat)
        attn_out = self.norm(x_flat + attn_out)
        out = tf.reshape(attn_out, [b, t, h, w, c])
        out = tf.keras.layers.ReLU()(out)
        return out

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, rnn_type='lstm'):
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.rnn_type = rnn_type
        
    def build(self, input_shape):
        # Определяем размерность в зависимости от типа RNN
        if self.rnn_type == 'bigru':
            # Для BiGRU размерность в 2 раза больше
            key_dim = input_shape[-1] // (2 * self.num_heads)
            value_dim = input_shape[-1] // (2 * self.num_heads)
            output_dim = input_shape[-1] // 2
        else:
            # Для LSTM используем стандартные размерности
            key_dim = self.embed_dim // self.num_heads
            value_dim = self.embed_dim // self.num_heads
            output_dim = self.embed_dim
            
        # Создаем слои с правильными размерностями
        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=key_dim,
            value_dim=value_dim,
            output_shape=output_dim
        )
        
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(self.ff_dim, activation="relu"),
            tf.keras.layers.Dense(input_shape[-1])  # Используем размерность входных данных
        ])
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(self.rate)
        self.dropout2 = tf.keras.layers.Dropout(self.rate)
        super(TransformerBlock, self).build(input_shape)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def compute_output_shape(self, input_shape):
        return input_shape

class TemporalConvNet(tf.keras.layers.Layer):
    """
    Временная сверточная сеть (TCN) для обработки последовательностей
    """
    def __init__(self, num_channels, kernel_size=3, dropout=0.2, **kwargs):
        super(TemporalConvNet, self).__init__(**kwargs)
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.layers = []
        
    def build(self, input_shape):
        # Создаем слои TCN
        for i, num_channels in enumerate(self.num_channels):
            dilation_rate = 2 ** i
            self.layers.append([
                tf.keras.layers.Conv1D(
                    filters=num_channels,
                    kernel_size=self.kernel_size,
                    dilation_rate=dilation_rate,
                    padding='causal',
                    activation='relu'
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(self.dropout)
            ])
        super(TemporalConvNet, self).build(input_shape)
        
    def call(self, inputs, training=None):
        x = inputs
        for layers in self.layers:
            residual = x
            for layer in layers:
                x = layer(x, training=training)
            # Добавляем skip connection если размерности совпадают
            if residual.shape[-1] == x.shape[-1]:
                x = x + residual
        return x
    
    def get_config(self):
        config = super(TemporalConvNet, self).get_config()
        config.update({
            'num_channels': self.num_channels,
            'kernel_size': self.kernel_size,
            'dropout': self.dropout
        })
        return config

class SequenceFBetaScore(tf.keras.metrics.FBetaScore):
    """
    Адаптированная версия FBetaScore для работы с последовательностями
    """
    def __init__(self, name='sequence_fbeta', beta=1.0, threshold=0.5, **kwargs):
        super().__init__(name=name, beta=beta, threshold=threshold, **kwargs)
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Преобразуем входные данные в 2D
        batch_size = tf.shape(y_true)[0]
        y_true = tf.reshape(y_true, [-1, y_true.shape[-1]])
        y_pred = tf.reshape(y_pred, [-1, y_pred.shape[-1]])
        
        if sample_weight is not None:
            sample_weight = tf.reshape(sample_weight, [-1])
            
        super().update_state(y_true, y_pred, sample_weight)

class TemporalBlock(tf.keras.layers.Layer):
    def __init__(self, num_heads=2, key_dim=64, **kwargs):  # Уменьшаем количество голов и размерность ключа
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            attention_axes=(1,),  # Применяем внимание только по временной оси
            output_shape=key_dim
        )
        self.norm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

def create_mobilenetv3_model(input_shape, num_classes=2, dropout_rate=0.3, lstm_units=128, class_weights=None, rnn_type='lstm', temporal_block_type='rnn'):
    """
    Создание модели на основе MobileNetV3
    
    Args:
        input_shape: форма входных данных
        num_classes: количество классов
        dropout_rate: коэффициент dropout
        lstm_units: количество юнитов в LSTM
        class_weights: веса классов
        rnn_type: тип RNN ('lstm', 'bigru')
        temporal_block_type: тип временного блока ('rnn', 'tcn', '3d_attention', 'transformer')
    """
    if class_weights is None:
        class_weights = {
            'background': 1.0,
            'action': 10
        }
    
    tf_class_weights = {
        0: class_weights['background'],  # фон
        1: class_weights['action']       # действие
    }
    
    print(f"[DEBUG] Используемые веса классов: {tf_class_weights}")
    print(f"[DEBUG] Входная форма: {input_shape}")
    print(f"[DEBUG] Тип RNN: {rnn_type}")
    print(f"[DEBUG] Тип временного блока: {temporal_block_type}")
    
    # Извлекаем размерность изображения из input_shape
    if len(input_shape) == 4:  # (sequence_length, height, width, channels)
        image_shape = input_shape[1:]  # (height, width, channels)
        sequence_length = input_shape[0]
    else:  # (height, width, channels)
        image_shape = input_shape
        sequence_length = Config.SEQUENCE_LENGTH    # значение по умолчанию
    
    print(f"[DEBUG] Размерность изображения: {image_shape}")
    print(f"[DEBUG] Длина последовательности: {sequence_length}")
    
    # Создаем базовую модель MobileNetV3
    base_model = MobileNetV3Small(
        input_shape=image_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Замораживаем веса базовой модели
    base_model.trainable = False
    
    # Создаем входной слой
    inputs = Input(shape=input_shape)
    
    # Применяем базовую модель к каждому кадру последовательности
    x = TimeDistributed(base_model)(inputs)
    print(f"[DEBUG] Форма после MobileNetV3: {x.shape}")
    
    # Преобразуем выход MobileNetV3 в последовательность для временных блоков
    x = Reshape((sequence_length, -1))(x)
    print(f"[DEBUG] Форма после Reshape: {x.shape}")
    
    # Добавляем регуляризацию
    regularizer = tf.keras.regularizers.l2(0.01)
    
    # Добавляем RNN слой в зависимости от типа
    if rnn_type == 'lstm':
        x = LSTM(lstm_units, return_sequences=True, 
                kernel_regularizer=regularizer,
                recurrent_regularizer=regularizer)(x)
    elif rnn_type == 'bigru':
        x = Bidirectional(GRU(lstm_units, return_sequences=True,
                            kernel_regularizer=regularizer,
                            recurrent_regularizer=regularizer))(x)
    else:
        raise ValueError(f"Неизвестный тип RNN: {rnn_type}")
    print(f"[DEBUG] Форма после RNN: {x.shape}")
    
    # Добавляем временной блок в зависимости от типа
    if temporal_block_type == 'rnn':
        # Используем уже добавленный RNN слой
        pass
    elif temporal_block_type == 'tcn':
        # Комбинируем RNN с TCN
        x = TemporalConvNet(num_channels=[lstm_units, lstm_units//2], kernel_size=3, dropout=dropout_rate)(x)
    elif temporal_block_type == '3d_attention':
        # Преобразуем в 3D форму для пространственно-временного внимания
        import numpy as np
        spatial_dim = int(np.ceil(np.sqrt(x.shape[-1])))
        new_features = spatial_dim * spatial_dim
        x = Dense(new_features, kernel_regularizer=regularizer)(x)  # Приводим к квадрату
        x = Reshape((sequence_length, spatial_dim, spatial_dim, 1))(x)
        x = SpatioTemporal3DAttention(num_heads=4, key_dim=32)(x)
        x = Reshape((sequence_length, -1))(x)
    elif temporal_block_type == 'transformer':
        # Используем трансформерный блок с учетом типа RNN
        x = TransformerBlock(
            embed_dim=lstm_units,
            num_heads=2,  # Уменьшаем количество голов
            ff_dim=lstm_units,  # Уменьшаем размер FF слоя
            rate=dropout_rate,
            rnn_type=rnn_type  # Передаем тип RNN
        )(x)
    else:
        raise ValueError(f"Неизвестный тип временного блока: {temporal_block_type}")
    print(f"[DEBUG] Форма после временного блока: {x.shape}")
    
    # Добавляем слой нормализации
    x = BatchNormalization()(x)
    print(f"[DEBUG] Форма после BatchNorm: {x.shape}")
    
    # Увеличиваем dropout
    x = Dropout(dropout_rate + 0.2)(x)  # Увеличиваем dropout

    # Добавляем выходной слой для двух классов
    outputs = Dense(2, activation='sigmoid', kernel_regularizer=regularizer)(x)  # 2 класса: фон и действие
    print(f"[DEBUG] Выходная форма: {outputs.shape}")
    
    # Создаем модель
    model = Model(inputs=inputs, outputs=outputs)
    
    # Оптимизатор с меньшим learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)  # Уменьшаем learning rate
    
    # Метрики для двухклассовой модели
    metrics = get_training_metrics()
    
    model.compile(
        optimizer=optimizer,
        loss=focal_loss(gamma=2.0, alpha=[class_weights['background'], class_weights['action']]),
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision_action', class_id=1, thresholds=0.5),
            tf.keras.metrics.Recall(name='recall_action', class_id=1, thresholds=0.5),
            F1ScoreAdapter(name='f1_action', class_id=1, threshold=0.5)
        ]
    )
    
    return model

def create_mobilenetv4_model(input_shape, num_classes=2, dropout_rate=0.3, class_weights=None):
    """
    Создание модели на основе MobileNetV4
    """
    print("[DEBUG] Создание модели MobileNetV4...")
    
    # Извлекаем размерность изображения из input_shape
    if len(input_shape) == 4:  # (sequence_length, height, width, channels)
        image_shape = input_shape[1:]  # (height, width, channels)
        sequence_length = input_shape[0]
    else:  # (height, width, channels)
        image_shape = input_shape
        sequence_length = Config.SEQUENCE_LENGTH  # значение по умолчанию
    
    print(f"[DEBUG] Размерность изображения: {image_shape}")
    print(f"[DEBUG] Длина последовательности: {sequence_length}")
    
    # Загружаем веса классов из конфига
    if class_weights is None:
        print("[WARNING] Веса классов не найдены в конфиге. Используем веса по умолчанию.")
        class_weights = {
            'background': 1.0,
            'action': 10
        }
    
    # Преобразуем веса в формат для TensorFlow
    tf_class_weights = {
        0: class_weights['background'],
        1: class_weights['action']
    }
    
    print(f"[DEBUG] Используемые веса классов: {tf_class_weights}")
    
    # Создаем модель
    inputs = Input(shape=input_shape)
    
    # Применяем слои MobileNetV4 к каждому кадру последовательности
    x = TimeDistributed(Conv2D(32, (3, 3), strides=(2, 2), padding='same'))(inputs)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(ReLU(max_value=6.0))(x)
    
    # Добавляем инвертированные бутылочные слои
    x = TimeDistributed(UniversalInvertedBottleneck(64, kernel_size=3, strides=1))(x)
    x = TimeDistributed(UniversalInvertedBottleneck(128, kernel_size=3, strides=2))(x)
    x = TimeDistributed(UniversalInvertedBottleneck(128, kernel_size=3, strides=1))(x)
    x = TimeDistributed(UniversalInvertedBottleneck(256, kernel_size=3, strides=2))(x)
    x = TimeDistributed(UniversalInvertedBottleneck(256, kernel_size=3, strides=1))(x)
    x = TimeDistributed(UniversalInvertedBottleneck(512, kernel_size=3, strides=2))(x)
    x = TimeDistributed(UniversalInvertedBottleneck(512, kernel_size=3, strides=1))(x)
    
    # Преобразуем в последовательность
    x = Reshape((sequence_length, -1))(x)
    
    # Добавляем временной блок
    x = LSTM(256, return_sequences=True)(x)
    x = GlobalAveragePooling1D()(x)
    
    # Добавляем слои классификации
    x = Dense(512, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(2, activation='softmax')(x)  # 2 класса: фон и действие
    
    # Создаем модель
    model = Model(inputs=inputs, outputs=outputs)
    
    # Оптимизатор
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    # Метрики для двухклассовой модели
    metrics = get_training_metrics()
    
    model.compile(
        optimizer=optimizer,
        loss=focal_loss(gamma=2.0, alpha=[class_weights['background'], class_weights['action']]),
        metrics=metrics
    )
    
    return model

def create_model_with_params(model_type, input_shape, num_classes, params, class_weights):
    """
    Создание модели с заданными параметрами
    
    Args:
        model_type: тип модели ('v3' или 'v4')
        input_shape: форма входных данных
        num_classes: количество классов
        params: словарь с параметрами модели
        class_weights: словарь с весами классов
    """
    print("\n[DEBUG] Создание модели с параметрами:")
    print(f"  - Тип модели: {model_type}")
    print(f"  - Тип модели (lower): {model_type.lower()}")
    print(f"  - Dropout: {params['dropout_rate']}")
    print(f"  - LSTM units: {params.get('lstm_units', 'N/A')}")
    print(f"  - RNN type: {params.get('rnn_type', 'lstm')}")
    print(f"  - Temporal block type: {params.get('temporal_block_type', 'rnn')}")
    print(f"  - Веса классов: {class_weights}")
    
    if model_type.lower() == 'v3':
        print("[DEBUG] Создание модели MobileNetV3...")
        return create_mobilenetv3_model(
            input_shape=input_shape,
            num_classes=num_classes,
            dropout_rate=params['dropout_rate'],
            lstm_units=params.get('lstm_units', 128),
            class_weights=class_weights,
            rnn_type=params.get('rnn_type', 'lstm'),
            temporal_block_type=params.get('temporal_block_type', 'rnn')
        )
    elif model_type.lower() == 'v4':
        print("[DEBUG] Создание модели MobileNetV4...")
        return create_mobilenetv4_model(
            input_shape=input_shape,
            num_classes=num_classes,
            dropout_rate=params['dropout_rate'],
            class_weights=class_weights
        )
    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}. Поддерживаемые типы: 'v3', 'v4'")

def postprocess_predictions(preds, threshold=0.5):
    """
    Извлекает индексы кадров с действием из предсказаний модели.
    preds: np.ndarray или tf.Tensor формы (frames, 2) или (batch, frames, 2)
    threshold: float, порог вероятности для класса действия
    Возвращает: список индексов кадров с действием
    """
    if isinstance(preds, tf.Tensor):
        preds = preds.numpy()
    if preds.ndim == 3:
        # Если батч, берём первый элемент
        preds = preds[0]
    # Берем вероятности для класса действия (индекс 1)
    action_probs = preds[:, 1]
    # Находим кадры, где вероятность действия выше порога
    action_indices = np.where(action_probs > threshold)[0]
    return action_indices.tolist()

def indices_to_seconds(indices, fps):
    """
    Переводит индексы кадров в секунды по fps.
    indices: список индексов кадров
    fps: частота кадров (float или int)
    Возвращает: список секунд (float)
    """
    return [round(idx / fps, 3) for idx in indices]

def load_model_from_pickle(filepath):
    """
    Загружает модель и метаданные из pickle файла
    
    Args:
        filepath: путь к pickle файлу с моделью
        
    Returns:
        tuple: (model, metadata) где metadata - словарь с метаданными модели
    """
    try:
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
            
        model = model_data['model']
        metadata = {
            'epoch': model_data['epoch'],
            'best_metric': model_data['best_metric'],
            'monitor': model_data['monitor'],
            'mode': model_data['mode'],
            'logs': model_data['logs']
        }
        
        logger.info(f"Модель успешно загружена из {filepath}")
        logger.info(f"Лучшая метрика: {metadata['best_metric']:.4f}")
        logger.info(f"Эпоха: {metadata['epoch']}")
        
        return model, metadata
        
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели из {filepath}: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Инициализация компонентов
        model, class_weights = create_model_with_params(
            model_type='v3',
            input_shape=(16, 224, 224, 3),  # 16 кадров, размер 224x224, 3 канала
            num_classes=2,  # Начало и конец элемента
            params={
                'dropout_rate': 0.3,
                'lstm_units': 128
            },
            class_weights=class_weights
        )
        model.summary()
        
    except Exception as e:
        logger.critical(f"Критическая ошибка: {str(e)}")
        # Сохранение состояния модели перед выходом
        if 'model' in locals():
            model.save('emergency_save.h5') 
