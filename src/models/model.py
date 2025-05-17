import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, Dense, Bidirectional, LSTM, GRU, 
    GlobalAveragePooling2D, Reshape,
    Multiply, Conv2D, BatchNormalization,
    Activation, Dropout, TimeDistributed,
    GlobalAveragePooling1D, LayerNormalization, Add, DepthwiseConv2D, ReLU,
    Layer, MultiHeadAttention, GlobalAveragePooling3D
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
import numpy as np
from src.models.losses import focal_loss, DynamicClassWeights, AdaptiveLearningRate
from src.data_proc.augmentation import BalancedDataGenerator
from tensorflow.keras.regularizers import l1_l2

logger = logging.getLogger(__name__)

def merge_classes(y):
    """
    Объединяет классы [1, 0] и [0, 1] в один положительный класс (1), фон — 0.
    y: shape (..., 2)
    Возвращает: shape (..., 1)
    """
    # Если хотя бы один из двух классов равен 1, то это положительный класс
    return tf.cast(tf.reduce_any(y == 1, axis=-1), tf.int32)

def f1_score_element(y_true, y_pred):
    """
    Вычисление F1-score для элемента с учетом временной размерности и one-hot encoded меток
    """
    # Объединяем классы
    y_true_bin = merge_classes(y_true)
    y_pred_bin = merge_classes(y_pred)
    
    true_positives = tf.reduce_sum(tf.cast((y_true_bin == 1) & (y_pred_bin == 1), tf.float32))
    predicted_positives = tf.reduce_sum(tf.cast(y_pred_bin == 1, tf.float32))
    possible_positives = tf.reduce_sum(tf.cast(y_true_bin == 1, tf.float32))
    
    # Добавляем epsilon для предотвращения деления на ноль
    epsilon = tf.keras.backend.epsilon()
    
    # Вычисляем precision и recall
    precision = true_positives / (predicted_positives + epsilon)
    recall = true_positives / (possible_positives + epsilon)
    
    # Вычисляем F1-score
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    
    return f1

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
                
                # Получаем генератор данных
                data_gen = self.data_loader.data_generator()
                
                # Обрабатываем первый батч для проверки размерностей
                try:
                    x_batch, y_batch = next(data_gen)
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
                    
                    # Создаем новый генератор с исправленными размерностями
                    def corrected_generator():
                        for x, y in data_gen:
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
                    test_gen = corrected_generator()
                    test_x, test_y = next(test_gen)
                    print(f"[DEBUG] Размерность после коррекции: {test_x.shape}")
                    
                    history = self.model.fit(
                        corrected_generator(),
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

class TemporalF1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score_element', threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Преобразуем one-hot encoded метки в индексы классов
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
        
        # Вычисляем метрики с учетом временной размерности
        true_positives = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 1), tf.equal(y_pred, 1)), tf.float32))
        false_positives = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 0), tf.equal(y_pred, 1)), tf.float32))
        false_negatives = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 1), tf.equal(y_pred, 0)), tf.float32))
        
        self.true_positives.assign_add(true_positives)
        self.false_positives.assign_add(false_positives)
        self.false_negatives.assign_add(false_negatives)

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + tf.keras.backend.epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + tf.keras.backend.epsilon())
        return 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())

    def reset_state(self):
        self.true_positives.assign(0)
        self.false_positives.assign(0)
        self.false_negatives.assign(0)

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
        return out

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(embed_dim),
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def create_mobilenetv3_model(input_shape, num_classes, dropout_rate=0.3, lstm_units=256, rnn_type='lstm', temporal_block_type='rnn', class_weights=None):
    """
    Создание модели MobileNetV3
    temporal_block_type: 'rnn', 'hybrid', '3d_attention', 'transformer'
    """
    print("[DEBUG] Создание модели MobileNetV3...")
    
    # Получаем параметры модели из конфигурации
    model_params = Config.MODEL_PARAMS['v3']
    
    # Используем параметры из конфигурации, если не указаны явно
    dropout_rate = dropout_rate or model_params['dropout_rate']
    lstm_units = lstm_units or model_params['lstm_units']
    
    # Загружаем веса классов из конфига
    if class_weights is None:
        print("[WARNING] Веса классов не найдены в конфиге. Используем веса по умолчанию.")
        class_weights = {
            'background': 1.0,
            'action': 4.299630443449974,
            'transition': 10.0
        }
    
    # Преобразуем веса в формат для TensorFlow
    tf_class_weights = {
        0: class_weights['background'],
        1: class_weights['action'],
        2: class_weights['transition']
    }
    
    print(f"[DEBUG] Используемые веса классов: {tf_class_weights}")
    
    # Создаем модель
    base_model = MobileNetV3Small(input_shape=input_shape[1:], include_top=False, weights='imagenet')
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(lstm_units, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Reshape((input_shape[0], lstm_units))(x) if len(input_shape) == 4 else x

    # Временной блок
    if temporal_block_type == 'rnn':
        if rnn_type == 'lstm':
            x = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
        else:  # bigru
            x = Bidirectional(GRU(lstm_units, return_sequences=True))(x)
    elif temporal_block_type == 'hybrid':
        if rnn_type == 'lstm':
            x = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
        else:  # bigru
            x = Bidirectional(GRU(lstm_units, return_sequences=True))(x)
        x = TemporalAttention(lstm_units)(x)
    elif temporal_block_type == '3d_attention':
        x = Reshape((input_shape[0], 1, 1, lstm_units))(x) if len(x.shape) == 3 else x
        x = SpatioTemporal3DAttention()(x)
        x = GlobalAveragePooling3D()(x)
    elif temporal_block_type == 'transformer':
        x = TransformerBlock(embed_dim=lstm_units, num_heads=4, ff_dim=128)(x)
    else:
        raise ValueError(f"Неизвестный temporal_block_type: {temporal_block_type}")

    x = GlobalAveragePooling1D()(x) if len(x.shape) == 3 else x
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=outputs)
    
    # Компилируем модель с весами
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    # Включаем mixed precision если используется GPU
    if Config.DEVICE_CONFIG['use_gpu'] and Config.MEMORY_OPTIMIZATION['use_mixed_precision']:
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    
    # Создаем метрики для трех классов
    metrics = [
        'accuracy',
        tf.keras.metrics.Precision(name='precision_background', class_id=0, thresholds=0.5),
        tf.keras.metrics.Precision(name='precision_action', class_id=1, thresholds=0.5),
        tf.keras.metrics.Precision(name='precision_transition', class_id=2, thresholds=0.5),
        tf.keras.metrics.Recall(name='recall_background', class_id=0, thresholds=0.5),
        tf.keras.metrics.Recall(name='recall_action', class_id=1, thresholds=0.5),
        tf.keras.metrics.Recall(name='recall_transition', class_id=2, thresholds=0.5)
    ]
    
    # Создаем адаптер для F1Score
    class F1ScoreAdapter(tf.keras.metrics.F1Score):
        def __init__(self, name, class_id, threshold=0.5):
            super().__init__(name=name, threshold=threshold)
            self.class_id = class_id
            
        def update_state(self, y_true, y_pred, sample_weight=None):
            y_true = tf.argmax(y_true, axis=-1)
            y_pred = tf.argmax(y_pred, axis=-1)
            y_true = tf.reshape(y_true, [-1])
            y_pred = tf.reshape(y_pred, [-1])
            y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=3)
            y_pred = tf.one_hot(tf.cast(y_pred, tf.int32), depth=3)
            return super().update_state(y_true, y_pred, sample_weight)
        
        def result(self):
            result = super().result()
            return result[self.class_id]
    
    # Добавляем F1Score для каждого класса
    metrics.extend([
        F1ScoreAdapter(name='f1_score_background', class_id=0, threshold=0.5),
        F1ScoreAdapter(name='f1_score_action', class_id=1, threshold=0.5),
        F1ScoreAdapter(name='f1_score_transition', class_id=2, threshold=0.5)
    ])
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=metrics,
        weighted_metrics=['accuracy'],
        class_weights=tf_class_weights
    )
    
    return model, tf_class_weights

def create_mobilenetv4_model(input_shape, num_classes, dropout_rate=0.5, class_weights=None):
    """
    Создает модель на основе MobileNetV4 с улучшенной архитектурой
    """
    print("[DEBUG] Создание модели MobileNetV4...")
    
    # Получаем параметры модели из конфигурации
    model_params = Config.MODEL_PARAMS['v4']
    
    # Используем параметры из конфигурации, если не указаны явно
    dropout_rate = dropout_rate or model_params['dropout_rate']
    
    # Загружаем веса классов из конфига
    if class_weights is None:
        print("[WARNING] Веса классов не найдены в конфиге. Используем веса по умолчанию.")
        class_weights = {
            'background': 1.0,
            'action': 4.299630443449974,
            'transition': 10.0
        }
    
    # Преобразуем веса в формат для TensorFlow
    tf_class_weights = {
        0: class_weights['background'],
        1: class_weights['action'],
        2: class_weights['transition']
    }
    
    print(f"[DEBUG] Используемые веса классов: {tf_class_weights}")
    
    # Создаем модель
    inputs = Input(shape=input_shape[1:])  # Убираем размерность последовательности
    x = inputs
    
    # Добавляем слои MobileNetV4
    x = Conv2D(32, (3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU(max_value=6.0)(x)
    
    # Добавляем инвертированные бутылочные слои
    x = UniversalInvertedBottleneck(64, kernel_size=3, strides=1)(x)
    x = UniversalInvertedBottleneck(128, kernel_size=3, strides=2)(x)
    x = UniversalInvertedBottleneck(128, kernel_size=3, strides=1)(x)
    x = UniversalInvertedBottleneck(256, kernel_size=3, strides=2)(x)
    x = UniversalInvertedBottleneck(256, kernel_size=3, strides=1)(x)
    x = UniversalInvertedBottleneck(512, kernel_size=3, strides=2)(x)
    x = UniversalInvertedBottleneck(512, kernel_size=3, strides=1)(x)
    
    # Добавляем временной блок
    x = Reshape((input_shape[0], -1))(x)  # Преобразуем в последовательность
    x = Bidirectional(LSTM(256, return_sequences=True))(x)
    x = GlobalAveragePooling1D()(x)
    
    # Добавляем слои классификации
    x = Dense(512, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Создаем модель
    model = Model(inputs=inputs, outputs=outputs)
    
    # Компилируем модель
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    # Включаем mixed precision если используется GPU
    if Config.DEVICE_CONFIG['use_gpu'] and Config.MEMORY_OPTIMIZATION['use_mixed_precision']:
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    
    # Создаем метрики
    metrics = [
        'accuracy',
        tf.keras.metrics.Precision(name='precision_background', class_id=0, thresholds=0.5),
        tf.keras.metrics.Precision(name='precision_action', class_id=1, thresholds=0.5),
        tf.keras.metrics.Precision(name='precision_transition', class_id=2, thresholds=0.5),
        tf.keras.metrics.Recall(name='recall_background', class_id=0, thresholds=0.5),
        tf.keras.metrics.Recall(name='recall_action', class_id=1, thresholds=0.5),
        tf.keras.metrics.Recall(name='recall_transition', class_id=2, thresholds=0.5)
    ]
    
    # Создаем адаптер для F1Score
    class F1ScoreAdapter(tf.keras.metrics.F1Score):
        def __init__(self, name, class_id, threshold=0.5):
            super().__init__(name=name, threshold=threshold)
            self.class_id = class_id
            
        def update_state(self, y_true, y_pred, sample_weight=None):
            y_true = tf.argmax(y_true, axis=-1)
            y_pred = tf.argmax(y_pred, axis=-1)
            y_true = tf.reshape(y_true, [-1])
            y_pred = tf.reshape(y_pred, [-1])
            y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=3)
            y_pred = tf.one_hot(tf.cast(y_pred, tf.int32), depth=3)
            return super().update_state(y_true, y_pred, sample_weight)
        
        def result(self):
            result = super().result()
            return result[self.class_id]
    
    # Добавляем F1Score для каждого класса
    metrics.extend([
        F1ScoreAdapter(name='f1_score_background', class_id=0, threshold=0.5),
        F1ScoreAdapter(name='f1_score_action', class_id=1, threshold=0.5),
        F1ScoreAdapter(name='f1_score_transition', class_id=2, threshold=0.5)
    ])
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=metrics,
        weighted_metrics=['accuracy'],
        class_weights=tf_class_weights
    )
    
    return model, tf_class_weights

def create_model(input_shape, num_classes, dropout_rate=0.5, lstm_units=64, model_type='v3', class_weights=None, rnn_type='lstm', temporal_block_type='rnn'):
    """
    Создание модели с заданными параметрами
    temporal_block_type: 'rnn' или 'hybrid'
    """
    print("\n[DEBUG] Создание модели...")
    print(f"[DEBUG] Параметры создания модели:")
    print(f"  - model_type: {model_type}")
    print(f"  - dropout_rate: {dropout_rate}")
    print(f"  - lstm_units: {lstm_units}")
    print(f"  - class_weights: {class_weights}")
    print(f"  - rnn_type: {rnn_type}")
    print(f"  - temporal_block_type: {temporal_block_type}")
    
    # Получаем параметры модели из конфигурации
    model_params = Config.MODEL_PARAMS[model_type]
    
    if model_type == 'v3':
        return create_mobilenetv3_model(
            input_shape=input_shape,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            lstm_units=lstm_units,
            rnn_type=rnn_type,
            temporal_block_type=temporal_block_type,
            class_weights=class_weights
        )
    elif model_type == 'v4':
        return create_mobilenetv4_model(
            input_shape=input_shape,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            class_weights=class_weights
        )
    else:
        raise ValueError(f"Неверный тип модели: {model_type}. Допустимые значения: v3, v4")

def focal_loss(gamma=2., alpha=None):
    """
    Focal loss с поддержкой разных alpha для каждого класса.
    alpha: массив длины num_classes или число (если одинаковый вес)
    """
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        # Если alpha не задан, используем 0.25 для всех классов
        if alpha is None:
            alpha_factor = tf.ones_like(y_true) * 0.25
        elif isinstance(alpha, (list, tuple, np.ndarray)):
            alpha_factor = tf.convert_to_tensor(alpha, dtype=tf.float32)
            alpha_factor = tf.reshape(alpha_factor, (1, 1, -1))
            alpha_factor = tf.ones_like(y_true) * alpha_factor
        else:
            alpha_factor = tf.ones_like(y_true) * float(alpha)
        
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha_factor * tf.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
    return focal_loss_fixed

def postprocess_predictions(preds, threshold=0.5):
    """
    Извлекает индексы кадров (или времени) начала и конца упражнения из предсказаний модели.
    preds: np.ndarray или tf.Tensor формы (frames, 2) или (batch, frames, 2)
    threshold: float, порог вероятности
    Возвращает: start_indices, end_indices (списки индексов кадров)
    """
    if isinstance(preds, tf.Tensor):
        preds = preds.numpy()
    if preds.ndim == 3:
        # Если батч, берём первый элемент
        preds = preds[0]
    start_indices = np.where(preds[:, 0] > threshold)[0]
    end_indices = np.where(preds[:, 1] > threshold)[0]
    return start_indices.tolist(), end_indices.tolist()

def indices_to_seconds(indices, fps):
    """
    Переводит индексы кадров в секунды по fps.
    indices: список индексов кадров
    fps: частота кадров (float или int)
    Возвращает: список секунд (float)
    """
    return [round(idx / fps, 3) for idx in indices]

class ActionDetectionModel:
    def __init__(self, input_shape=(224, 224, 3)):
        self.input_shape = input_shape
        self.model = self._build_model()
        
    def _build_model(self):
        # Базовая модель (ResNet50)
        base_model = tf.keras.applications.ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=self.input_shape
        )
        
        # Замораживаем веса базовой модели
        base_model.trainable = False
        
        # Добавляем слои для классификации
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(3, activation='sigmoid')(x)
        
        # Создаем модель
        model = Model(inputs=base_model.input, outputs=outputs)
        
        # Компилируем модель с улучшенным focal loss
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE),
            loss=focal_loss(gamma=2.0, alpha=0.25, beta=0.999),
            metrics=['accuracy', tf.keras.metrics.F1Score()]
        )
        
        return model
    
    def train(self, train_data, val_data, epochs=Config.EPOCHS):
        # Создаем генераторы данных с балансировкой
        train_generator = BalancedDataGenerator(
            train_data[0], train_data[1],
            batch_size=Config.BATCH_SIZE,
            augment=True
        )
        
        val_generator = BalancedDataGenerator(
            val_data[0], val_data[1],
            batch_size=Config.BATCH_SIZE,
            augment=False
        )
        
        # Создаем callbacks
        callbacks = [
            # Сохранение лучшей модели
            tf.keras.callbacks.ModelCheckpoint(
                filepath=Config.MODEL_PATH,
                monitor='val_f1_score',
                mode='max',
                save_best_only=True
            ),
            
            # Ранняя остановка
            tf.keras.callbacks.EarlyStopping(
                monitor='val_f1_score',
                patience=Config.EARLY_STOPPING_PATIENCE,
                mode='max'
            ),
            
            # Динамические веса классов
            DynamicClassWeights(
                validation_data=val_data,
                update_frequency=5
            ),
            
            # Адаптивное обучение
            AdaptiveLearningRate(
                class_metrics={
                    'background': {'f1_score': 0.0},
                    'action': {'f1_score': 0.0},
                    'transition': {'f1_score': 0.0}
                },
                patience=3
            )
        ]
        
        # Обучаем модель
        history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks
        )
        
        return history
    
    def predict(self, image):
        return self.model.predict(image)
    
    def save(self, path):
        self.model.save(path)
    
    @classmethod
    def load(cls, path):
        model = cls()
        model.model = tf.keras.models.load_model(
            path,
            custom_objects={'focal_loss_fixed': focal_loss()}
        )
        return model

if __name__ == "__main__":
    try:
        # Инициализация компонентов
        model, class_weights = create_model(
            input_shape=(16, 224, 224, 3),  # 16 кадров, размер 224x224, 3 канала
            num_classes=2,  # Начало и конец элемента
            model_type='v3'
        )
        model.summary()
        
    except Exception as e:
        logger.critical(f"Критическая ошибка: {str(e)}")
        # Сохранение состояния модели перед выходом
        if 'model' in locals():
            model.save('emergency_save.h5') 
