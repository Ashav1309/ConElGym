import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, Dense, Bidirectional, LSTM, 
    GlobalAveragePooling2D, Reshape,
    Multiply, Conv2D, BatchNormalization,
    Activation, Dropout, TimeDistributed,
    GlobalAveragePooling1D, LayerNormalization, Add, DepthwiseConv2D, ReLU,
    Layer
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

logger = logging.getLogger(__name__)

def f1_score_element(y_true, y_pred):
    """
    Вычисление F1-score для элемента с учетом временной размерности и one-hot encoded меток
    """
    # Преобразуем one-hot encoded метки в индексы классов
    y_true = tf.argmax(y_true, axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1)
    
    # Вычисляем метрики с учетом временной размерности
    true_positives = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 1), tf.equal(y_pred, 1)), tf.float32))
    predicted_positives = tf.reduce_sum(tf.cast(tf.equal(y_pred, 1), tf.float32))
    possible_positives = tf.reduce_sum(tf.cast(tf.equal(y_true, 1), tf.float32))
    
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
        # Используем один и тот же вход для query и values
        query = x
        values = x
        
        # Вычисляем attention scores
        score = tf.nn.tanh(self.W1(query) + self.W2(values))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        
        # Применяем веса к значениям
        context_vector = attention_weights * values
        
        # Возвращаем только context_vector
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
            use_bias=False
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
            padding='same'
        )
        self.se_expand = tf.keras.layers.Conv2D(
            self.expanded_filters,
            kernel_size=1,
            padding='same',
            activation='sigmoid'
        )
        
        self.project_conv = tf.keras.layers.Conv2D(
            filters,
            kernel_size=1,
            padding='same',
            use_bias=False
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
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        print("[DEBUG] Очистка памяти после эпохи")

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

def create_mobilenetv4_model(input_shape, num_classes, dropout_rate=0.5, model_type='small', expansion_factor=4, se_ratio=0.25):
    try:
        print(f"\n[DEBUG] Инициализация MobileNetV4: input_shape={input_shape}, num_classes={num_classes}, dropout_rate={dropout_rate}")
        
        # Проверяем и корректируем input_shape
        if len(input_shape) == 5:  # Если есть лишняя размерность
            print(f"[DEBUG] Обнаружена лишняя размерность в input_shape: {input_shape}")
            input_shape = tuple(s for i, s in enumerate(input_shape) if i != 1 or s != 1)
            print(f"[DEBUG] Исправленный input_shape: {input_shape}")
        
        # Конфигурация для small модели
        config = {
            'initial_filters': 32,
            'blocks': [
                {'filters': 64, 'expansion': 4, 'stride': 2},
                {'filters': 128, 'expansion': 4, 'stride': 2},
                {'filters': 256, 'expansion': 4, 'stride': 2},
                {'filters': 512, 'expansion': 4, 'stride': 2},
            ]
        }
        
        def _create_model_operation():
            try:
                print("[DEBUG] Начало создания модели...")
                
                # Входной слой
                inputs = Input(shape=input_shape)
                print(f"[DEBUG] Форма входных данных после Input: {inputs.shape}")
                
                # Добавляем слой Reshape для гарантии правильной формы
                x = Reshape(input_shape)(inputs)
                print(f"[DEBUG] Форма после Reshape: {x.shape}")
                
                # Обработка последовательности кадров
                sequence_length = input_shape[0]
                height = input_shape[1]
                width = input_shape[2]
                channels = input_shape[3]
                
                print(f"[DEBUG] Начальные размерности: sequence_length={sequence_length}, height={height}, width={width}, channels={channels}")
                
                try:
                    # Начальный слой
                    x = TimeDistributed(Conv2D(config['initial_filters'], 3, strides=2, padding='same'))(x)
                    print(f"[DEBUG] После начального Conv2D: {x.shape}")
                    
                    x = TimeDistributed(BatchNormalization())(x)
                    x = TimeDistributed(ReLU())(x)
                    
                    # UIB блоки
                    for i, block in enumerate(config['blocks']):
                        try:
                            x = TimeDistributed(UniversalInvertedBottleneck(
                                filters=block['filters'],
                                expansion=block['expansion'],
                                stride=block['stride'],
                                se_ratio=se_ratio
                            ))(x)
                            print(f"[DEBUG] После UIB блока {i+1}: {x.shape}")
                        except Exception as e:
                            print(f"[ERROR] Ошибка в UIB блоке {i+1}: {str(e)}")
                            print(f"[DEBUG] Форма данных перед блоком: {x.shape}")
                            raise
                    
                    # Временная обработка
                    x = TimeDistributed(GlobalAveragePooling2D())(x)
                    print(f"[DEBUG] После GlobalAveragePooling2D: {x.shape}")
                    
                    x = Bidirectional(LSTM(64, return_sequences=True))(x)
                    x = Dropout(dropout_rate)(x)
                    x = Bidirectional(LSTM(32))(x)
                    x = Dropout(dropout_rate)(x)
                    
                    # Выходной слой
                    outputs = Dense(num_classes, activation='softmax')(x)
                    
                    model = Model(inputs=inputs, outputs=outputs)
                    model.compile(
                        optimizer=Adam(learning_rate=0.001),
                        loss='categorical_crossentropy',
                        metrics=['accuracy']
                    )
                    
                    print("[DEBUG] MobileNetV4 успешно создана")
                    return model
                    
                except Exception as e:
                    print(f"[ERROR] Ошибка при создании слоев модели: {str(e)}")
                    print(f"[DEBUG] Текущая форма данных: {x.shape if 'x' in locals() else 'не определена'}")
                    raise
                
            except Exception as e:
                print(f"[ERROR] Ошибка при создании MobileNetV4: {str(e)}")
                print(f"[ERROR] Stack trace: {traceback.format_exc()}")
                raise
                
        return _create_model_operation()
        
    except Exception as e:
        print(f"[ERROR] Критическая ошибка при инициализации MobileNetV4: {str(e)}")
        print(f"[ERROR] Stack trace: {traceback.format_exc()}")
        raise

def create_model(input_shape, num_classes, dropout_rate=0.5, lstm_units=64, model_type='v3', model_size='small', expansion_factor=4, se_ratio=0.25, positive_class_weight=200.0):
    """
    Создание модели с выбором типа архитектуры
    Args:
        input_shape: форма входных данных
        num_classes: количество классов
        dropout_rate: коэффициент dropout
        lstm_units: количество юнитов в LSTM слоях
        model_type: тип модели ('v3' или 'v4') - временно используется только v3
        model_size: размер модели ('small', 'medium', 'large')
        expansion_factor: коэффициент расширения для UIB блоков (только для v4)
        se_ratio: коэффициент для Squeeze-and-Excitation (только для v4)
        positive_class_weight: вес положительного класса
    """
    print(f"\n[DEBUG] ===== Создание модели =====")
    print(f"[DEBUG] Параметры:")
    print(f"  - model_type: {model_type} (временно используется только v3)")
    print(f"  - model_size: {model_size}")
    print(f"  - input_shape: {input_shape}")
    print(f"  - num_classes: {num_classes}")
    print(f"  - dropout_rate: {dropout_rate}")
    print(f"  - lstm_units: {lstm_units}")
    print(f"  - positive_class_weight: {positive_class_weight}")
    
    try:
        # Временно используем только MobileNetV3
        print("[DEBUG] Создание MobileNetV3...")
        model, class_weights = create_mobilenetv3_model(
            input_shape=input_shape,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            lstm_units=lstm_units,
            positive_class_weight=positive_class_weight
        )
        
        print("[DEBUG] Модель успешно создана")
        return model, class_weights
        
    except Exception as e:
        print(f"[ERROR] Ошибка при создании модели: {str(e)}")
        print("[DEBUG] Stack trace:", flush=True)
        traceback.print_exc()
        raise

def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return tf.reduce_sum(loss, axis=-1)
    return focal_loss_fixed

def create_mobilenetv3_model(input_shape, num_classes, dropout_rate=0.3, lstm_units=256, model_type='v3', model_size='small', positive_class_weight=200.0):
    """
    Создание модели MobileNetV3 с LSTM и оптимизацией памяти
    """
    try:
        print("\n[DEBUG] Создание модели MobileNetV3...")
        
        # Включаем mixed precision для оптимизации памяти
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("[DEBUG] Включен mixed precision")
        
        # Валидация входных параметров
        if not isinstance(input_shape, tuple) or len(input_shape) != 4:
            raise ValueError(f"Неверный формат input_shape: {input_shape}. Ожидается (sequence_length, height, width, channels)")
            
        if num_classes <= 0:
            raise ValueError(f"Количество классов должно быть положительным: {num_classes}")
            
        if not 0 <= dropout_rate <= 1:
            raise ValueError(f"Коэффициент dropout должен быть в диапазоне [0, 1]: {dropout_rate}")
            
        if lstm_units <= 0:
            raise ValueError(f"Количество LSTM юнитов должно быть положительным: {lstm_units}")
            
        if model_type not in ['v3', 'v4']:
            raise ValueError(f"Неверный тип модели: {model_type}. Допустимые значения: v3, v4")
            
        if model_size not in ['small', 'medium', 'large']:
            raise ValueError(f"Неверный размер модели: {model_size}. Допустимые значения: small, medium, large")
        
        # Получаем параметры модели
        model_params = Config.MODEL_PARAMS[model_type]
        
        # Создаем базовую модель MobileNetV3
        if model_type == 'v3':
            base_model = MobileNetV3Small(
                input_shape=input_shape[1:],
                alpha=1.0,
                minimalistic=False,
                include_top=False,
                weights='imagenet',
                pooling=None,
                classes=num_classes,
                classifier_activation=None
            )
        else:  # v4
            base_model = MobileNetV3Small(
                input_shape=input_shape[1:],
                alpha=1.0,
                minimalistic=False,
                include_top=False,
                weights='imagenet',
                pooling=None,
                classes=num_classes,
                classifier_activation=None,
                expansion_factor=model_params.get('expansion_factor', 4),
                se_ratio=model_params.get('se_ratio', 0.25)
            )
        
        # Создаем входной слой
        inputs = tf.keras.layers.Input(shape=input_shape)
        
        # Применяем базовую модель к каждому кадру
        x = tf.keras.layers.TimeDistributed(base_model)(inputs)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling2D())(x)
        
        # Добавляем LSTM слои с BatchNormalization
        x = tf.keras.layers.LSTM(lstm_units, return_sequences=True)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        
        x = tf.keras.layers.LSTM(lstm_units, return_sequences=True)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        
        # Добавляем выходной слой для каждого кадра
        outputs = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(num_classes, activation='softmax', dtype='float32')
        )(x)
        
        # Создаем модель
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Создаем метрики
        metrics = [
            'accuracy',
            tf.keras.metrics.Precision(name='precision_element', class_id=1),
            tf.keras.metrics.Recall(name='recall_element', class_id=1),
            tf.keras.metrics.F1Score(name='f1_score_element', threshold=0.5, num_classes=2)
        ]
        
        # Компилируем модель с focal loss и метриками
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE),
            loss=focal_loss(gamma=2., alpha=0.25),
            metrics=metrics
        )
        
        # Создаем словарь весов классов
        class_weights = {
            0: 1.0,  # вес для фонового класса
            1: positive_class_weight  # вес для класса элемента
        }
        
        print("[DEBUG] Модель MobileNetV3 успешно создана")
        return model, class_weights
        
    except Exception as e:
        print(f"[ERROR] Ошибка при создании модели: {str(e)}")
        print("[DEBUG] Stack trace:", flush=True)
        traceback.print_exc()
        raise

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