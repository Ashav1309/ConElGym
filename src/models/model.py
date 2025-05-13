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

logger = logging.getLogger(__name__)

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

class UniversalInvertedBottleneck(Layer):
    def __init__(self, filters, expansion=4, stride=1, se_ratio=0.25, **kwargs):
        super(UniversalInvertedBottleneck, self).__init__(**kwargs)
        self.filters = filters
        self.expansion = expansion
        self.stride = stride
        self.se_ratio = se_ratio
        
    def build(self, input_shape):
        self.expanded_filters = int(input_shape[-1] * self.expansion)
        
        # Расширяющий слой
        self.expand_conv = Conv2D(
            self.expanded_filters,
            kernel_size=1,
            padding='same',
            use_bias=False
        )
        self.expand_bn = BatchNormalization()
        self.expand_activation = ReLU()
        
        # Depthwise слой
        self.depthwise_conv = DepthwiseConv2D(
            kernel_size=3,
            strides=self.stride,
            padding='same',
            use_bias=False
        )
        self.depthwise_bn = BatchNormalization()
        self.depthwise_activation = ReLU()
        
        # Squeeze-and-Excitation
        self.se_reduce = Conv2D(
            max(1, int(self.expanded_filters * self.se_ratio)),
            kernel_size=1,
            activation='relu',
            padding='same'
        )
        self.se_expand = Conv2D(
            self.expanded_filters,
            kernel_size=1,
            activation='sigmoid',
            padding='same'
        )
        
        # Проецирующий слой
        self.project_conv = Conv2D(
            self.filters,
            kernel_size=1,
            padding='same',
            use_bias=False
        )
        self.project_bn = BatchNormalization()
        
        # Skip connection
        self.use_residual = self.stride == 1 and input_shape[-1] == self.filters
        
    def call(self, inputs):
        x = self.expand_conv(inputs)
        x = self.expand_bn(x)
        x = self.expand_activation(x)
        
        x = self.depthwise_conv(x)
        x = self.depthwise_bn(x)
        x = self.depthwise_activation(x)
        
        # Squeeze-and-Excitation
        se = GlobalAveragePooling2D()(x)
        se = self.se_reduce(se)
        se = self.se_expand(se)
        x = Multiply()([x, se])
        
        x = self.project_conv(x)
        x = self.project_bn(x)
        
        if self.use_residual:
            x = Add()([x, inputs])
            
        return x
        
    def compute_output_shape(self, input_shape):
        if self.stride > 1:
            height = input_shape[1] // self.stride
            width = input_shape[2] // self.stride
        else:
            height = input_shape[1]
            width = input_shape[2]
        return (input_shape[0], height, width, self.filters)

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
                            )
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

def create_model(input_shape, num_classes, dropout_rate=0.5, lstm_units=64, model_type='v3', model_size='small', expansion_factor=4, se_ratio=0.25):
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
    """
    print(f"\n[DEBUG] ===== Создание модели =====")
    print(f"[DEBUG] Параметры:")
    print(f"  - model_type: {model_type} (временно используется только v3)")
    print(f"  - model_size: {model_size}")
    print(f"  - input_shape: {input_shape}")
    print(f"  - num_classes: {num_classes}")
    print(f"  - dropout_rate: {dropout_rate}")
    print(f"  - lstm_units: {lstm_units}")
    
    try:
        # Временно используем только MobileNetV3
        print("[DEBUG] Создание MobileNetV3...")
        model = create_mobilenetv3_model(
            input_shape=input_shape,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            lstm_units=lstm_units
        )
        
        print("[DEBUG] Модель успешно создана")
        return model
        
    except Exception as e:
        print(f"[ERROR] Ошибка при создании модели: {str(e)}")
        print("[DEBUG] Stack trace:", flush=True)
        traceback.print_exc()
        raise

def create_mobilenetv3_model(input_shape, num_classes, dropout_rate=0.5, lstm_units=64):
    """
    Создает модель MobileNetV3 с LSTM для обработки временных последовательностей.
    
    Args:
        input_shape: Форма входных данных (sequence_length, height, width, channels)
        num_classes: Количество классов
        dropout_rate: Коэффициент dropout
        lstm_units: Количество юнитов в LSTM слое (по умолчанию 64)
    """
    try:
        print(f"\n[DEBUG] Инициализация MobileNetV3: input_shape={input_shape}, num_classes={num_classes}, dropout_rate={dropout_rate}, lstm_units={lstm_units}")
        
        # Проверяем и устанавливаем значение по умолчанию для lstm_units
        if lstm_units is None:
            lstm_units = 64
            print(f"[DEBUG] lstm_units установлен по умолчанию: {lstm_units}")
        
        def _create_model_operation():
            try:
                # Входной слой
                inputs = Input(shape=input_shape)
                print(f"[DEBUG] Форма входных данных после Input: {inputs.shape}")
                
                # Базовый MobileNetV3
                base_model = MobileNetV3Small(
                    input_shape=input_shape[1:],
                    include_top=False,
                    weights='imagenet'
                )
                
                # Замораживаем веса базовой модели
                base_model.trainable = False
                
                # Обработка последовательности
                x = TimeDistributed(base_model)(inputs)
                x = TimeDistributed(GlobalAveragePooling2D())(x)
                
                # LSTM для временной последовательности
                x = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
                x = Dropout(dropout_rate)(x)
                x = Bidirectional(LSTM(lstm_units))(x)
                x = Dropout(dropout_rate)(x)
                
                # Выходной слой
                outputs = Dense(num_classes, activation='softmax')(x)
                
                model = Model(inputs=inputs, outputs=outputs)
                model.compile(
                    optimizer=Adam(learning_rate=0.001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                print("[DEBUG] MobileNetV3 успешно создана")
                return model
                
            except Exception as e:
                print(f"[ERROR] Ошибка при создании MobileNetV3: {str(e)}")
                print(f"[ERROR] Stack trace: {traceback.format_exc()}")
                raise
                
        return _create_model_operation()
        
    except Exception as e:
        print(f"[ERROR] Критическая ошибка при инициализации MobileNetV3: {str(e)}")
        print(f"[ERROR] Stack trace: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    try:
        # Инициализация компонентов
        model = create_model(
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