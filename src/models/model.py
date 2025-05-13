import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, Dense, Bidirectional, LSTM, 
    GlobalAveragePooling2D, Reshape,
    Multiply, Conv2D, BatchNormalization,
    Activation, Dropout, TimeDistributed,
    GlobalAveragePooling1D, LayerNormalization, Add, DepthwiseConv2D
)
from tensorflow.keras.applications import MobileNetV3Small
from src.utils.network_handler import NetworkErrorHandler, NetworkMonitor
import logging
import gc

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

class UniversalInvertedBottleneck(tf.keras.layers.Layer):
    def __init__(self, filters, expansion_factor=6, kernel_size=3, stride=1, se_ratio=0.25):
        super(UniversalInvertedBottleneck, self).__init__()
        self.filters = filters
        self.expansion_factor = expansion_factor
        self.kernel_size = kernel_size
        self.stride = stride
        self.se_ratio = se_ratio
        
    def build(self, input_shape):
        expanded_filters = int(input_shape[-1] * self.expansion_factor)
        
        # Pointwise expansion
        self.expand_conv = Conv2D(expanded_filters, (1, 1), padding='same')
        self.expand_bn = BatchNormalization()
        self.expand_act = Activation('relu')
        
        # Depthwise convolution
        self.depthwise_conv = DepthwiseConv2D(
            (self.kernel_size, self.kernel_size),
            strides=(self.stride, self.stride),
            padding='same'
        )
        self.depthwise_bn = BatchNormalization()
        self.depthwise_act = Activation('relu')
        
        # Squeeze-and-Excitation
        self.se_reduce = Conv2D(
            max(1, int(expanded_filters * self.se_ratio)),
            (1, 1),
            padding='same'
        )
        self.se_expand = Conv2D(expanded_filters, (1, 1), padding='same')
        
        # Pointwise projection
        self.project_conv = Conv2D(self.filters, (1, 1), padding='same')
        self.project_bn = BatchNormalization()
        
        super(UniversalInvertedBottleneck, self).build(input_shape)
        
    def compute_output_shape(self, input_shape):
        """
        Вычисляет выходную форму слоя
        Args:
            input_shape: Входная форма (batch_size, height, width, channels)
        Returns:
            Выходная форма (batch_size, height/stride, width/stride, filters)
        """
        height = input_shape[1]
        width = input_shape[2]
        
        # Применяем stride к размерам
        if self.stride > 1:
            height = height // self.stride
            width = width // self.stride
            
        return (input_shape[0], height, width, self.filters)
        
    def call(self, inputs):
        x = self.expand_conv(inputs)
        x = self.expand_bn(x)
        x = self.expand_act(x)
        
        x = self.depthwise_conv(x)
        x = self.depthwise_bn(x)
        x = self.depthwise_act(x)
        
        # Squeeze-and-Excitation
        se = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        se = self.se_reduce(se)
        se = tf.nn.relu(se)
        se = self.se_expand(se)
        se = tf.sigmoid(se)
        x = x * se
        
        x = self.project_conv(x)
        x = self.project_bn(x)
        
        if self.stride == 1 and inputs.shape[-1] == self.filters:
            x = x + inputs
            
        return x

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
                history = self.model.fit(
                    self.data_loader.data_generator(),
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
                
            except tf.errors.ResourceExhaustedError as e:
                logger.error(f"Недостаточно памяти GPU: {str(e)}")
                # Попытка освободить память
                tf.keras.backend.clear_session()
                gc.collect()
                raise
                
            except Exception as e:
                logger.error(f"Ошибка при обучении: {str(e)}")
                raise
                
        return self.network_handler.handle_network_operation(_train_operation)

def create_mobilenetv4_model(input_shape, num_classes, dropout_rate=0.5, model_type='small', expansion_factor=4, se_ratio=0.25):
    """
    Создание модели MobileNetV4
    Args:
        input_shape: форма входных данных
        num_classes: количество классов
        dropout_rate: коэффициент dropout
        model_type: тип модели (используется только 'small')
        expansion_factor: коэффициент расширения для UIB блоков
        se_ratio: коэффициент для Squeeze-and-Excitation
    """
    print(f"[DEBUG] Инициализация MobileNetV4: input_shape={input_shape}, num_classes={num_classes}, dropout_rate={dropout_rate}")
    
    # Конфигурация для модели small
    config = {
        'initial_filters': 32,
        'blocks': [
            {'filters': 64, 'expansion': 4, 'stride': 2},
            {'filters': 128, 'expansion': 4, 'stride': 2},
            {'filters': 256, 'expansion': 4, 'stride': 2},
            {'filters': 512, 'expansion': 4, 'stride': 2},
        ]
    }
    
    network_handler = NetworkErrorHandler()
    
    def _create_model_operation():
        try:
            inputs = Input(shape=input_shape)
            
            # Начальный слой
            x = TimeDistributed(Conv2D(config['initial_filters'], (3, 3), strides=(2, 2), padding='same'))(inputs)
            x = TimeDistributed(BatchNormalization())(x)
            x = TimeDistributed(Activation('relu'))(x)
            
            # UIB блоки
            for block_config in config['blocks']:
                x = TimeDistributed(UniversalInvertedBottleneck(
                    filters=block_config['filters'],
                    expansion_factor=block_config['expansion'],
                    stride=block_config['stride']
                ))(x)
                x = TimeDistributed(Dropout(dropout_rate))(x)
            
            # Пространственное внимание
            x = TimeDistributed(SpatialAttention())(x)
            
            # Глобальное среднее объединение
            x = TimeDistributed(GlobalAveragePooling2D())(x)
            
            # Временное внимание
            x = TemporalAttention(units=128)(x)
            
            # Нормализация
            x = LayerNormalization(axis=-1)(x)
            
            # Финальный слой классификации
            outputs = TimeDistributed(Dense(num_classes, activation='softmax', dtype='float32'))(x)
            
            return Model(inputs=inputs, outputs=outputs)
            
        except Exception as e:
            logger.error(f"Ошибка при создании модели: {str(e)}")
            raise
            
    return network_handler.handle_network_operation(_create_model_operation)

def create_model(input_shape, num_classes, dropout_rate=0.5, lstm_units=64, model_type='v3', model_size='small', expansion_factor=4, se_ratio=0.25):
    """
    Создание модели с выбором типа архитектуры
    Args:
        input_shape: форма входных данных
        num_classes: количество классов
        dropout_rate: коэффициент dropout
        lstm_units: количество юнитов в LSTM слоях
        model_type: тип модели ('v3' или 'v4')
        model_size: размер модели ('small', 'medium', 'large')
        expansion_factor: коэффициент расширения для UIB блоков (только для v4)
        se_ratio: коэффициент для Squeeze-and-Excitation (только для v4)
    """
    if model_type == 'v4':
        return create_mobilenetv4_model(
            input_shape=input_shape,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            model_type=model_size,
            expansion_factor=expansion_factor,
            se_ratio=se_ratio
        )
    else:
        return create_mobilenetv3_model(
            input_shape=input_shape,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            lstm_units=lstm_units
        )

def create_mobilenetv3_model(input_shape, num_classes, dropout_rate=0.5, lstm_units=64):
    """
    Создание модели на основе MobileNetV3 (текущая реализация)
    """
    print(f"[DEBUG] Инициализация MobileNetV3: input_shape={input_shape}, num_classes={num_classes}, dropout_rate={dropout_rate}, lstm_units={lstm_units}")
    network_handler = NetworkErrorHandler()
    network_monitor = NetworkMonitor()
    
    def _create_model_operation():
        try:
            # Проверка состояния сети
            network_monitor.check_network_status()
            
            # Входной слой
            inputs = Input(shape=input_shape)
            
            # MobileNetV3Small как backbone
            base_model = MobileNetV3Small(
                include_top=False,
                weights='imagenet',
                input_shape=input_shape[1:],
                include_preprocessing=True
            )
            
            # Проверка доступности GPU
            if not tf.config.list_physical_devices('GPU'):
                logger.warning("GPU не доступен, используется CPU")
                
            base_model.trainable = False
            
            # Применяем MobileNetV3 к каждому кадру в последовательности
            x = TimeDistributed(base_model)(inputs)
            # Пространственное внимание для каждого кадра
            spatial_attention = TimeDistributed(SpatialAttention())(x)
            x = Multiply()([x, spatial_attention])
            # Подготовка для LSTM
            x = TimeDistributed(GlobalAveragePooling2D())(x)
            # BiLSTM слои
            x = Bidirectional(LSTM(lstm_units * 2, return_sequences=True))(x)
            x = Dropout(dropout_rate)(x)
            x = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
            x = TimeDistributed(Dense(lstm_units // 2, activation='relu'))(x)
            x = Dropout(dropout_rate / 2)(x)
            outputs = TimeDistributed(Dense(num_classes, activation='softmax', dtype='float32'))(x)
            
            return Model(inputs=inputs, outputs=outputs)
            
        except tf.errors.ResourceExhaustedError as e:
            logger.error(f"Недостаточно ресурсов GPU: {str(e)}")
            tf.keras.backend.clear_session()
            gc.collect()
            raise
            
        except Exception as e:
            logger.error(f"Ошибка при создании модели: {str(e)}")
            raise
            
    return network_handler.handle_network_operation(_create_model_operation)

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