class Config:
    # Параметры модели
    INPUT_SHAPE = (224, 224, 3)
    NUM_CLASSES = 2  # Начало и конец элемента
    
    # Параметры обучения
    BATCH_SIZE = 1  # Размер батча для оптимизации гиперпараметров
    EPOCHS = 30  # Количество эпох для оптимизации
    LEARNING_RATE = 0.001
    STEPS_PER_EPOCH = 5  # Шаги на эпоху
    VALIDATION_STEPS = 2  # Шаги валидации
    
    # Параметры данных
    SEQUENCE_LENGTH = 16
    TARGET_SIZE = (224, 224)
    INPUT_SIZE = (224, 224)  # Размер входного изображения для MobileNetV3
    
    # Пути
    TRAIN_DATA_PATH = 'data/train'
    VALID_DATA_PATH = 'data/valid'
    MODEL_SAVE_PATH = 'src/models/saved'
    
    # Аугментация
    AUGMENTATION = True
    ROTATION_RANGE = 15
    WIDTH_SHIFT_RANGE = 0.1
    HEIGHT_SHIFT_RANGE = 0.1
    HORIZONTAL_FLIP = True
    
    # Параметры оптимизации гиперпараметров
    HYPERPARAM_TUNING = {
        'n_trials': 10,  # Количество испытаний
        'learning_rate_range': (1e-4, 1e-3),  # Диапазон learning rate
        'dropout_range': (0.3, 0.7),  # Диапазон dropout
        'lstm_units': [16, 32],  # Возможные значения LSTM units
    }
    
    # Настройки CPU/GPU
    DEVICE_CONFIG = {
        'use_gpu': False,  # Использовать ли GPU
        'cpu_threads': 4,  # Количество потоков CPU
        'gpu_memory_limit': 4096,  # Лимит памяти GPU в МБ
    }
    
    # Настройки оптимизации памяти
    MEMORY_OPTIMIZATION = {
        'clear_memory_after_trial': True,  # Очищать память после каждого испытания
        'use_mixed_precision': False,  # Использовать mixed precision
        'cache_dataset': True,  # Кэшировать датасет
    } 