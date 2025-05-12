class Config:
    # Параметры модели
    INPUT_SHAPE = (224, 224, 3)
    NUM_CLASSES = 2  # Начало и конец элемента
    
    # Параметры обучения
    BATCH_SIZE = 2
    EPOCHS = 10  # Уменьшаем количество эпох для подбора гиперпараметров
    LEARNING_RATE = 0.001
    STEPS_PER_EPOCH = 10  # Уменьшаем количество шагов
    VALIDATION_STEPS = 5  # Уменьшаем количество шагов валидации
    
    # Параметры данных
    SEQUENCE_LENGTH = 8
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
        'n_trials': 10,  # Увеличиваем количество испытаний
        'learning_rate_range': (1e-5, 1e-3),  # Расширяем диапазон learning rate
        'dropout_range': (0.2, 0.6),  # Расширяем диапазон dropout
        'lstm_units': [16, 32, 64],  # Увеличиваем варианты LSTM units
        'epochs_per_trial': 5,  # Количество эпох для каждого испытания
        'steps_per_epoch': 10,  # Количество шагов на эпоху
        'validation_steps': 5,  # Количество шагов валидации
    }
    
    # Настройки CPU/GPU
    DEVICE_CONFIG = {
        'use_gpu': True,  # Включаем GPU
        'gpu_memory_limit': 8192,  # Увеличиваем лимит памяти GPU до 8 ГБ
        'cpu_threads': 4,  # Количество потоков CPU
        'allow_gpu_memory_growth': True,  # Разрешаем динамический рост памяти GPU
        'per_process_gpu_memory_fraction': 0.8,  # Используем 80% доступной памяти GPU
    }
    
    # Настройки оптимизации памяти
    MEMORY_OPTIMIZATION = {
        'clear_memory_after_trial': True,  # Очищать память после каждого испытания
        'use_mixed_precision': True,  # Включаем mixed precision для GPU
        'cache_dataset': False,  # Отключаем кэширование датасета
        'prefetch_buffer_size': 2,  # Увеличиваем размер буфера предзагрузки
        'allow_memory_growth': True,  # Разрешаем динамический рост памяти
    }
    
    # Параметры для предотвращения переобучения
    OVERFITTING_PREVENTION = {
        'early_stopping_patience': 10,  # Количество эпох без улучшения для early stopping
        'reduce_lr_patience': 5,  # Количество эпох без улучшения для уменьшения learning rate
        'reduce_lr_factor': 0.2,  # Фактор уменьшения learning rate
        'min_lr': 1e-6,  # Минимальный learning rate
        'validation_split': 0.2,  # Доля данных для валидации
        'max_overfitting_threshold': 0.1,  # Максимально допустимая разница между train и val accuracy
    } 