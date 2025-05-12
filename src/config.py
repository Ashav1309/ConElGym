class Config:
    # Параметры модели
    INPUT_SHAPE = (224, 224, 3)
    NUM_CLASSES = 2  # Начало и конец элемента
    
    # Параметры обучения
    BATCH_SIZE = 1  # Уменьшаем размер батча
    EPOCHS = 10
    LEARNING_RATE = 0.001
    STEPS_PER_EPOCH = 10
    VALIDATION_STEPS = 5
    
    # Параметры данных
    SEQUENCE_LENGTH = 8
    TARGET_SIZE = (224, 224)
    INPUT_SIZE = (224, 224)
    
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
    
    # Настройки CPU/GPU
    DEVICE_CONFIG = {
        'use_gpu': True,
        'gpu_memory_limit': 4096,  # Уменьшаем лимит памяти GPU
        'cpu_threads': 2,  # Уменьшаем количество потоков CPU
        'allow_gpu_memory_growth': True,
        'per_process_gpu_memory_fraction': 0.6,  # Уменьшаем долю используемой памяти GPU
    }
    
    # Настройки оптимизации памяти
    MEMORY_OPTIMIZATION = {
        'clear_memory_after_trial': True,
        'use_mixed_precision': True,
        'cache_dataset': False,  # Отключаем кэширование датасета
        'prefetch_buffer_size': 1,  # Минимальный размер буфера
        'allow_memory_growth': True,
    }
    
    # Параметры оптимизации гиперпараметров
    HYPERPARAM_TUNING = {
        'n_trials': 10,
        'learning_rate_range': (1e-5, 1e-3),
        'dropout_range': (0.2, 0.6),
        'lstm_units': [16, 32],  # Уменьшаем варианты LSTM units
        'epochs_per_trial': 3,  # Уменьшаем количество эпох
        'steps_per_epoch': 5,  # Уменьшаем количество шагов
        'validation_steps': 3,  # Уменьшаем количество шагов валидации
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