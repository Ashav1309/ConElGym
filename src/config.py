class Config:
    # Параметры модели
    INPUT_SHAPE = (224, 224, 3)
    NUM_CLASSES = 2  # Начало и конец элемента
    
    # Параметры обучения
    EPOCHS = 10
    STEPS_PER_EPOCH = 100  # Увеличиваем количество шагов в эпохе
    VALIDATION_STEPS = 20  # Увеличиваем количество шагов валидации
    
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
    
    # Настройки предотвращения переобучения
    OVERFITTING_PREVENTION = {
        'early_stopping_patience': 5,
        'reduce_lr_factor': 0.2,
        'reduce_lr_patience': 3,
        'min_lr': 1e-6,
        'max_overfitting_threshold': 0.1
    }
    
    # Настройки подбора гиперпараметров
    HYPERPARAM_TUNING = {
        'n_trials': 20,
        'timeout': 3600,  # 1 час
        'n_jobs': 1
    } 