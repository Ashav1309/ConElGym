class Config:
    # Параметры модели
    INPUT_SHAPE = (224, 224, 3)
    NUM_CLASSES = 2  # Начало и конец элемента
    MODEL_TYPE = 'v4'  # Используем только v4 для тестирования
    
    # Параметры обучения
    EPOCHS = 10
    STEPS_PER_EPOCH = 50
    VALIDATION_STEPS = 10
    
    # Параметры данных
    SEQUENCE_LENGTH = 8
    TARGET_SIZE = (224, 224)
    INPUT_SIZE = (224, 224)
    MAX_SEQUENCES_PER_VIDEO = 10
    
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
        'gpu_memory_limit': 4096,
        'cpu_threads': 2,
        'allow_gpu_memory_growth': True,
        'per_process_gpu_memory_fraction': 0.6,
    }
    
    # Настройки оптимизации памяти
    MEMORY_OPTIMIZATION = {
        'clear_memory_after_trial': True,
        'use_mixed_precision': True,
        'cache_dataset': False,
        'prefetch_buffer_size': 1,
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
        'n_trials': 1,  # Только один trial для тестирования
        'timeout': 3600,
        'n_jobs': 1
    }
    
    # Параметры моделей
    MODEL_PARAMS = {
        'v3': {
            'lstm_units': 64,
            'dropout_rate': 0.5,
            'model_type': 'small'
        },
        'v4': {
            'dropout_rate': 0.5,
            'model_type': 'small',
            'expansion_factor': 4,
            'se_ratio': 0.25,
            'initial_filters': 32,
            'blocks': [
                {'filters': 64, 'expansion': 4, 'stride': 2},
                {'filters': 128, 'expansion': 4, 'stride': 2},
                {'filters': 256, 'expansion': 4, 'stride': 2},
                {'filters': 512, 'expansion': 4, 'stride': 2},
            ]
        }
    } 