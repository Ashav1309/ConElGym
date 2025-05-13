import os
import sys

class Config:
    # Параметры модели
    INPUT_SHAPE = (224, 224, 3)
    NUM_CLASSES = 2  # Начало и конец элемента
    MODEL_TYPE = 'v4'  # Используем только v4 для тестирования
    MODEL_SIZE = 'small'  # Размер модели (small, medium, large)
    EXPANSION_FACTOR = 4  # Фактор расширения для UIB блоков
    SE_RATIO = 0.25  # Коэффициент для Squeeze-and-Excitation
    
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
    TRAIN_ANNOTATION_PATH = 'data/train/annotations'
    VALID_ANNOTATION_PATH = 'data/valid/annotations'
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
    
    @classmethod
    def validate(cls):
        """Проверка конфигурации"""
        print("\n[DEBUG] ===== Валидация конфигурации =====")
        
        # Проверка путей
        paths = [
            cls.TRAIN_DATA_PATH,
            cls.VALID_DATA_PATH,
            cls.TRAIN_ANNOTATION_PATH,
            cls.VALID_ANNOTATION_PATH,
            cls.MODEL_SAVE_PATH
        ]
        
        for path in paths:
            if not os.path.exists(path):
                print(f"[ERROR] Директория не найдена: {path}")
                os.makedirs(path, exist_ok=True)
                print(f"[DEBUG] Создана директория: {path}")
        
        # Проверка параметров модели
        if cls.MODEL_TYPE not in ['v3', 'v4']:
            raise ValueError(f"Неверный тип модели: {cls.MODEL_TYPE}. Допустимые значения: v3, v4")
            
        if cls.MODEL_SIZE not in ['small', 'medium', 'large']:
            raise ValueError(f"Неверный размер модели: {cls.MODEL_SIZE}. Допустимые значения: small, medium, large")
            
        if cls.EXPANSION_FACTOR <= 0:
            raise ValueError(f"Фактор расширения должен быть положительным: {cls.EXPANSION_FACTOR}")
            
        if not 0 < cls.SE_RATIO <= 1:
            raise ValueError(f"Коэффициент SE должен быть в диапазоне (0, 1]: {cls.SE_RATIO}")
            
        # Проверка параметров обучения
        if cls.EPOCHS <= 0:
            raise ValueError(f"Количество эпох должно быть положительным: {cls.EPOCHS}")
            
        if cls.STEPS_PER_EPOCH <= 0:
            raise ValueError(f"Количество шагов на эпоху должно быть положительным: {cls.STEPS_PER_EPOCH}")
            
        if cls.VALIDATION_STEPS <= 0:
            raise ValueError(f"Количество шагов валидации должно быть положительным: {cls.VALIDATION_STEPS}")
            
        # Проверка параметров данных
        if cls.SEQUENCE_LENGTH <= 0:
            raise ValueError(f"Длина последовательности должна быть положительной: {cls.SEQUENCE_LENGTH}")
            
        if cls.MAX_SEQUENCES_PER_VIDEO <= 0:
            raise ValueError(f"Максимальное количество последовательностей должно быть положительным: {cls.MAX_SEQUENCES_PER_VIDEO}")
            
        # Проверка настроек GPU
        if cls.DEVICE_CONFIG['use_gpu']:
            if cls.DEVICE_CONFIG['gpu_memory_limit'] <= 0:
                raise ValueError(f"Лимит памяти GPU должен быть положительным: {cls.DEVICE_CONFIG['gpu_memory_limit']}")
                
            if not 0 < cls.DEVICE_CONFIG['per_process_gpu_memory_fraction'] <= 1:
                raise ValueError(f"Доля памяти GPU должна быть в диапазоне (0, 1]: {cls.DEVICE_CONFIG['per_process_gpu_memory_fraction']}")
                
        # Проверка настроек CPU
        if cls.DEVICE_CONFIG['cpu_threads'] <= 0:
            raise ValueError(f"Количество потоков CPU должно быть положительным: {cls.DEVICE_CONFIG['cpu_threads']}")
            
        print("[DEBUG] Валидация конфигурации успешно завершена\n")
        
# Валидация конфигурации при импорте
Config.validate() 