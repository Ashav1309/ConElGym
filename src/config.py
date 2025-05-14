import os
import sys

class Config:
    # Базовые пути
    DATA_DIR = 'data'
    MODEL_SAVE_PATH = 'models'
    LOG_DIR = 'logs'
    
    # Пути к данным
    TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'train')
    VALID_DATA_PATH = os.path.join(DATA_DIR, 'valid')
    TEST_DATA_PATH = os.path.join(DATA_DIR, 'test')
    
    # Параметры модели
    MODEL_TYPE = 'v3'  # 'v3' или 'v4'
    NUM_CLASSES = 2  # Фон и элемент
    INPUT_SIZE = (224, 224)  # Размер входного изображения
    SEQUENCE_LENGTH = 8  # Длина последовательности кадров
    BATCH_SIZE = 4  # Уменьшаем размер батча для экономии памяти
    EPOCHS = 20
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 20
    MAX_SEQUENCES_PER_VIDEO = 100  # Ограничиваем количество последовательностей
    
    # Параметры оптимизации
    LEARNING_RATE = 0.00005
    DROPOUT_RATE = 0.3
    LSTM_UNITS = 64
    
    # Параметры модели v3
    MODEL_PARAMS = {
        'v3': {
            'dropout_rate': 0.3,
            'lstm_units': 64,
            'positive_class_weight': 200.0
        },
        'v4': {
            'dropout_rate': 0.3,
            'expansion_factor': 4,
            'se_ratio': 0.25,
            'positive_class_weight': 200.0
        }
    }
    
    # Параметры модели v4
    MODEL_PARAMS_V4 = {
        'dropout_rate': 0.3,
        'expansion_factor': 4,
        'se_ratio': 0.25
    }
    
    # Параметры оптимизации памяти
    MEMORY_OPTIMIZATION = {
        'use_mixed_precision': True,
        'clear_memory_after_epoch': True,
        'gradient_accumulation_steps': 4,
        'cache_dataset': True
    }
    
    # Параметры устройства
    DEVICE_CONFIG = {
        'use_gpu': True,
        'gpu_memory_fraction': 0.8,
        'gpu_memory_limit': 1024 * 8,  # 8GB
        'allow_gpu_memory_growth': True,
        'cpu_threads': 4
    }
    
    # Параметры модели
    INPUT_SHAPE = (224, 224, 3)
    MODEL_SIZE = 'small'  # Размер модели (small, medium, large)
    EXPANSION_FACTOR = 4  # Фактор расширения для UIB блоков
    SE_RATIO = 0.25  # Коэффициент для Squeeze-and-Excitation
    
    # Пути к аннотациям
    TRAIN_ANNOTATION_PATH = os.path.join(DATA_DIR, 'train', 'annotations')
    VALID_ANNOTATION_PATH = os.path.join(DATA_DIR, 'valid', 'annotations')
    
    # Аугментация
    AUGMENTATION = True
    ROTATION_RANGE = 15
    WIDTH_SHIFT_RANGE = 0.1
    HEIGHT_SHIFT_RANGE = 0.1
    HORIZONTAL_FLIP = True
    
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
        'n_trials': 10,
        'timeout': 3600,
        'n_jobs': 1
    }
    
    @classmethod
    def validate(cls):
        """Проверка конфигурации"""
        print("\n[DEBUG] ===== Валидация конфигурации =====")
        
        # Проверка путей
        paths = [
            cls.DATA_DIR,
            cls.TRAIN_DATA_PATH,
            cls.VALID_DATA_PATH,
            cls.TEST_DATA_PATH,
            cls.TRAIN_ANNOTATION_PATH,
            cls.VALID_ANNOTATION_PATH,
            cls.MODEL_SAVE_PATH,
            cls.LOG_DIR
        ]
        
        for path in paths:
            if not os.path.exists(path):
                print(f"[DEBUG] Создание директории: {path}")
                os.makedirs(path, exist_ok=True)
        
        # Проверка параметров модели
        if cls.MODEL_TYPE not in ['v3', 'v4']:
            raise ValueError(f"Неверный тип модели: {cls.MODEL_TYPE}. Допустимые значения: v3, v4")
            
        if cls.NUM_CLASSES <= 0:
            raise ValueError(f"Количество классов должно быть положительным: {cls.NUM_CLASSES}")
            
        if cls.SEQUENCE_LENGTH <= 0:
            raise ValueError(f"Длина последовательности должна быть положительной: {cls.SEQUENCE_LENGTH}")
            
        if cls.BATCH_SIZE <= 0:
            raise ValueError(f"Размер батча должен быть положительным: {cls.BATCH_SIZE}")
            
        if cls.EPOCHS <= 0:
            raise ValueError(f"Количество эпох должно быть положительным: {cls.EPOCHS}")
            
        if cls.STEPS_PER_EPOCH <= 0:
            raise ValueError(f"Количество шагов на эпоху должно быть положительным: {cls.STEPS_PER_EPOCH}")
            
        if cls.VALIDATION_STEPS <= 0:
            raise ValueError(f"Количество шагов валидации должно быть положительным: {cls.VALIDATION_STEPS}")
            
        if cls.MAX_SEQUENCES_PER_VIDEO <= 0:
            raise ValueError(f"Максимальное количество последовательностей должно быть положительным: {cls.MAX_SEQUENCES_PER_VIDEO}")
            
        # Проверка параметров оптимизации
        if cls.LEARNING_RATE <= 0:
            raise ValueError(f"Скорость обучения должна быть положительной: {cls.LEARNING_RATE}")
            
        if not 0 <= cls.DROPOUT_RATE <= 1:
            raise ValueError(f"Коэффициент dropout должен быть в диапазоне [0, 1]: {cls.DROPOUT_RATE}")
            
        if cls.LSTM_UNITS <= 0:
            raise ValueError(f"Количество LSTM юнитов должно быть положительным: {cls.LSTM_UNITS}")
            
        # Проверка настроек GPU
        if cls.DEVICE_CONFIG['use_gpu']:
            if not 0 < cls.DEVICE_CONFIG['gpu_memory_fraction'] <= 1:
                raise ValueError(f"Доля памяти GPU должна быть в диапазоне (0, 1]: {cls.DEVICE_CONFIG['gpu_memory_fraction']}")
                
            if cls.DEVICE_CONFIG['gpu_memory_limit'] <= 0:
                raise ValueError(f"Лимит памяти GPU должен быть положительным: {cls.DEVICE_CONFIG['gpu_memory_limit']}")
                
            if cls.DEVICE_CONFIG['cpu_threads'] <= 0:
                raise ValueError(f"Количество CPU потоков должно быть положительным: {cls.DEVICE_CONFIG['cpu_threads']}")
        
        print("[DEBUG] Валидация конфигурации успешно завершена\n")
        
# Валидация конфигурации при импорте
Config.validate() 