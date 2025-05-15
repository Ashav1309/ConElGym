import os
import sys
import optuna.visualization
import plotly.io as pio
import matplotlib.pyplot as plt
import tensorflow as tf

class Config:
    # Базовые пути
    DATA_DIR = 'data'
    MODEL_SAVE_PATH = 'models'
    LOG_DIR = 'logs'
    CONFIG_PATH = 'config_weights.json'  # Путь к конфигурационному файлу
    
    # Пути к данным
    TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'train')
    VALID_DATA_PATH = os.path.join(DATA_DIR, 'valid')
    TEST_DATA_PATH = os.path.join(DATA_DIR, 'test')
    
    # Параметры модели
    MODEL_TYPE = 'v3'  # 'v3' или 'v4'
    NUM_CLASSES = 2  # Фон и элемент
    INPUT_SIZE = (224, 224)  # Размер входного изображения
    SEQUENCE_LENGTH = 12  # Уменьшаем с 16 до 12 кадров для экономии памяти
    INPUT_SHAPE = (SEQUENCE_LENGTH, *INPUT_SIZE, 3)  # Полная форма входных данных
    BATCH_SIZE = 4  # Увеличиваем с 2 до 4
    EPOCHS = 20
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 20
    MAX_SEQUENCES_PER_VIDEO = 200
    MAX_VIDEOS = 3
    
    # Параметры оптимизации
    LEARNING_RATE = 0.00005  # Уменьшаем с 0.0001 до 0.00005
    DROPOUT_RATE = 0.3
    LSTM_UNITS = 128  # Уменьшаем с 256 до 128
    
    # Параметры модели
    MODEL_PARAMS = {
        'v3': {
            'dropout_rate': 0.3,
            'lstm_units': 256,  # Увеличиваем с 128 до 256
            'positive_class_weight': None,  # Будет рассчитано автоматически на основе данных
            'base_input_shape': INPUT_SIZE + (3,)  # Форма для базовой модели (height, width, channels)
        },
        'v4': {
            'dropout_rate': 0.3,
            'expansion_factor': 4,
            'se_ratio': 0.25,
            'positive_class_weight': None,  # Будет рассчитано автоматически на основе данных
            'base_input_shape': INPUT_SIZE + (3,)  # Форма для базовой модели (height, width, channels)
        }
    }
    
    # Параметры оптимизации памяти
    MEMORY_OPTIMIZATION = {
        'use_mixed_precision': True,
        'clear_memory_after_epoch': True,
        'use_gradient_checkpointing': True,
        'use_xla': True,
        'gradient_accumulation_steps': 2,  # Уменьшаем с 4 до 2
        'cache_dataset': False  # Отключаем кэширование для экономии памяти
    }
    
    # Параметры устройства
    DEVICE_CONFIG = {
        'use_gpu': True,
        'gpu_memory_fraction': 0.8,
        'gpu_memory_limit': 1024 * 8,  # 8GB
        'allow_gpu_memory_growth': True,
        'cpu_threads': 4
    }
    
    # Пути к аннотациям
    TRAIN_ANNOTATION_PATH = os.path.join(DATA_DIR, 'train', 'annotations')
    VALID_ANNOTATION_PATH = os.path.join(DATA_DIR, 'valid', 'annotations')
    
    # Аугментация
    AUGMENTATION = {
        'enabled': True,
        'probability': 0.5,
        'brightness_range': (0.8, 1.2),
        'contrast_range': (0.8, 1.2),
        'rotation_range': (-10, 10),
        'flip_probability': 0.5
    }
    
    # Настройки предотвращения переобучения
    OVERFITTING_PREVENTION = {
        'early_stopping_patience': 5,
        'reduce_lr_factor': 0.1,  # Уменьшаем с 0.2 до 0.1
        'reduce_lr_patience': 3,
        'min_lr': 1e-7,  # Уменьшаем с 1e-6 до 1e-7
        'max_overfitting_threshold': 0.1
    }
    
    # Настройки подбора гиперпараметров
    HYPERPARAM_TUNING = {
        'n_trials': 20,
        'timeout': 3600,
        'n_jobs': 1
    }
    
    # Параметры градиентной аккумуляции
    GRADIENT_ACCUMULATION = {
        'enabled': False,
        'steps': 1
    }
    
    # Параметры focal loss
    FOCAL_LOSS = {
        'gamma': 2.0,
        'alpha': 0.25,
        'use_class_weights': True,  # Добавляем использование весов классов
        'class_weights': {0: 1.0, 1: 50.0}  # Добавляем веса классов
    }
    
    # Параметры адаптивного порога
    ADAPTIVE_THRESHOLD = {
        'enabled': True,
        'threshold_range': (0.1, 1.0),
        'threshold_step': 0.05,
        'use_validation': True,  # Добавляем использование валидационного набора
        'update_frequency': 1  # Обновляем порог каждую эпоху
    }
    
    # Параметры метрик
    METRICS = {
        'use_auc': True,  # Добавляем AUC
        'use_f1': True,  # Используем F1-score
        'use_precision_recall': True,  # Используем Precision и Recall
        'threshold': 0.5  # Порог для бинарной классификации
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

def plot_tuning_results(study):
    """
    Визуализация результатов подбора гиперпараметров и сохранение графиков в PNG
    """
    try:
        tuning_dir = os.path.join(Config.MODEL_SAVE_PATH, 'tuning')
        os.makedirs(tuning_dir, exist_ok=True)

        # 1. История оптимизации
        fig = optuna.visualization.plot_optimization_history(study)
        pio.write_image(fig, os.path.join(tuning_dir, 'optimization_history.png'))

        # 2. Важность гиперпараметров
        fig = optuna.visualization.plot_param_importances(study)
        pio.write_image(fig, os.path.join(tuning_dir, 'param_importances.png'))

        # 3. Параллельные координаты
        fig = optuna.visualization.plot_parallel_coordinate(study)
        pio.write_image(fig, os.path.join(tuning_dir, 'parallel_coordinate.png'))

        # 4. Slice plot
        fig = optuna.visualization.plot_slice(study)
        pio.write_image(fig, os.path.join(tuning_dir, 'slice_plot.png'))

        # 5. Contour plot
        fig = optuna.visualization.plot_contour(study)
        pio.write_image(fig, os.path.join(tuning_dir, 'contour_plot.png'))

        print("[DEBUG] Визуализации Optuna успешно сохранены в PNG.")

    except Exception as e:
        print(f"[ERROR] Не удалось построить или сохранить графики Optuna: {str(e)}")
        import traceback
        traceback.print_exc()

def plot_training_results(history, save_path):
    """
    Визуализация результатов обучения и сохранение графиков в PNG
    """
    try:
        print("\n[DEBUG] Визуализация результатов обучения...")

        if not isinstance(history, tf.keras.callbacks.History):
            raise ValueError("history должен быть экземпляром tf.keras.callbacks.History")

        if not os.path.exists(save_path):
            raise ValueError(f"Директория не существует: {save_path}")

        # Создаем директорию для графиков
        plot_path = os.path.join(save_path, 'plots')
        os.makedirs(plot_path, exist_ok=True)

        # 1. Графики потерь и точности
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(plot_path, 'training_metrics.png'))
        plt.close()

        # 2. График F1-score
        if 'f1_score_element' in history.history and 'val_f1_score_element' in history.history:
            plt.figure(figsize=(6, 4))
            plt.plot(history.history['f1_score_element'], label='Training F1-score')
            plt.plot(history.history['val_f1_score_element'], label='Validation F1-score')
            plt.title('F1-score')
            plt.xlabel('Epoch')
            plt.ylabel('F1-score')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_path, 'f1_score.png'))
            plt.close()

        # 3. Precision и Recall (если есть)
        if 'precision_element' in history.history and 'val_precision_element' in history.history:
            plt.figure(figsize=(6, 4))
            plt.plot(history.history['precision_element'], label='Training Precision')
            plt.plot(history.history['val_precision_element'], label='Validation Precision')
            plt.title('Precision')
            plt.xlabel('Epoch')
            plt.ylabel('Precision')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_path, 'precision.png'))
            plt.close()

        if 'recall_element' in history.history and 'val_recall_element' in history.history:
            plt.figure(figsize=(6, 4))
            plt.plot(history.history['recall_element'], label='Training Recall')
            plt.plot(history.history['val_recall_element'], label='Validation Recall')
            plt.title('Recall')
            plt.xlabel('Epoch')
            plt.ylabel('Recall')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_path, 'recall.png'))
            plt.close()

        print("[DEBUG] Результаты обучения успешно визуализированы и сохранены в PNG.")

    except Exception as e:
        print(f"[ERROR] Ошибка при визуализации результатов: {str(e)}")
        import traceback
        traceback.print_exc()
        raise 