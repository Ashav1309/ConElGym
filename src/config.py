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
    MAX_VIDEOS = 3  # Максимум видео для одновременной обработки
    
    # Параметры оптимизации
    LEARNING_RATE = 0.0001
    DROPOUT_RATE = 0.3
    LSTM_UNITS = 256
    
    # Параметры модели v3
    MODEL_PARAMS = {
        'v3': {
            'dropout_rate': 0.3,
            'lstm_units': 256,
            'positive_class_weight': 300.0
        },
        'v4': {
            'dropout_rate': 0.3,
            'expansion_factor': 4,
            'se_ratio': 0.25,
            'positive_class_weight': 300.0
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