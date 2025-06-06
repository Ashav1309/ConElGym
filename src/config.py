import os
import sys
import optuna.visualization
import plotly.io as pio
import matplotlib.pyplot as plt
import tensorflow as tf
import json
from pathlib import Path
import glob

class Config:
    # Пути
    PROJECT_ROOT = Path(__file__).parent.parent  # Поднимаемся на уровень выше src/
    CONFIG_PATH = os.path.join(PROJECT_ROOT, 'src', 'config_weights.json')
    MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'best_model.h5')
    
    # Базовые пути
    DATA_DIR = os.path.join('data')
    MODEL_SAVE_PATH = os.path.join('models')
    LOG_DIR = os.path.join('logs')
    
    # Пути к данным
    TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'train')
    VALID_DATA_PATH = os.path.join(DATA_DIR, 'valid')
    TEST_DATA_PATH = os.path.join(DATA_DIR, 'test')
    
    # Пути к аннотациям (определяются автоматически в VideoDataLoader на основе пути к данным)
    # TRAIN_ANNOTATION_PATH и VALID_ANNOTATION_PATH используются только для валидации
    TRAIN_ANNOTATION_PATH = os.path.join(TRAIN_DATA_PATH, 'annotations')
    VALID_ANNOTATION_PATH = os.path.join(VALID_DATA_PATH, 'annotations')
    
    # Параметры модели
    MODEL_TYPE = 'v3'
    INPUT_SHAPE = (224, 224, 3)
    BATCH_SIZE = 2
    EPOCHS = 30
    LEARNING_RATE = 1e-4
    EARLY_STOPPING_PATIENCE = 10
    NUM_CLASSES = 2  # Фон и действие
    SEQUENCE_LENGTH = 16
    INPUT_SIZE = (224, 224)
    STEPS_PER_EPOCH = 5
    VALIDATION_STEPS = 2
    MAX_SEQUENCES_PER_VIDEO = 200
    MAX_VIDEOS = 100
    
    # Параметры загрузчика данных
    MAX_STUCK_BATCHES = 10  # Максимальное количество попыток получения батча
    CACHE_CLEANUP_THRESHOLD = 90  # Порог в процентах для очистки кэша видео
    
    # Параметры оптимизации
    DROPOUT_RATE = 0.3
    LSTM_UNITS = 128  # Уменьшаем с 256 до 128
    
    # Параметры модели
    MODEL_PARAMS = {
        'v3': {
            'input_shape': (16, 224, 224, 3),  # (sequence_length, height, width, channels)
            'num_classes': 2,  # background, action
            'dropout_rate': 0.3,
            'lstm_units': 128,
            'rnn_type': 'lstm',
            'temporal_block_type': 'rnn'
        }
    }
    
    # Параметры оптимизации памяти
    MEMORY_OPTIMIZATION = {
        'use_mixed_precision': True,
        'clear_memory_after_epoch': True,
        'use_gradient_checkpointing': True,
        'use_xla': True,
        'gradient_accumulation_steps': 2,  # Уменьшаем с 4 до 2
        'cache_dataset': False,  # Отключаем кэширование для экономии памяти
        'shuffle_buffer_size': 64,  # Размер буфера для перемешивания
        'prefetch_buffer_size': tf.data.AUTOTUNE,  # Размер буфера для предзагрузки
        'num_parallel_calls': tf.data.AUTOTUNE,  # Количество параллельных вызовов
        'cache_size': 100  # Размер кэша для последовательностей
    }
    
    # Параметры устройства
    DEVICE_CONFIG = {
        'use_gpu': True,
        'gpu_memory_fraction': 0.8,
        'gpu_memory_limit': 1024 * 8,  
        'allow_gpu_memory_growth': True,
        'cpu_threads': 4
    }
    
    # Параметры балансировки данных
    DATA_BALANCING = {
        'enabled': True,
        'class_ratio': 0.5,  # Соотношение классов в батче (0.5 = равное количество)
        'oversample_positive': True,  # Увеличение положительных примеров
        'oversample_factor': 2.0,  # Во сколько раз увеличивать положительные примеры
        'use_smote': False,  # Использовать SMOTE для генерации синтетических примеров
        'smote_k_neighbors': 5  # Количество соседей для SMOTE
    }
    
    # Параметры аугментации
    AUGMENTATION = {
        'brightness_range': 0.2,
        'contrast_range': 0.2,
        'rotation_range': 10,
        'noise_std': 0.05,
        'blur_sigma': 1.0,
        'brightness_prob': 0.5,
        'contrast_prob': 0.5,
        'rotation_prob': 0.5,
        'noise_prob': 0.3,
        'blur_prob': 0.2
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
        'n_trials': 10,
        'timeout': 3600,  # 1 час
        'epochs': 30
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
        'beta': 0.999
    }
    
    # Алиас для весов классов
    CLASS_WEIGHTS = None  # Будет загружено из config_weights.json
    
    # Параметры адаптивного порога
    ADAPTIVE_THRESHOLD = {
        'enabled': True,
        'threshold_range': (0.1, 1.0),
        'threshold_step': 0.05,
        'use_validation': True,
        'update_frequency': 1,
        'num_classes': 2  # 2 класса: фон, действие
    }
    
    # Параметры метрик
    METRICS = {
        'use_auc': True,
        'use_f1': True,
        'use_precision_recall': True,
        'threshold': 0.5,
        'num_classes': 2  # 2 класса: фон, действие
    }
    
    # Константы для валидации данных
    MIN_TRAIN_VIDEOS = 10
    MIN_VAL_VIDEOS = 3    # Минимальное количество видео для валидации
    MIN_VIDEO_WIDTH = 320  # Минимальная ширина видео
    MIN_VIDEO_HEIGHT = 240 # Минимальная высота видео
    MIN_FPS = 15          # Минимальный FPS
    MIN_FRAMES_PER_VIDEO = 30  # Минимальное количество кадров в видео
    MIN_POSITIVE_EXAMPLES = 6
    MIN_POSITIVE_RATIO = 0.0005
    MAX_POSITIVE_RATIO = 0.9
    MIN_TRAIN_BATCHES = 5     # Минимальное количество батчей для обучения
    MIN_VAL_BATCHES = 20        # Минимальное количество батчей для валидации
    
    DEBUG_SMALL_DATASET = False  # Включить для тестов на малых датасетах
    AUGMENT_POSITIVE_ONLY = True  # Применять аугментацию только к положительным примерам
    
    # Параметры балансировки
    BALANCING = {
        'enabled': True,
        'update_frequency': 5,
        'f1_threshold': 0.5,
        'weight_increase': 1.1,
        'num_classes': 2  # 2 класса: фон, действие
    }
    
    # Параметры адаптивного обучения
    ADAPTIVE_LEARNING = {
        'enabled': True,
        'patience': 3,
        'lr_reduction': 0.5,
        'weight_increase': 1.1,
        'num_classes': 2  # 2 класса: фон, действие
    }
    
    # Параметры логирования
    TENSORBOARD_DIR = os.path.join(LOG_DIR, 'tensorboard')
    
    @classmethod
    def apply_debug_small_dataset(cls):
        if cls.DEBUG_SMALL_DATASET:
            print("[WARNING] DEBUG_SMALL_DATASET включён: все пороги и размеры снижены для теста!")
            cls.MIN_TRAIN_VIDEOS = 1
            cls.MIN_VAL_VIDEOS = 1
            cls.MIN_TRAIN_BATCHES = 1
            cls.MIN_VAL_BATCHES = 1
            cls.BATCH_SIZE = 1
            cls.SEQUENCE_LENGTH = 4
            cls.AUGMENTATION['enabled'] = True
            cls.AUGMENTATION['probability'] = 0.2
            cls.AUGMENTATION['brightness_range'] = (0.95, 1.05)
            cls.AUGMENTATION['contrast_range'] = (0.95, 1.05)
            cls.AUGMENTATION['rotation_range'] = (-2, 2)
            cls.AUGMENTATION['flip_probability'] = 0.0
            cls.AUGMENT_POSITIVE_ONLY = True
            cls.OVERFITTING_PREVENTION['early_stopping_patience'] = 20
    
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
        if cls.MODEL_TYPE not in ['v3']:
            raise ValueError(f"Неверный тип модели: {cls.MODEL_TYPE}. Допустимое значение: v3")
            
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
        
    @classmethod
    def load_config(cls):
        """Загружает конфигурацию из JSON файла"""
        with open(cls.CONFIG_PATH, 'r') as f:
            config = json.load(f)
        return config
    
    @classmethod
    def save_config(cls, config):
        """Сохраняет конфигурацию в JSON файл"""
        with open(cls.CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=4)

    @classmethod
    def create_directories(cls):
        os.makedirs(cls.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        os.makedirs(cls.TENSORBOARD_DIR, exist_ok=True)
        os.makedirs(os.path.join(cls.MODEL_SAVE_PATH, 'v3'), exist_ok=True)
        os.makedirs(os.path.join(cls.MODEL_SAVE_PATH, 'tuning'), exist_ok=True)

# Валидация и применение debug-режима при импорте
Config.apply_debug_small_dataset()
Config.validate()

def plot_tuning_results(study):
    """
    Визуализация результатов подбора гиперпараметров с использованием Plotly
    """
    try:
        tuning_dir = os.path.join(Config.MODEL_SAVE_PATH, 'tuning')
        os.makedirs(tuning_dir, exist_ok=True)

        # 1. История оптимизации (интерактивный)
        fig = optuna.visualization.plot_optimization_history(study)
        fig.update_layout(
            title="История оптимизации",
            xaxis_title="Номер trial",
            yaxis_title="F1-score",
            template="plotly_white"
        )
        fig.write_html(os.path.join(tuning_dir, 'optimization_history.html'))
        fig.write_image(os.path.join(tuning_dir, 'optimization_history.png'))

        # 2. Важность параметров (интерактивный)
        fig = optuna.visualization.plot_param_importances(study)
        fig.update_layout(
            title="Важность гиперпараметров",
            template="plotly_white"
        )
        fig.write_html(os.path.join(tuning_dir, 'param_importances.html'))
        fig.write_image(os.path.join(tuning_dir, 'param_importances.png'))

        # 3. Параллельные координаты (интерактивный)
        fig = optuna.visualization.plot_parallel_coordinate(study)
        fig.update_layout(
            title="Параллельные координаты",
            template="plotly_white"
        )
        fig.write_html(os.path.join(tuning_dir, 'parallel_coordinate.html'))
        fig.write_image(os.path.join(tuning_dir, 'parallel_coordinate.png'))

        # 4. Slice plot (интерактивный)
        fig = optuna.visualization.plot_slice(study)
        fig.update_layout(
            title="Slice plot",
            template="plotly_white"
        )
        fig.write_html(os.path.join(tuning_dir, 'slice_plot.html'))
        fig.write_image(os.path.join(tuning_dir, 'slice_plot.png'))

        # 5. Contour plot (интерактивный)
        fig = optuna.visualization.plot_contour(study)
        fig.update_layout(
            title="Contour plot",
            template="plotly_white"
        )
        fig.write_html(os.path.join(tuning_dir, 'contour_plot.html'))
        fig.write_image(os.path.join(tuning_dir, 'contour_plot.png'))

        # 6. Дополнительный анализ корреляций
        import pandas as pd
        import seaborn as sns
        
        # Создаем DataFrame из trials
        trials_df = pd.DataFrame([t.params for t in study.trials if t.value is not None])
        trials_df['value'] = [t.value for t in study.trials if t.value is not None]
        
        # Корреляционная матрица
        plt.figure(figsize=(12, 8))
        sns.heatmap(trials_df.corr(), annot=True, cmap='coolwarm', center=0)
        plt.title('Корреляционная матрица параметров')
        plt.tight_layout()
        plt.savefig(os.path.join(tuning_dir, 'correlation_matrix.png'))
        plt.close()

        # 7. Распределение значений параметров
        plt.figure(figsize=(15, 10))
        for i, param in enumerate(trials_df.columns[:-1], 1):
            plt.subplot(3, 3, i)
            sns.histplot(data=trials_df, x=param, bins=20)
            plt.title(f'Распределение {param}')
        plt.tight_layout()
        plt.savefig(os.path.join(tuning_dir, 'parameter_distributions.png'))
        plt.close()

        # 8. Создаем HTML-отчет
        html_report = f"""
        <html>
        <head>
            <title>Отчет по оптимизации гиперпараметров</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .plot-container {{ margin: 20px 0; }}
                h1, h2 {{ color: #333; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Отчет по оптимизации гиперпараметров</h1>
                <h2>Лучшие параметры:</h2>
                <pre>{json.dumps(study.best_params, indent=2)}</pre>
                <h2>Лучшее значение F1-score: {study.best_value:.4f}</h2>
                
                <div class="plot-container">
                    <h2>Интерактивные графики:</h2>
                    <iframe src="optimization_history.html" width="100%" height="600px" frameborder="0"></iframe>
                    <iframe src="param_importances.html" width="100%" height="600px" frameborder="0"></iframe>
                    <iframe src="parallel_coordinate.html" width="100%" height="600px" frameborder="0"></iframe>
                    <iframe src="slice_plot.html" width="100%" height="600px" frameborder="0"></iframe>
                    <iframe src="contour_plot.html" width="100%" height="600px" frameborder="0"></iframe>
                </div>
                
                <div class="plot-container">
                    <h2>Статические графики:</h2>
                    <img src="correlation_matrix.png" width="100%">
                    <img src="parameter_distributions.png" width="100%">
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(os.path.join(tuning_dir, 'tuning_report.html'), 'w') as f:
            f.write(html_report)

        print("[DEBUG] Визуализации Optuna успешно сохранены.")
        print(f"[DEBUG] Отчет доступен в: {os.path.join(tuning_dir, 'tuning_report.html')}")

    except Exception as e:
        print(f"[ERROR] Не удалось построить или сохранить графики Optuna: {str(e)}")
        import traceback
        traceback.print_exc()

def plot_training_history(history):
    """
    Визуализация истории обучения
    """
    plt.figure(figsize=(12, 4))
    
    # График точности
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # График F1-score
    plt.subplot(1, 2, 2)
    if 'f1_action' in history.history and 'val_f1_action' in history.history:
        plt.plot(history.history['f1_action'], label='Training F1-score')
        plt.plot(history.history['val_f1_action'], label='Validation F1-score')
        plt.title('Model F1-score')
        plt.xlabel('Epoch')
        plt.ylabel('F1-score')
        plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_training_results(history, save_path):
    """
    Визуализация результатов обучения
    """
    try:
        print("\n[DEBUG] Визуализация результатов обучения...")
        
        # Валидация входных параметров
        if not isinstance(history, tf.keras.callbacks.History):
            raise ValueError("history должен быть экземпляром tf.keras.callbacks.History")
            
        if not os.path.exists(save_path):
            raise ValueError(f"Директория не существует: {save_path}")
        
        # Создаем директорию для графиков
        plot_path = os.path.join(save_path, 'plots')
        os.makedirs(plot_path, exist_ok=True)
        
        # Графики потерь и точности
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
        
        # График F1-score
        if 'scalar_f1_score' in history.history and 'val_scalar_f1_score' in history.history:
            plt.figure(figsize=(6, 4))
            plt.plot(history.history['scalar_f1_score'], label='Training F1-score')
            plt.plot(history.history['val_scalar_f1_score'], label='Validation F1-score')
            plt.title('F1-score')
            plt.xlabel('Epoch')
            plt.ylabel('F1-score')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_path, 'f1_score.png'))
            plt.close()
        
        print("[DEBUG] Результаты обучения успешно визуализированы")
        
    except Exception as e:
        print(f"[ERROR] Ошибка при визуализации результатов: {str(e)}")
        print("[DEBUG] Stack trace:", flush=True)
        import traceback
        traceback.print_exc()
        raise 
