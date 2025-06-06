# ConElGym

## Описание проекта
ConElGym - это система для анализа и аннотирования видео с упражнениями с использованием методов компьютерного зрения и глубокого обучения. Проект предназначен для автоматического определения времени начала и конца выполнения контрольных элементов упражнений в видеопоследовательностях.

## Основные компоненты
- `src/models/model.py` - архитектура нейронной сети для анализа видео
- `src/data_proc/data_loader.py` - загрузка и предобработка данных
- `src/models/train.py` - процесс обучения модели
- `src/data_proc/annotation.py` - инструменты для аннотирования видео
- `src/data_proc/annotate_video.py` - скрипт для автоматического аннотирования видео
- `src/models/hyperparameter_tuning.py` - оптимизация гиперпараметров модели
- `src/config.py` - конфигурационные параметры проекта

## Технический стек
- TensorFlow >= 2.10.0
- Optuna >= 3.0.0
- OpenCV >= 4.5.0
- NumPy >= 1.19.2
- Pandas >= 1.3.0
- Matplotlib >= 3.4.0
- Seaborn >= 0.11.0
- Plotly >= 5.3.0
- Scikit-learn >= 0.24.0
- PyTorch >= 1.9.0 (для очистки памяти GPU)

## Конфигурация
Проект поддерживает следующие настройки:
- Размер входного изображения: 224x224x3 (MobileNetV3)
- Длина последовательности: 16 кадров
- Размер батча: 2 (для оптимизации гиперпараметров)
- Количество эпох: 30 (для оптимизации гиперпараметров)
- Шаги на эпоху: 5
- Шаги валидации: 2

## Структура проекта
```
ConElGym/
├── data/              # Директория с данными
│   ├── train/         # Обучающие данные
│   └── valid/         # Валидационные данные
├── src/               # Исходный код
│   ├── models/        # Модели и обучение
│   │   ├── model.py
│   │   ├── train.py
│   │   └── hyperparameter_tuning.py
│   ├── data_proc/     # Обработка данных
│   │   ├── data_loader.py
│   │   ├── annotation.py
│   │   └── annotate_video.py
│   └── config.py      # Конфигурация
├── venv/              # Виртуальное окружение
└── requirements.txt   # Зависимости
```

## Использование

### Установка зависимостей
```bash
pip install -r requirements.txt
pip install -U kaleido  # Для визуализации результатов оптимизации
```

### Подготовка данных
1. Разместите видео в соответствующих директориях:
   - `data/train/` - для обучающих данных
   - `data/valid/` - для валидационных данных

### Аннотирование видео
```bash
python run_annotate.py
```

### Оптимизация гиперпараметров
```bash
python run_tuning.py
```
Результаты оптимизации будут сохранены в директории `src/models/tuning/`:
- `optuna_results.txt` - результаты всех испытаний
- `optimization_history.png` - график истории оптимизации
- `param_importances.png` - важность параметров

### Обучение модели
```bash
python run_train.py
```

## Примечания
- Для работы с GPU требуется CUDA и cuDNN
- Рекомендуется использовать виртуальное окружение Python
- При недостатке памяти GPU можно уменьшить размер батча или количество шагов