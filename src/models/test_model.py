import os
import numpy as np
from tensorflow.keras.models import load_model
from src.data_proc.data_loader import VideoDataLoader
from src.config import Config
from src.models.inference_utils import get_element_intervals

def test_model(model_path, test_data_path, batch_size=1):
    # Загрузка модели
    model = load_model(model_path)
    print(f"Модель загружена из {model_path}")

    # Загрузка данных
    loader = VideoDataLoader(test_data_path)
    generator = loader.load_data(
        Config.SEQUENCE_LENGTH,
        batch_size,
        target_size=Config.INPUT_SIZE,
        one_hot=True,
        infinite_loop=False
    )

    # Тестирование
    video_idx = 0
    for X_batch, _ in generator:
        preds = model.predict(X_batch)
        for i, pred in enumerate(preds):
            result = get_element_intervals(pred)
            print(f"Видео {video_idx}: {result}")
            video_idx += 1

if __name__ == "__main__":
    # Укажите путь к вашей сохранённой модели и тестовой папке
    model_path = "src/models/saved/best_model.h5"
    test_data_path = "data/test"  # или data/valid для валидации
    test_model(model_path, test_data_path) 