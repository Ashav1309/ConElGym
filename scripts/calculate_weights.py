import os
import sys

# Добавляем корневую директорию проекта в PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_proc.calculate_class_weights import calculate_dataset_weights, save_weights_to_config

if __name__ == "__main__":
    try:
        print("[INFO] Начало расчета весов классов...")
        weights = calculate_dataset_weights()
        if weights is not None:
            save_weights_to_config(weights)
            print("\n[SUCCESS] Веса классов успешно рассчитаны и сохранены в конфигурацию")
        else:
            print("\n[ERROR] Не удалось рассчитать веса классов")
    except Exception as e:
        print(f"\n[ERROR] Ошибка при расчете весов классов: {str(e)}") 