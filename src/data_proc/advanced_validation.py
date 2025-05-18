import json
import os
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from src.config import Config

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    stats: Dict

class AnnotationValidator:
    def __init__(self, video_path: str, annotation_path: str):
        self.video_path = video_path
        self.annotation_path = annotation_path
        self.video_name = os.path.basename(video_path)
        self.cap = None
        self.total_frames = 0
        self.annotations = None
        self.frame_labels = None

    def validate(self) -> ValidationResult:
        """
        Выполняет полную валидацию аннотаций видео
        """
        errors = []
        warnings = []
        stats = {
            'total_frames': 0,
            'action_frames': 0,
            'transition_frames': 0,
            'background_frames': 0,
            'overlaps': 0,
            'class_distribution': {}
        }

        try:
            # Проверка существования файлов
            if not os.path.exists(self.video_path):
                errors.append(f"Видео не найдено: {self.video_path}")
                return ValidationResult(False, errors, warnings, stats)
            
            if not os.path.exists(self.annotation_path):
                errors.append(f"Файл аннотаций не найден: {self.annotation_path}")
                return ValidationResult(False, errors, warnings, stats)

            # Загрузка видео
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                errors.append(f"Не удалось открыть видео: {self.video_path}")
                return ValidationResult(False, errors, warnings, stats)
            
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            stats['total_frames'] = self.total_frames

            # Загрузка и валидация JSON
            try:
                with open(self.annotation_path, 'r', encoding='utf-8') as f:
                    self.annotations = json.load(f)
            except json.JSONDecodeError as e:
                errors.append(f"Ошибка в формате JSON: {str(e)}")
                return ValidationResult(False, errors, warnings, stats)

            # Проверка структуры аннотаций
            if not isinstance(self.annotations, dict):
                errors.append("Аннотации должны быть объектом")
                return ValidationResult(False, errors, warnings, stats)

            if 'annotations' not in self.annotations:
                errors.append("Отсутствует поле 'annotations'")
                return ValidationResult(False, errors, warnings, stats)

            if not isinstance(self.annotations['annotations'], list):
                errors.append("Поле 'annotations' должно быть массивом")
                return ValidationResult(False, errors, warnings, stats)

            # Инициализация массива меток
            self.frame_labels = np.zeros((self.total_frames, 3), dtype=np.float32)

            # Проверка каждой аннотации
            for i, ann in enumerate(self.annotations['annotations']):
                # Проверка обязательных полей
                required_fields = ['start_frame', 'end_frame', 'element_type']
                for field in required_fields:
                    if field not in ann:
                        errors.append(f"Аннотация #{i+1}: отсутствует обязательное поле '{field}'")
                        continue

                # Проверка типов данных
                if not isinstance(ann['start_frame'], int) or not isinstance(ann['end_frame'], int):
                    errors.append(f"Аннотация #{i+1}: start_frame и end_frame должны быть целыми числами")
                    continue

                if not isinstance(ann['element_type'], str):
                    errors.append(f"Аннотация #{i+1}: element_type должен быть строкой")
                    continue

                # Проверка диапазона кадров
                if ann['start_frame'] < 0 or ann['end_frame'] >= self.total_frames:
                    errors.append(f"Аннотация #{i+1}: кадры вне диапазона [0, {self.total_frames-1}]")
                    continue

                if ann['start_frame'] > ann['end_frame']:
                    errors.append(f"Аннотация #{i+1}: start_frame > end_frame")
                    continue

                # Отмечаем метки
                start, end = ann['start_frame'], ann['end_frame']
                self.frame_labels[start:end+1, 1] = 1  # Действие
                self.frame_labels[start, 2] = 1  # Начало
                self.frame_labels[end, 2] = 1  # Конец

            # Подсчет статистики
            stats['action_frames'] = np.sum(self.frame_labels[:, 1] == 1)
            stats['transition_frames'] = np.sum(self.frame_labels[:, 2] == 1)
            stats['background_frames'] = self.total_frames - stats['action_frames']

            # Проверка перекрытий
            overlaps = self._check_overlaps()
            if overlaps:
                warnings.append(f"Обнаружены перекрытия в {len(overlaps)} местах")
                stats['overlaps'] = len(overlaps)

            # Проверка дисбаланса классов
            action_ratio = stats['action_frames'] / self.total_frames
            if action_ratio < 0.1:
                warnings.append(f"Низкая доля кадров действия: {action_ratio:.1%}")
            elif action_ratio > 0.9:
                warnings.append(f"Высокая доля кадров действия: {action_ratio:.1%}")

            # Проверка распределения по типам элементов
            element_types = {}
            for ann in self.annotations['annotations']:
                element_type = ann['element_type']
                element_types[element_type] = element_types.get(element_type, 0) + 1
            
            stats['class_distribution'] = element_types

            # Проверка дисбаланса типов элементов
            if len(element_types) > 1:
                max_count = max(element_types.values())
                min_count = min(element_types.values())
                if max_count / min_count > 5:
                    warnings.append("Значительный дисбаланс в распределении типов элементов")

            return ValidationResult(len(errors) == 0, errors, warnings, stats)

        except Exception as e:
            errors.append(f"Непредвиденная ошибка: {str(e)}")
            return ValidationResult(False, errors, warnings, stats)
        finally:
            if self.cap is not None:
                self.cap.release()

    def _check_overlaps(self) -> List[Tuple[int, int]]:
        """
        Проверяет перекрытия в аннотациях
        Returns:
            List[Tuple[int, int]]: Список пар перекрывающихся аннотаций
        """
        overlaps = []
        for i in range(len(self.annotations['annotations'])):
            for j in range(i + 1, len(self.annotations['annotations'])):
                ann1 = self.annotations['annotations'][i]
                ann2 = self.annotations['annotations'][j]
                
                # Проверяем перекрытие
                if (ann1['start_frame'] <= ann2['end_frame'] and 
                    ann2['start_frame'] <= ann1['end_frame']):
                    overlaps.append((i, j))
        
        return overlaps

def validate_dataset() -> Tuple[Dict[str, ValidationResult], Dict[str, int]]:
    """
    Валидирует весь датасет
    Returns:
        Tuple[Dict[str, ValidationResult], Dict[str, int]]: 
            - Словарь результатов валидации для каждого видео
            - Общая статистика по датасету
    """
    results = {}
    dataset_stats = {
        'total_videos': 0,
        'valid_videos': 0,
        'total_errors': 0,
        'total_warnings': 0,
        'total_frames': 0,
        'total_action_frames': 0,
        'total_transition_frames': 0
    }

    # Валидация тренировочных данных
    train_dir = Config.TRAIN_DATA_PATH
    for video_file in os.listdir(train_dir):
        if not video_file.endswith('.mp4'):
            continue

        video_path = os.path.join(train_dir, video_file)
        annotation_path = os.path.join(Config.TRAIN_ANNOTATION_PATH, 
                                     os.path.splitext(video_file)[0] + '.json')

        validator = AnnotationValidator(video_path, annotation_path)
        result = validator.validate()
        results[video_file] = result

        # Обновляем статистику
        dataset_stats['total_videos'] += 1
        if result.is_valid:
            dataset_stats['valid_videos'] += 1
        dataset_stats['total_errors'] += len(result.errors)
        dataset_stats['total_warnings'] += len(result.warnings)
        
        if result.stats:
            dataset_stats['total_frames'] += result.stats['total_frames']
            dataset_stats['total_action_frames'] += result.stats['action_frames']
            dataset_stats['total_transition_frames'] += result.stats['transition_frames']

    # Валидация валидационных данных
    valid_dir = Config.VALID_DATA_PATH
    for video_file in os.listdir(valid_dir):
        if not video_file.endswith('.mp4'):
            continue

        video_path = os.path.join(valid_dir, video_file)
        annotation_path = os.path.join(Config.VALID_ANNOTATION_PATH, 
                                     os.path.splitext(video_file)[0] + '.json')

        validator = AnnotationValidator(video_path, annotation_path)
        result = validator.validate()
        results[video_file] = result

        # Обновляем статистику
        dataset_stats['total_videos'] += 1
        if result.is_valid:
            dataset_stats['valid_videos'] += 1
        dataset_stats['total_errors'] += len(result.errors)
        dataset_stats['total_warnings'] += len(result.warnings)
        
        if result.stats:
            dataset_stats['total_frames'] += result.stats['total_frames']
            dataset_stats['total_action_frames'] += result.stats['action_frames']
            dataset_stats['total_transition_frames'] += result.stats['transition_frames']

    return results, dataset_stats 