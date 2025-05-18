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
        self.video_name = os.path.splitext(os.path.basename(video_path))[0]
        self.logger = logging.getLogger(__name__)

    def validate(self) -> ValidationResult:
        """Выполняет все проверки аннотаций"""
        try:
            # Проверяем существование файлов
            if not self._check_files_exist():
                return ValidationResult(False, "Файлы не найдены")

            # Загружаем данные
            video_data = self._load_video()
            annotation_data = self._load_annotations()
            if video_data is None or annotation_data is None:
                return ValidationResult(False, "Ошибка загрузки данных")

            # Проверяем структуру аннотаций
            if not self._validate_annotation_structure(annotation_data):
                return ValidationResult(False, "Неверная структура аннотаций")

            # Проверяем временные метки
            if not self._validate_timestamps(annotation_data, video_data):
                return ValidationResult(False, "Ошибка в временных метках")

            # Проверяем перекрытия
            if not self._check_overlaps(annotation_data):
                return ValidationResult(False, "Обнаружены перекрытия в аннотациях")

            # Проверяем статистику
            stats = self._calculate_statistics(annotation_data, video_data)
            if not self._validate_statistics(stats):
                return ValidationResult(False, "Статистика не соответствует требованиям")

            return ValidationResult(True, "Аннотации валидны", stats)

        except Exception as e:
            self.logger.error(f"Ошибка при валидации {self.video_name}: {str(e)}")
            return ValidationResult(False, f"Ошибка валидации: {str(e)}")

    def _check_files_exist(self) -> bool:
        """Проверяет существование файлов"""
        if not os.path.exists(self.video_path):
            self.logger.error(f"Видео не найдено: {self.video_path}")
            return False
        if not os.path.exists(self.annotation_path):
            self.logger.error(f"Аннотации не найдены: {self.annotation_path}")
            return False
        return True

    def _load_video(self) -> Optional[Dict]:
        """Загружает информацию о видео"""
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.logger.error(f"Не удалось открыть видео: {self.video_path}")
                return None

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps

            cap.release()

            return {
                'fps': fps,
                'frame_count': frame_count,
                'width': width,
                'height': height,
                'duration': duration
            }
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке видео: {str(e)}")
            return None

    def _load_annotations(self) -> Optional[Dict]:
        """Загружает аннотации"""
        try:
            with open(self.annotation_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке аннотаций: {str(e)}")
            return None

    def _validate_annotation_structure(self, data: Dict) -> bool:
        """Проверяет структуру аннотаций"""
        required_fields = ['video_name', 'annotations']
        if not all(field in data for field in required_fields):
            self.logger.error(f"Отсутствуют обязательные поля: {required_fields}")
            return False

        if data['video_name'] != self.video_name:
            self.logger.error(f"Несоответствие имени видео: {data['video_name']} != {self.video_name}")
            return False

        if not isinstance(data['annotations'], list):
            self.logger.error("Аннотации должны быть списком")
            return False

        for ann in data['annotations']:
            if not all(field in ann for field in ['start_frame', 'end_frame']):
                self.logger.error(f"Отсутствуют обязательные поля в аннотации: {ann}")
                return False

        return True

    def _validate_timestamps(self, data: Dict, video_data: Dict) -> bool:
        """Проверяет временные метки"""
        frame_count = video_data['frame_count']
        for ann in data['annotations']:
            if not (0 <= ann['start_frame'] < frame_count and 
                   0 <= ann['end_frame'] < frame_count):
                self.logger.error(f"Кадры вне диапазона: {ann}")
                return False
            if ann['start_frame'] > ann['end_frame']:
                self.logger.error(f"Неверный порядок кадров: {ann}")
                return False
        return True

    def _check_overlaps(self, data: Dict) -> bool:
        """Проверяет перекрытия в аннотациях"""
        frames = set()
        for ann in data['annotations']:
            for frame in range(ann['start_frame'], ann['end_frame'] + 1):
                if frame in frames:
                    self.logger.error(f"Обнаружено перекрытие на кадре {frame}")
                    return False
                frames.add(frame)
        return True

    def _calculate_statistics(self, data: Dict, video_data: Dict) -> Dict:
        """Рассчитывает статистику аннотаций"""
        total_frames = video_data['frame_count']
        frame_labels = np.zeros((total_frames, 2), dtype=np.float32)  # 2 класса: фон, действие
        frame_labels[:, 0] = 1  # По умолчанию все кадры - фон

        for ann in data['annotations']:
            for frame_idx in range(ann['start_frame'], ann['end_frame'] + 1):
                if frame_idx < len(frame_labels):
                    frame_labels[frame_idx, 1] = 1  # [0,1] - действие
                    frame_labels[frame_idx, 0] = 0  # Убираем метку фона

        background_frames = np.sum(frame_labels[:, 0] == 1)
        action_frames = np.sum(frame_labels[:, 1] == 1)

        return {
            'total_frames': total_frames,
            'background_frames': background_frames,
            'action_frames': action_frames,
            'background_ratio': background_frames / total_frames,
            'action_ratio': action_frames / total_frames
        }

    def _validate_statistics(self, stats: Dict) -> bool:
        """Проверяет статистику на соответствие требованиям"""
        if stats['action_frames'] == 0:
            self.logger.error("Отсутствуют кадры действия")
            return False

        if stats['action_ratio'] < 0.01:  # Минимум 1% действия
            self.logger.error(f"Слишком мало кадров действия: {stats['action_ratio']:.2%}")
            return False

        if stats['action_ratio'] > 0.5:  # Максимум 50% действия
            self.logger.error(f"Слишком много кадров действия: {stats['action_ratio']:.2%}")
            return False

        return True

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