import cv2
import numpy as np
import pickle
import argparse
from src.config import Config
from src.models.inference_utils import get_element_intervals
import os

def load_model_from_pickle(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)['model']
    return model

def preprocess_frame(frame, target_size):
    frame = cv2.resize(frame, target_size)
    frame = frame.astype(np.float32) / 255.0
    return frame

def realtime_action_detection(model, video_source=0, sequence_length=8, fps=25, target_size=(96, 96)):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[ERROR] Не удалось открыть источник видео: {video_source}")
        return
    frame_buffer = []
    action_active = False
    intervals = []
    start_time = None
    print("[INFO] Для выхода нажмите 'q'")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = preprocess_frame(frame, target_size)
        frame_buffer.append(frame)
        if len(frame_buffer) == sequence_length:
            X = np.array(frame_buffer)[None, ...]  # (1, seq_len, H, W, 3)
            pred = model.predict(X)[0]  # (seq_len, 2)
            pred_class = np.argmax(pred, axis=-1)
            last_pred = pred_class[-1]
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # время в секундах
            if last_pred == 1 and not action_active:
                start_time = current_time
                action_active = True
                print(f"Начало действия: {start_time:.2f} сек")
            elif last_pred == 0 and action_active:
                end_time = current_time
                action_active = False
                intervals.append((start_time, end_time))
                print(f"Конец действия: {end_time:.2f} сек")
            frame_buffer.pop(0)
        cv2.imshow('Realtime Action Detection', (frame * 255).astype(np.uint8))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    print("Интервалы действия (сек):", intervals)
    return intervals

def main():
    parser = argparse.ArgumentParser(description="Реальное время: определение действия моделью")
    parser.add_argument('--model_path', type=str, required=True, help='Путь к .pkl модели')
    parser.add_argument('--video_source', type=str, default='0', help='Источник видео: 0 (камера) или путь к файлу')
    parser.add_argument('--sequence_length', type=int, default=8, help='Длина последовательности')
    parser.add_argument('--fps', type=int, default=25, help='FPS видео')
    parser.add_argument('--width', type=int, default=96, help='Ширина кадра')
    parser.add_argument('--height', type=int, default=96, help='Высота кадра')
    args = parser.parse_args()
    video_source = int(args.video_source) if args.video_source.isdigit() else args.video_source
    model = load_model_from_pickle(args.model_path)
    realtime_action_detection(
        model,
        video_source=video_source,
        sequence_length=args.sequence_length,
        fps=args.fps,
        target_size=(args.width, args.height)
    )

if __name__ == "__main__":
    main() 