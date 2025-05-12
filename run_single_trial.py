import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['TF_NUM_INTRAOP_THREADS'] = '2'
os.environ['TF_NUM_INTEROP_THREADS'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print('[DEBUG] Ограничены потоки TensorFlow/OMP', flush=True)
import time
print('[DEBUG] run_single_trial.py: старт', flush=True)
import argparse
import json
import os
print('[DEBUG] Импорт tensorflow...', flush=True)
import tensorflow as tf
print('[DEBUG] TensorFlow импортирован', flush=True)
from src.models.model import create_model
from src.data_proc.data_loader import VideoDataLoader
from src.config import Config
from tensorflow.keras.metrics import Precision, Recall
import gc

def clear_memory():
    tf.keras.backend.clear_session()
    gc.collect()

def f1_score_element(y_true, y_pred):
    y_true = tf.argmax(y_true, axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1)
    true_positives = tf.reduce_sum(tf.cast((y_true == 1) & (y_pred == 1), tf.float32))
    predicted_positives = tf.reduce_sum(tf.cast(y_pred == 1, tf.float32))
    possible_positives = tf.reduce_sum(tf.cast(y_true == 1, tf.float32))
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
    return 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())

def main():
    print('[DEBUG] Парсинг аргументов...', flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, required=True)
    parser.add_argument('--dropout_rate', type=float, required=True)
    parser.add_argument('--lstm_units', type=int, required=True)
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE)
    parser.add_argument('--sequence_length', type=int, default=Config.SEQUENCE_LENGTH)
    parser.add_argument('--max_sequences_per_video', type=int, default=10)
    args = parser.parse_args()
    print(f'[DEBUG] Аргументы: {args}', flush=True)

    print('[DEBUG] clear_memory()', flush=True)
    clear_memory()

    print('[DEBUG] Создание модели...', flush=True)
    input_shape = (args.sequence_length, *Config.INPUT_SIZE, 3)
    model = create_model(
        input_shape=input_shape,
        num_classes=Config.NUM_CLASSES,
        dropout_rate=args.dropout_rate,
        lstm_units=args.lstm_units
    )
    print('[DEBUG] Модель создана', flush=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            Precision(class_id=1, name='precision_element'),
            Recall(class_id=1, name='recall_element'),
            f1_score_element
        ]
    )
    print('[DEBUG] Модель скомпилирована', flush=True)

    print('[DEBUG] Загрузка данных...', flush=True)
    train_loader = VideoDataLoader(Config.TRAIN_DATA_PATH)
    val_loader = VideoDataLoader(Config.VALID_DATA_PATH)
    print('[DEBUG] VideoDataLoader создан', flush=True)
    train_dataset = train_loader.data_generator(
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        target_size=Config.INPUT_SIZE,
        one_hot=True,
        infinite_loop=True,
        max_sequences_per_video=args.max_sequences_per_video
    )
    val_dataset = val_loader.data_generator(
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        target_size=Config.INPUT_SIZE,
        one_hot=True,
        infinite_loop=True,
        max_sequences_per_video=args.max_sequences_per_video
    )
    print('[DEBUG] Генераторы данных созданы', flush=True)
    train_dataset = tf.data.Dataset.from_generator(
        lambda: train_dataset,
        output_signature=(
            tf.TensorSpec(shape=(args.sequence_length, *Config.INPUT_SIZE, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(args.sequence_length, 2), dtype=tf.float32)
        )
    ).batch(args.batch_size).prefetch(1)
    val_dataset = tf.data.Dataset.from_generator(
        lambda: val_dataset,
        output_signature=(
            tf.TensorSpec(shape=(args.sequence_length, *Config.INPUT_SIZE, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(args.sequence_length, 2), dtype=tf.float32)
        )
    ).batch(args.batch_size).prefetch(1)
    print('[DEBUG] tf.data.Dataset создан', flush=True)

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=2,
            min_lr=1e-6
        )
    ]

    print('[DEBUG] Старт fit()', flush=True)
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=Config.EPOCHS,
        steps_per_epoch=Config.STEPS_PER_EPOCH,
        validation_steps=Config.VALIDATION_STEPS,
        callbacks=callbacks,
        verbose=0
    )
    print('[DEBUG] fit() завершён', flush=True)
    best_val_accuracy = max(history.history['val_accuracy'])
    print(f'[DEBUG] best_val_accuracy={best_val_accuracy}', flush=True)
    print(best_val_accuracy)
    print('[DEBUG] clear_memory() после fit', flush=True)
    clear_memory()
    print('[DEBUG] Конец run_single_trial.py', flush=True)

if __name__ == '__main__':
    main() 