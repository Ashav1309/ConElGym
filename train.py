import tensorflow as tf
from model import create_model
from data_loader import VideoDataLoader
from config import Config
import os
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)

def train():
    # Создание директорий
    os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
    
    # Создание модели
    model = create_model(
        input_shape=Config.INPUT_SHAPE,
        num_classes=Config.NUM_CLASSES
    )
    
    # Компиляция модели
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(Config.MODEL_SAVE_PATH, 'best_model.h5'),
            monitor='val_loss',
            save_best_only=True
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6
        )
    ]
    
    # Загрузка данных
    train_loader = VideoDataLoader(Config.TRAIN_DATA_PATH)
    val_loader = VideoDataLoader(Config.VALID_DATA_PATH)
    
    X_train, y_train = train_loader.load_data(Config.SEQUENCE_LENGTH)
    X_val, y_val = val_loader.load_data(Config.SEQUENCE_LENGTH)
    
    # Обучение
    history = model.fit(
        X_train, y_train,
        batch_size=Config.BATCH_SIZE,
        epochs=Config.EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )
    
    return history

if __name__ == "__main__":
    train() 