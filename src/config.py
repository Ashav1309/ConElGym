class Config:
    # Параметры модели
    INPUT_SHAPE = (224, 224, 3)
    NUM_CLASSES = 2  # Начало и конец элемента
    
    # Параметры обучения
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    
    # Параметры данных
    SEQUENCE_LENGTH = 16
    TARGET_SIZE = (224, 224)
    
    # Пути
    TRAIN_DATA_PATH = "data_proc/train"
    VALID_DATA_PATH = "data_proc/valid"
    MODEL_SAVE_PATH = "models"
    
    # Аугментация
    AUGMENTATION = True
    ROTATION_RANGE = 15
    WIDTH_SHIFT_RANGE = 0.1
    HEIGHT_SHIFT_RANGE = 0.1
    HORIZONTAL_FLIP = True 