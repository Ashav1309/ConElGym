from src.models.train import train
import tensorflow as tf

if __name__ == "__main__":
    train()
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("GPU Device: ", tf.test.gpu_device_name()) 