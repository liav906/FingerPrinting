import os
import time
import yaml
import psutil
from datetime import datetime
from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback
from preprocess import load_data, preprocess_data, split_data
from model import build_model
from logger import get_logger
from utils import create_output_dir

def train(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Setup logging
    start_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = create_output_dir(config['paths']['outputs'])
    logger = get_logger(os.path.join(config['paths']['logs'], f"{start_time}.log"))

    # Load and preprocess data
    train_data, val_data, _ = load_data(config)
    train_features, train_labels, scaler = preprocess_data(train_data)
    val_features, val_labels, _ = preprocess_data(val_data, scaler, is_training=False)
    x_train, x_val, y_train, y_val = split_data(train_features, train_labels, config)

    # Build and compile model
    model = build_model(config)
    checkpoint_path = os.path.join(output_dir, 'model_weights_epoch_{epoch:02d}.h5')
    checkpoint = ModelCheckpoint(checkpoint_path, save_weights_only=True, save_best_only=False)

    # Callback to print loss after each epoch and measure CPU/memory usage
    def on_epoch_end(epoch, logs):
        epoch_time = time.time() - start_epoch_time
        cpu_usage = psutil.cpu_percent()
        memory_info = psutil.virtual_memory()
        print(f"Epoch {epoch+1}: loss = {logs['loss']:.4f}, val_loss = {logs['val_loss']:.4f}, time = {epoch_time:.2f}s, CPU = {cpu_usage}%, Memory = {memory_info.percent}%")
        logger.info(f"Epoch {epoch+1}: loss = {logs['loss']:.4f}, val_loss = {logs['val_loss']:.4f}, time = {epoch_time:.2f}s, CPU = {cpu_usage}%, Memory = {memory_info.percent}%")

    print_callback = LambdaCallback(on_epoch_begin=lambda epoch, logs: globals().__setitem__('start_epoch_time', time.time()), 
                                    on_epoch_end=on_epoch_end)

    # Training the model
    logger.info('Training started.')
    start = time.time()
    history = model.fit(x_train, y_train,
                        validation_data=(x_val, y_val),
                        epochs=config['training']['epochs'],
                        batch_size=config['training']['batch_size'],
                        callbacks=[checkpoint, print_callback])
    end = time.time()
    logger.info(f"Training finished in {end - start:.2f} seconds.")

    # Save history and model
    with open(os.path.join(output_dir, 'training_history.yaml'), 'w') as f:
        yaml.dump(history.history, f)
    model.save(os.path.join(output_dir, 'trained_model.h5'))

if __name__ == "__main__":
    train('config/config.yaml')