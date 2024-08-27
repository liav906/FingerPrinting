import os
import yaml
from preprocess import load_data, preprocess_data
from model import build_model
from logger import get_logger

def test(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    output_dir = sorted(os.listdir(config['paths']['outputs']))[-1]
    model_path = os.path.join(config['paths']['outputs'], output_dir, 'trained_model.h5')
    logger = get_logger(os.path.join(config['paths']['logs'], f"test_{output_dir}.log"))

    # Load data
    _, _, test_data = load_data(config)
    test_features, test_labels, scaler = preprocess_data(test_data, is_training=False)

    # Load model
    model = build_model(config)
    model.load_weights(model_path)

    # Evaluate model
    test_loss, test_mae = model.evaluate(test_features, test_labels)
    logger.info(f"Test completed. Loss: {test_loss:.4f}, MAE: {test_mae:.4f}")

if __name__ == "__main__":
    test('config/config.yaml')