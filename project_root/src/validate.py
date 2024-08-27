import os
import yaml
from preprocess import load_data, preprocess_data
from model import build_model
from logger import get_logger
from map_generator import generate_map

def validate(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    output_dir = sorted(os.listdir(config['paths']['outputs']))[-1]
    model_path = os.path.join(config['paths']['outputs'], output_dir, 'trained_model.h5')
    logger = get_logger(os.path.join(config['paths']['logs'], f"validate_{output_dir}.log"))

    # Load data
    _, val_data, _ = load_data(config)
    val_features, val_labels, scaler = preprocess_data(val_data, is_training=False)

    # Load model
    model = build_model(config)
    model.load_weights(model_path)

    # Predict and evaluate
    predictions = model.predict(val_features)
    logger.info(f"Validation completed. Predictions: {predictions[:5]}")

    # Generate map of predictions vs. real data
    map_file_path = os.path.join(config['paths']['outputs'], output_dir, 'map.html')
    generate_map(predictions, val_labels.to_numpy(), map_file_path)
    logger.info(f"Map generated and saved to {map_file_path}")

if __name__ == "__main__":
    validate('config/config.yaml')