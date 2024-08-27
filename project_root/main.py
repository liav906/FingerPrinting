from src.train import train
from src.validate import validate
from src.test import test

if __name__ == "__main__":
    config_path = 'config/config.yaml'
    train(config_path)
    validate(config_path)
    test(config_path)