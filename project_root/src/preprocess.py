import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(config):
    train_data = pd.read_csv(config['paths']['train_data'])
    val_data = pd.read_csv(config['paths']['val_data'])
    test_data = pd.read_csv(config['paths']['test_data'])
    return train_data, val_data, test_data

def preprocess_data(data, scaler=None, is_training=True):
    features = data.drop(['x', 'y'], axis=1)  # Adjust this to drop columns not used as features
    labels = data[['x', 'y']]  # Output labels, which are the true coordinates (latitude, longitude or x, y)

    if is_training:
        scaler = StandardScaler().fit(features)  # Fit scaler on training data
        features = scaler.transform(features)  # Transform training features
    else:
        features = scaler.transform(features)  # Use existing scaler for validation/test features

    return features, labels, scaler

def split_data(features, labels, config):
    from sklearn.model_selection import train_test_split
    return train_test_split(features, labels, test_size=config['training']['validation_split'], random_state=42)
