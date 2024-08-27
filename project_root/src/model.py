import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def build_model(config):
    model = Sequential()
    model.add(Dense(config['model']['hidden_layers'][0], input_dim=config['model']['input_size'], activation=config['model']['activation']))
    for units in config['model']['hidden_layers'][1:]:
        model.add(Dense(units, activation=config['model']['activation']))
        model.add(Dropout(config['model']['dropout_rate']))
    model.add(Dense(config['model']['output_size'], activation='linear'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config['training']['learning_rate']),
                  loss='mse',
                  metrics=['mae'])
    return model