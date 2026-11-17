"""CNN used both in the hyperparameter search and for the final model."""
from tensorflow.keras import layers, models, optimizers


def build_cnn(config, input_shape, num_classes):
    model = models.Sequential([layers.Input(shape=input_shape)])

    for i in range(config['num_conv_blocks']):
        model.add(layers.Conv2D(config[f'filters_{i}'], (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2), padding='same'))
        model.add(layers.BatchNormalization())

    model.add(layers.Flatten())
    model.add(layers.Dense(config['dense_units'], activation='relu'))
    model.add(layers.Dropout(config['dropout_rate']))
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(optimizer=optimizers.Adam(learning_rate=config['learning_rate']),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def build_tunable_model(hp, input_shape, num_classes):
    config = {'num_conv_blocks': hp.Int('num_conv_blocks', min_value=2, max_value=3)}
    for i in range(config['num_conv_blocks']):
        config[f'filters_{i}'] = hp.Choice(f'filters_{i}', values=[32, 64, 128])
    config['dense_units'] = hp.Choice('dense_units', values=[64, 128])
    config['dropout_rate'] = hp.Float('dropout_rate', min_value=0.3, max_value=0.5, step=0.1)
    config['learning_rate'] = hp.Choice('learning_rate', values=[1e-3, 1e-4])
    return build_cnn(config, input_shape, num_classes)
