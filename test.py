import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import seaborn as sns
import warnings
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn.metrics import confusion_matrix
from kerastuner.tuners import RandomSearch
from kerastuner import Objective
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import seaborn as sns
import warnings
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.regularizers import l2
from kerastuner.tuners import RandomSearch
from kerastuner import Objective
from sklearn.metrics import confusion_matrix


warnings.filterwarnings('ignore')

# Load data
X_train = np.load('Processed_data/X_train.npy')
y_train = np.load('Processed_data/y_train.npy')
X_val = np.load('Processed_data/X_val.npy')
y_val = np.load('Processed_data/y_val.npy')
X_test = np.load('Processed_data/X_test.npy')
y_test = np.load('Processed_data/y_test.npy')

# Extract static features
X_train_static = X_train[:, 0, :4]
X_val_static = X_val[:, 0, :4]
X_test_static = X_test[:, 0, :4]

def build_model(hp):
    n_static = 4
    n_timesteps = 4
    n_dynamic = 12
    n_output = 1

    recurrent_input = keras.Input(shape=(n_timesteps, n_dynamic), name="TIMESERIES_INPUT")
    static_input = keras.Input(shape=(n_static, ), name="STATIC_INPUT")

    rec_layer_one = layers.Bidirectional(layers.LSTM(
        units=hp.Int('lstm_units_1', min_value=64, max_value=256, step=32),
        kernel_regularizer=l2(hp.Choice('l2_1', values=[0.01, 0.001, 0.0001])),
        recurrent_regularizer=l2(hp.Choice('l2_2', values=[0.01, 0.001, 0.0001])),
        return_sequences=True
    ), name="BIDIRECTIONAL_LAYER_1")(recurrent_input)
    rec_layer_one = layers.Dropout(hp.Choice('dropout_1', values=[0.1, 0.2, 0.3]), name="DROPOUT_LAYER_1")(rec_layer_one)

    rec_layer_two = layers.Bidirectional(layers.LSTM(
        units=hp.Int('lstm_units_2', min_value=32, max_value=128, step=32),
        kernel_regularizer=l2(hp.Choice('l2_3', values=[0.01, 0.001, 0.0001])),
        recurrent_regularizer=l2(hp.Choice('l2_4', values=[0.01, 0.001, 0.0001]))
    ), name="BIDIRECTIONAL_LAYER_2")(rec_layer_one)
    rec_layer_two = layers.Dropout(hp.Choice('dropout_2', values=[0.1, 0.2, 0.3]), name="DROPOUT_LAYER_2")(rec_layer_two)

    static_layer_one = layers.Dense(
        units=hp.Int('dense_units_1', min_value=32, max_value=128, step=32),
        kernel_regularizer=l2(hp.Choice('l2_5', values=[0.01, 0.001, 0.0001])),
        activation=hp.Choice('activation_1', values=['relu', 'tanh']),
        name="DENSE_LAYER_1"
    )(static_input)

    combined = layers.Concatenate(axis=1, name="CONCATENATED_TIMESERIES_STATIC")([rec_layer_two, static_layer_one])
    combined_dense_two = layers.Dense(
        units=hp.Int('dense_units_2', min_value=32, max_value=128, step=32),
        activation=hp.Choice('activation_2', values=['relu', 'tanh']),
        name="DENSE_LAYER_2"
    )(combined)

    output = layers.Dense(n_output, activation='sigmoid', name="OUTPUT_LAYER")(combined_dense_two)

    model = keras.Model(inputs=[recurrent_input, static_input], outputs=[output])
    model.compile(
        loss='binary_crossentropy',
        optimizer=keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', values=[0.001, 0.01, 0.1])),
        metrics=[
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc'),
            keras.metrics.AUC(name='prc', curve='PR')
        ]
    )
    return model

# MLflow setup
experiment_name = "stroke_Models"
mlflow.set_experiment(experiment_name)

# Create an instance of the RandomSearch tuner
tuner = RandomSearch(
    build_model,
    objective=Objective("val_accuracy", direction="max"), # You can change the objective metric
    max_trials=2, # Number of different hyperparameter combinations to try
    executions_per_trial=1, # Number of times to train the model for each trial
    directory='my_dir', # Directory where results are saved
    project_name='intro_to_kt' # Name of the project
)

# Start MLflow run
with mlflow.start_run() as run:
    # Log parameters
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("epochs", 1)
    class_weight= { 0: 0.52 , 1 : 15.14 }
    # Perform the search
    tuner.search(
        [X_train[:, :, 4:], X_train_static], y_train,
        epochs=3, # Number of epochs to train each model
        validation_data=([X_val[:, :, 4:], X_val_static], y_val)
    )

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Build the model with the optimal hyperparameters
    model = build_model(best_hps)

    # Model callbacks including MLflow callback
    early_stopping_loss = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min',
        verbose=1,
        restore_best_weights=True
    )
    checkpoint_loss = keras.callbacks.ModelCheckpoint(
        "best_weights_loss.h5",
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [early_stopping_loss, checkpoint_loss]

    # Train the final model
    history = model.fit(
        x=[X_train[:, :, 4:], X_train_static],
        y=y_train,
        epochs=1,
        batch_size=32,
        validation_data=([X_val[:, :, 4:], X_val_static], y_val),
        callbacks=callbacks_list,
        class_weight=class_weight
    )

    # Log metrics
    for epoch in range(len(history.history['loss'])):
        for metric in history.history:
            mlflow.log_metric(metric, history.history[metric][epoch], step=epoch)

    # Evaluate the model
    results = model.evaluate([X_test[:,:,4:], X_test_static], y_test)
    for metric, value in zip(model.metrics_names, results):
        mlflow.log_metric(metric, value)

    # Log the model
    mlflow.keras.log_model(model, "model")

    # Confusion matrix
    preds = model.predict([X_test[:,:,4:], X_test_static])
    preds = (preds > 0.5).astype(int)  # Thresholding
    print(f"{preds[0]}")
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

    # Register the model
    mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/model", "StrokePredictionCustomModel")