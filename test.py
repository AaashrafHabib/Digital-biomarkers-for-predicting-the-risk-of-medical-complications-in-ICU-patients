import pandas as pd
import matplotlib.pyplot as plt
import test
import mlflow.keras
import test
import mlflow.sklearn
from sklearn.metrics import accuracy_score
train_df = pd.read_csv(r"F:\MLOPS\data\train_data.csv")
val_df = pd.read_csv(r"F:\MLOPS\data\val_data.csv")
test_df = pd.read_csv(r"F:\MLOPS\data\test_data.csv")






"""# Organisation des colonnes"""

new_columns = [col for col in train_df.columns if col != 'stroke'] + ['stroke']

train_df=train_df[new_columns]
val_df=val_df[new_columns]
test_df=test_df[new_columns]


# @title HADM_ID

from matplotlib import pyplot as plt
train_df['HADM_ID'].plot(kind='hist', bins=20, title='HADM_ID')
plt.gca().spines[['top', 'right',]].set_visible(False)

filtered_df = train_df[train_df['stroke'] == 1]

proportions = train_df['stroke'].value_counts(normalize=True)
# Définition des couleurs et des labels
colors = ['lightcoral', 'lightskyblue']
labels = ['Stroke patients', 'None stroke patients']

# Tracer le diagramme à secteurs (pie chart)
plt.figure(figsize=(6, 6))
plt.pie(proportions, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)

# Personnalisation du graphique
plt.title('Distribution of stroke and none stroke patients')
plt.axis('equal')  # Assure que le diagramme est un cercle plutôt qu'une ellipse

# Afficher le graphique
plt.show()

#Imports
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Input,Dense,LSTM,Dropout,Conv1D,MaxPooling1D,Flatten,Bidirectional,TimeDistributed,Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import tensorflow.keras as keras
warnings.filterwarnings('ignore')
FEATURES=10
TIMESTEPS=4
OUTPUT=1
METRICS = [
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]

"""# Mise à l'échelle (Scaling)

"""

from sklearn.preprocessing import StandardScaler,RobustScaler
def static_scale(df):
    data=df.copy()
    scaler=RobustScaler()
    static_cols=["AGE",'Admission Weight (Kg)',"Height (cm)"]
    data[static_cols]=scaler.fit_transform(data[static_cols])
    return data,scaler

from sklearn.preprocessing import StandardScaler,RobustScaler
from tqdm import tqdm
def dynamic_scale(df):
    final_df=pd.DataFrame()
    cols=['Heart Rate',
                  'Non Invasive Blood Pressure diastolic', 'Non Invasive Blood Pressure mean',
                 'Non Invasive Blood Pressure systolic', 'Respiratory Rate']
    for hadm in tqdm(df['HADM_ID'].unique()):
        scaler=RobustScaler()
        #extract the dataframe
        hadm_df=df[df['HADM_ID']==hadm]
        #scale the float cols
        hadm_df[cols]=scaler.fit_transform(hadm_df[cols])
#         print(hadm_df)
        final_df = pd.concat([final_df,hadm_df])
    return final_df

"""# Robust Scaling des colonnes à valeurs statiques"""

static_cols=["AGE",'Admission Weight (Kg)',"Height (cm)"]
scaled_train,scaler=static_scale(train_df)
val_df[static_cols]=scaler.transform(val_df[static_cols])
test_df[static_cols]=scaler.transform(test_df[static_cols])

"""# Standard Scaling des colonnes à valeurs dynamiques"""

scaled_train=dynamic_scale(scaled_train)
val_df=dynamic_scale(val_df)
test_df=dynamic_scale(test_df)

"""# Window creation function for creating sequences from time series data"""

#apply this function for eacxh unique admission
#so we won't mix timestamps from different admission in the same training sequence
def create_window(df):
    n_observation=4
    n_forecast=2
    n_target=1
    X,y=[],[]
    for adm_id in tqdm(df['HADM_ID'].unique()):
        data=df[df['HADM_ID']==adm_id]
#         target_cols=["stroke"]
#         target_df=data[target_cols]
    #     data.drop(target_cols,axis=1,inplace=True)
        for i in range(len(data)-8):
            X.append(data.iloc[i:i+n_observation,2:])
            y.append(data.iloc[i+n_observation+n_forecast:i+n_observation+n_forecast+n_target,-1])
    return np.array(X),np.array(y)

"""#Class balancing with weights"""

neg,pos=np.bincount(train_df['stroke'])
total=neg+pos
print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
    total, pos, 100 * pos / total))
# Scaling by total/2 helps keep the loss to a similar magnitude.
# The sum of the weights of all examples stays the same.
weight_for_0 = (1 / neg) * (total / 2.0)
weight_for_1 = (1 / pos) * (total / 2.0)

class_weight = {0: weight_for_0, 1: weight_for_1}

print('Weight for class 0: {:.2f}'.format(weight_for_0))
print('Weight for class 1: {:.2f}'.format(weight_for_1))

X_train,y_train=create_window(scaled_train)
X_val,y_val=create_window(val_df)
X_test,y_test=create_window(test_df)
print(X_train.shape," ",y_train.shape)
print(X_val.shape," ",y_val.shape)
print(X_test.shape," ",y_test.shape)

"""# Conversion des prédictions en binaire"""

def clean_preds(model,data):
    real_predictions=[]
    predictions=model.predict(data)
    for pred in predictions:
        if pred > 0.5:
            real_predictions.append(1)
        else:
            real_predictions.append(0)
    return np.array(real_predictions)

from sklearn.metrics import f1_score, confusion_matrix, classification_report
import seaborn as sns
def plot_cm (real,preds):
    plt.figure(figsize=(6,6))
    cm = confusion_matrix(real,preds)
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in  cm.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in  zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cm,annot=labels,fmt="",cmap='Blues')
    print(classification_report(real, preds))
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
#     plt.title('STEP 1')

"""# Hyperparameters tunning using Keras-Tuner

---
"""


#extracting static feats
X_train_static=X_train[:, 0, :4]
X_val_static=X_val[:,0,:4]
X_test_static=X_test[:,0,:4]

"""## Custom model"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from kerastuner.tuners import RandomSearch
from kerastuner import Objective
def build_model(hp):
    # ... [rest of the code remains unchanged]
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

"""# Using MLflow to track all metrics , model versionning , easy deployment to different cloud services ect !...."""


# Set the experiment
experiment_description = "This is the stroke prediction project"
experiment_tags = {
    "project_name": "stroke_prediction",
    "mlflow.note.content": experiment_description,
}

# Create the experiment (ensure experiment name is unique)
experiment_name = "stroke_Models"
experiment_id = test.create_experiment(experiment_name, tags=experiment_tags)

# Start an MLflow run
with test.start_run() as run:
    # Log parameters
    test.log_param("batch_size", 32)
    test.log_param("epochs", 1)

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

    # Train the model
    model = build_model
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
    for epoch, metrics in enumerate(history.history):
        for metric, value in metrics.items():
            test.log_metric(metric, value[-1], step=epoch)

    # Evaluate the model
    results = model.evaluate([X_test[:,:,4:],X_test_static], y_test)
    for metric, value in zip(model.metrics_names, results):
        test.log_metric(metric, value)

    # Log the model
    test.keras.log_model(model, "model")

    # Save confusion matrix as an artifact
    preds = clean_preds(model, [X_test[:,:,4:],X_test_static])
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")
    test.log_artifact("confusion_matrix.png")

    # Register the model
    test.register_model("runs:/{}/model".format(run.info.run_id), "StrokePredictionModel")

    # You can also log other artifacts like model architecture, plots, etc.