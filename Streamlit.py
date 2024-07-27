import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Preprocessing import static_scale, dynamic_scale
from streamlit_option_menu import option_menu
import time
from keras_tuner import RandomSearch
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import explained_variance_score, mean_squared_error, r2_score
import seaborn as sns
import time

# Load the saved model
model = tf.keras.models.load_model('best_model_loss.h5')

# Define input fields for the Streamlit app
st.title("Stroke Prediction")
st.write("Please enter the following information:")

# Sidebar for navigation
with st.sidebar:
    selected = option_menu('Stroke Prediction System',
                           ['Data Viz',
                            'Training',
                            'Deploying'],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart', 'person'],
                           default_index=0)

def load_data(data_path):
    return pd.read_csv(data_path)  # Adjust if needed

def get_last_timestamp(patient_data):
    return patient_data['timestamp'].max()  # Adjust based on your data

if selected == "Training":
    st.header('Training')

    patient_selection = st.radio("Select patient from:", ('Existing patients', 'New patient'))
    
    if patient_selection == 'Existing patients':
        # Load and select patient data
        data_path = st.text_input('Enter the path of your data file:')
        df = load_data(data_path) if data_path else None
        
        if df is not None:
            patient_ids = df['SUBJECT_ID'].unique()
            selected_patient = st.selectbox('Select Patient ID', patient_ids)
            
            if selected_patient:
                # patient_data = df[df['SUBJECT_ID'] == selected_patient]
                # last_timestamp = get_last_timestamp(patient_data)
                # st.write(f"Last measurement was on: {last_timestamp}")
                
                prediction_day = st.date_input('Select the day for prediction')
                # prediction_timestamp = (prediction_day - pd.to_datetime(last_timestamp)).days
                
                # st.write(f"Predicting for {prediction_timestamp} days ahead")

                # for t in range(last_timestamp, last_timestamp + prediction_timestamp):
                #     st.write(f"Timestep {t + 1}")
                #     dynamic_col1, dynamic_col2 = st.columns(2)
                #     with dynamic_col1:
                #         for key in ['Heart Rate', 'Non Invasive Blood Pressure diastolic']:
                #             value = st.number_input(f'{key} (Timestep {t + 1})', min_value=0.0, max_value=200.0, step=0.1,
                #                                     key=f"{key}_{t}")
                #             timeseries_data[key].append(value)
                #     with dynamic_col2:
                #         for key in ['Non Invasive Blood Pressure mean', 'Non Invasive Blood Pressure systolic', 'Respiratory Rate']:
                #             value = st.number_input(f'{key} (Timestep {t + 1})', min_value=0.0, max_value=200.0, step=0.1,
                #                                     key=f"{key}_{t}")
                #             timeseries_data[key].append(value)

                # Add fake HADM_ID column and preprocess
                # dynamic_df = pd.DataFrame(timeseries_data)
                # dynamic_df['HADM_ID'] = 1

                # # Preprocess the input data
                # dynamic_df = dynamic_scale(dynamic_df)
                # dynamic_df = dynamic_df.drop(columns=['HADM_ID'])

                # # Convert to numpy arrays
                # timeseries_input = dynamic_df.values.reshape(1, prediction_timestamp, -1)
                
                # Further processing can be added here based on the model requirements
                
        else:
            st.error("Please provide a valid data path.")

    elif patient_selection == 'New patient':
        # Static input fields for new patient
        st.subheader('Static Input Fields')
        static_col1, static_col2 = st.columns(2)
        with static_col1:
            age = st.number_input('Age', min_value=0, max_value=120, step=1)
        with static_col2:
            weight = st.number_input('Admission Weight (Kg)', min_value=0.0, max_value=200.0, step=0.1)
        static_col3, static_col4 = st.columns(2)
        with static_col3:
            height = st.number_input('Height (cm)', min_value=0.0, max_value=250.0, step=0.1)
        static_input = [age, weight, height]

        prediction_days = st.slider('Number of days to predict', min_value=1, max_value=10, value=6)

        # Dynamic input fields for new patient
        st.subheader('Dynamic Input Fields')
        timeseries_data = {
            'Heart Rate': [],
            'Non Invasive Blood Pressure diastolic': [],
            'Non Invasive Blood Pressure mean': [],
            'Non Invasive Blood Pressure systolic': [],
            'Respiratory Rate': []
        }

        for t in range(prediction_days):
            st.write(f"Timestep {t + 1}")
            dynamic_col1, dynamic_col2 = st.columns(2)
            with dynamic_col1:
                for key in ['Heart Rate', 'Non Invasive Blood Pressure diastolic']:
                    value = st.number_input(f'{key} (Timestep {t + 1})', min_value=0.0, max_value=200.0, step=0.1,
                                            key=f"{key}_{t}")
                    timeseries_data[key].append(value)
            with dynamic_col2:
                for key in ['Non Invasive Blood Pressure mean', 'Non Invasive Blood Pressure systolic', 'Respiratory Rate']:
                    value = st.number_input(f'{key} (Timestep {t + 1})', min_value=0.0, max_value=200.0, step=0.1,
                                            key=f"{key}_{t}")
                    timeseries_data[key].append(value)

        # Convert inputs to DataFrame
        static_df = pd.DataFrame([static_input], columns=["AGE", 'Admission Weight (Kg)', "Height (cm)"])
        dynamic_df = pd.DataFrame(timeseries_data)

        # Add a fake HADM_ID column to use the dynamic_scale function
        dynamic_df['HADM_ID'] = 1

        # Preprocess the input data
        static_df, scaler = static_scale(static_df)
        dynamic_df = dynamic_scale(dynamic_df)

        # Drop the fake HADM_ID column
        dynamic_df = dynamic_df.drop(columns=['HADM_ID'])

        # Convert to numpy arrays
        static_input = static_df.values
        timeseries_input = dynamic_df.values.reshape(1, prediction_days, -1)
        
        # Further processing can be added here based on the model requirements


    # Model selection
    model_selection = st.selectbox(
        'Select a Model',
        ['LSTM ( Long Short Term Memory)', 'GRU (Gated Recurrent Unit)', 'Custom Model']
    )

    # Cross-validation
    cross_val_folds = st.slider('Number of Cross-validation Folds', min_value=2, max_value=10, value=5)

    # Hyperparameter tuning options
    if model_selection == 'LSTM ( Long Short Term Memory)':
        st.subheader('Hyperparameter Tuning')
        tune_neurons = st.slider('Number of Neurons in Hidden Layer', min_value=32, max_value=512, step=32, value=128)
        tune_epochs = st.slider('Number of Training Epochs', min_value=5, max_value=100, step=5, value=20)
        
    # Mock data for demonstration purposes
    y_true = np.random.rand(100)
    y_pred_nn = y_true + np.random.normal(0, 0.05, 100)
    y_pred_rf = y_true + np.random.normal(0, 0.07, 100)
    y_pred_gb = y_true + np.random.normal(0, 0.06, 100)

    # Calculate statistics for demonstration
    model_metrics = {
        'LSTM ( Long Short Term Memory)': {
            'Variance Explained': explained_variance_score(y_true, y_pred_nn),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred_nn)),
            'R2': r2_score(y_true, y_pred_nn),
            'Predictions': y_pred_nn
        },
        'GRU (Gated Recurrent Unit)': {
            'Variance Explained': explained_variance_score(y_true, y_pred_rf),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred_rf)),
            'R2': r2_score(y_true, y_pred_rf),
            'Predictions': y_pred_rf
        },
        'Custom Model': {
            'Variance Explained': explained_variance_score(y_true, y_pred_gb),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred_gb)),
            'R2': r2_score(y_true, y_pred_gb),
            'Predictions': y_pred_gb
        }
    }

    # Train button
    if st.button('Train'):
        st.subheader('Model Summary')

        # Display summary statistics
        best_model = max(model_metrics, key=lambda m: model_metrics[m]['Variance Explained'])
        best_model_metrics = model_metrics[best_model]

        st.write(f"*Best Model:* {best_model}")
        st.write(f"*Variance Explained:* {best_model_metrics['Variance Explained'] * 100:.1f} %")
        st.write(f"*RMSE:* {best_model_metrics['RMSE']:.2f}")
        st.write(f"*Model Type:* Regression")
        st.write(f"*Validation Folds:* {cross_val_folds}")
        st.write(f"*Models Trained:* 37")

        # Model Rankings
        st.subheader('Model Rankings')
        model_names = list(model_metrics.keys())
        explained_variances = [model_metrics[m]['Variance Explained'] for m in model_names]

        fig, ax = plt.subplots()
        ax.barh(model_names, explained_variances, color='skyblue')
        ax.set_xlabel('Explained Variance')
        ax.set_title('Model Rankings')
        st.pyplot(fig)

        # Individual Predicted vs. Observed Values Plots in separate columns
        st.subheader('Predicted vs. Observed Values')
        col1, col2, col3 = st.columns(3)
        
        with col1:
            model = 'Neural Network'
            fig, ax = plt.subplots()
            ax.scatter(y_true, model_metrics[model]['Predictions'], label=model, alpha=0.5)
            ax.plot([0, 1], [0, 1], color='black', linestyle='--')
            ax.set_xlabel('Observed Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title(f'{model}')
            ax.legend()
            st.pyplot(fig)
        
        with col2:
            model = 'Random Forest'
            fig, ax = plt.subplots()
            ax.scatter(y_true, model_metrics[model]['Predictions'], label=model, alpha=0.5)
            ax.plot([0, 1], [0, 1], color='black', linestyle='--')
            ax.set_xlabel('Observed Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title(f'{model}')
            ax.legend()
            st.pyplot(fig)

        with col3:
            model = 'Gradient Boosting'
            fig, ax = plt.subplots()
            ax.scatter(y_true, model_metrics[model]['Predictions'], label=model, alpha=0.5)
            ax.plot([0, 1], [0, 1], color='black', linestyle='--')
            ax.set_xlabel('Observed Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title(f'{model}')
            ax.legend()
            st.pyplot(fig)

        # Statistics
        st.subheader('Statistics')
        for model in model_metrics:
            st.write(f"*{model}*")
            st.write(f"Variance Explained: {model_metrics[model]['Variance Explained'] * 100:.1f} %")
            st.write(f"RMSE: {model_metrics[model]['RMSE']:.2f}")
            st.write(f"R2: {model_metrics[model]['R2']:.2f}")
            st.write("")

# Data Visualization Section
selected = 'Data Viz'  # Assuming you are running this block based on some condition
if selected == 'Data Viz':
    st.header('Data Visualization')

    # Load the locally stored DataFrame
    data_path = st.text_input('Enter the path of your data file:')
    df = None
    if data_path:
        try:
            df = pd.read_csv(data_path)  # Assuming the file is a CSV
            df.index = pd.to_datetime(df.index)  # Assuming the index is the CHARTTIME
            st.write(df.head())

            # Number of patients
            num_patients = df['SUBJECT_ID'].nunique()
            st.write(f"Number of patients: {num_patients}")

            # Average hospitalization time
            hospitalization_time = df.groupby('HADM_ID').size().mean()
            st.write(f"Average hospitalization time: {hospitalization_time:.2f} timesteps")

            # Real-time vital signs for a selected patient
            selected_patient = st.selectbox('Select Patient ID', df['SUBJECT_ID'].unique())
            if selected_patient:
                patient_data = df[df['SUBJECT_ID'] == selected_patient]

                # List all hospitalizations for the selected patient
                hospitalizations = patient_data['HADM_ID'].unique()
                selected_hospitalization = st.selectbox('Select Hospitalization ID', hospitalizations)
                if selected_hospitalization:
                    hospitalization_data = patient_data[patient_data['HADM_ID'] == selected_hospitalization]
                    hospitalization_data = hospitalization_data.set_index('CHARTTIME')

                    # Select vital sign to visualize
                    selected_vital_sign = st.selectbox('Select Vital Sign to Visualize',
                                                       ['Heart Rate',
                                                        'Non Invasive Blood Pressure diastolic',
                                                        'Non Invasive Blood Pressure mean',
                                                        'Non Invasive Blood Pressure systolic',
                                                        'Respiratory Rate'])
                    if selected_vital_sign:
                        st.subheader(f'{selected_vital_sign} Variation Over Time')
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(hospitalization_data.index, hospitalization_data[selected_vital_sign], label=selected_vital_sign)
                        ax.set_xlabel('Time')
                        ax.set_ylabel('Value')
                        ax.set_title(f'{selected_vital_sign} Variation for Patient {selected_patient} during Hospitalization {selected_hospitalization}')
                        ax.legend()
                        st.pyplot(fig)

        except Exception as e:
            st.error(f"Error loading data: {e}")

    if df is not None:
        c1, c2 = st.columns(2)

        with c1:
            # Age distribution
            st.subheader('Age Distribution')
            plt.figure(figsize=(10, 6))
            plt.hist(df['AGE'], bins=20, color='skyblue', edgecolor='black')
            plt.xlabel('Age')
            plt.ylabel('Frequency')
            plt.title('Distribution of Ages')
            st.pyplot(plt)

        with c2:
            # Gender distribution
            st.subheader('Gender Distribution')
            fig, ax = plt.subplots(figsize=(10, 6))
            gender_counts = df['GENDER'].value_counts()
            ax.bar(gender_counts.index, gender_counts.values, color='skyblue', edgecolor='black')
            ax.set_xlabel('Gender')
            ax.set_ylabel('Count')
            ax.set_title('Gender Distribution')
            st.pyplot(fig)

        c3, c4 = st.columns(2)

        with c3:
            # Blood pressure analysis by age
            st.subheader('Blood Pressure by Age')
            fig, ax = plt.subplots(figsize=(12, 8))  # Adjusted to match the density plots' size
            for bp_type in ['Non Invasive Blood Pressure diastolic', 'Non Invasive Blood Pressure mean',
                            'Non Invasive Blood Pressure systolic']:
                ax.plot(df['AGE'], df[bp_type], label=bp_type)
            ax.set_xlabel('Age')
            ax.set_ylabel('Blood Pressure')
            ax.set_title('Blood Pressure by Age')
            ax.legend()
            st.pyplot(fig)

        with c4:
            # Density plots for Respiratory Rate and Admission Weight
            st.subheader('Density Plots')
            fig, ax = plt.subplots(1, 2, figsize=(12, 8))  # Adjusted to match the blood pressure plot's size

            sns.kdeplot(df['Respiratory Rate'], ax=ax[0], color='blue')
            ax[0].set_title('Density of Respiratory Rate')
            ax[0].set_xlabel('Respiratory Rate')
            ax[0].set_ylabel('Density')

            sns.kdeplot(df['Admission Weight (Kg)'], ax=ax[1], color='green')
            ax[1].set_title('Density of Admission Weight')
            ax[1].set_xlabel('Admission Weight (Kg)')
            ax[1].set_ylabel('Density')

            st.pyplot(fig)
        c5, c6 = st.columns(2)

        with c5:
            # Gender distribution
            st.subheader('Gender Distribution')
            fig, ax = plt.subplots(figsize=(10, 6))
            gender_counts = df['GENDER'].value_counts()
            ax.bar(gender_counts.index, gender_counts.values, color='skyblue', edgecolor='black')
            ax.set_xlabel('Gender')
            ax.set_ylabel('Count')
            ax.set_title('Gender Distribution')
            st.pyplot(fig)
            # Stroke and anomaly analysis
            st.subheader('Stroke and Anomaly Analysis')
            fig, ax = plt.subplots(figsize=(10, 6))
            stroke_counts = df['stroke'].value_counts()
            ax.bar(stroke_counts.index,stroke_counts.values,color='skyblue', edgecolor='black')
            ax.set_xlabel('stroke')
            ax.set_ylabel('Count')
            ax.set_title('Stroke Distribution')
            
        with c6 :
         # Age distribution comparison between positive and negative cases
           st.subheader('Age Distribution: Positive vs Negative Cases')
           df['Status'] = df['stroke'].apply(lambda x: 'positive' if x == 1 else 'negative')
           plt.figure(figsize=(10, 6))
           sns.histplot(data=df, x='AGE', hue='Status', kde=True, stat="density", common_norm=False, palette=['#1f77b4', '#ff7f0e'])
           plt.xlabel('AGE')
           plt.ylabel('Density')
           plt.title('Age Distribution')
           plt.legend(title='Status', labels=['positive', 'negative'])
           st.pyplot(plt)