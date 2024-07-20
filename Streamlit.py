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

if selected == 'Data Viz':
    st.header('Data Visualization')

    # Load the locally stored DataFrame
    data_path = st.text_input('Enter the path of your data file:')
    df = None
    if data_path:
        try:
            df = pd.read_csv(data_path)  # Assuming the file is a CSV
            st.write(df.head())

            # Number of patients
            num_patients = df['SUBJECT_ID'].nunique()
            st.write(f"Number of patients: {num_patients}")

            # Average hospitalization time
            hospitalization_time = df.groupby('HADM_ID').size().mean()
            st.write(f"Average hospitalization time: {hospitalization_time:.2f} timesteps")

            # Real-time vital signs for a selected patient
            selected_patient = st.selectbox('Select Patient ID', df['SUBJECT_ID'].unique())
            selected_vital_sign = st.selectbox('Select Vital Sign to Visualize',
                                               ['Heart Rate',
                                                'Non Invasive Blood Pressure diastolic',
                                                'Non Invasive Blood Pressure mean',
                                                'Non Invasive Blood Pressure systolic',
                                                'Respiratory Rate'])
            if selected_patient:
                patient_data = df[df['SUBJECT_ID'] == selected_patient]
                vital_sign_data = patient_data[['HADM_ID', selected_vital_sign]].set_index('HADM_ID')

                st.subheader(f'Real-time {selected_vital_sign}')
                placeholder = st.empty()
                for _ in range(100):
                    new_data = pd.DataFrame({
                        selected_vital_sign: [np.random.randn()]
                    }, index=[vital_sign_data.index[-1] + 1])

                    vital_sign_data = pd.concat([vital_sign_data, new_data])
                    vital_sign_data = vital_sign_data.iloc[-100:]

                    with placeholder.container():
                        st.line_chart(vital_sign_data)

                    time.sleep(0.1)  # Update every 100 ms

                # Additional Visualizations
        except Exception as e:
            st.error(f"Error loading data: {e}")

    if df is not None:
        c1, c2 = st.columns(2)

        with c1:
            # Vital parameters over time for a selected patient
            st.subheader('Vital Parameters Over Time')
            if 'selected_patient' in locals():
                vitals = ['Heart Rate', 'Non Invasive Blood Pressure diastolic', 'Non Invasive Blood Pressure mean',
                          'Non Invasive Blood Pressure systolic', 'Respiratory Rate']
                fig, ax = plt.subplots(figsize=(10, 6))
                for vital in vitals:
                    ax.plot(patient_data['HADM_ID'], patient_data[vital], label=vital)
                ax.set_xlabel('Time')
                ax.set_ylabel('Value')
                ax.set_title(f'Vital Parameters Over Time for Patient {selected_patient}')
                ax.legend()
                st.pyplot(fig)

        with c2:
            # Age distribution
            st.subheader('Age Distribution')
            plt.figure(figsize=(10, 6))
            plt.hist(df['AGE'], bins=20, color='skyblue', edgecolor='black')
            plt.xlabel('Age')
            plt.ylabel('Frequency')
            plt.title('Distribution of Ages')
            st.pyplot(plt)

        c3, c4 = st.columns(2)

        with c3:
            # Gender distribution
            st.subheader('Gender Distribution')
            gender_counts = df['GENDER'].value_counts()
            st.bar_chart(gender_counts)

        with c4:
            # Blood pressure analysis by age
            st.subheader('Blood Pressure by Age')
            fig, ax = plt.subplots(figsize=(10, 6))
            for bp_type in ['Non Invasive Blood Pressure diastolic', 'Non Invasive Blood Pressure mean',
                            'Non Invasive Blood Pressure systolic']:
                ax.plot(df['AGE'], df[bp_type], label=bp_type)
            ax.set_xlabel('Age')
            ax.set_ylabel('Blood Pressure')
            ax.set_title('Blood Pressure by Age')
            ax.legend()
            st.pyplot(fig)

        c5, c6 = st.columns(2)

        with c5:
            # Stroke and anomaly analysis
            st.subheader('Stroke and Anomaly Analysis')
            stroke_counts = df['stroke'].value_counts()
            tachycardia_counts = df['Tachycardia'].value_counts()
            bradycardia_counts = df['Bradycardia'].value_counts()
            hypertension_counts = df['Hypertension'].value_counts()
            hypotension_counts = df['Hypotension'].value_counts()
            tachypnea_counts = df['Tachypnea'].value_counts()
            bradypnea_counts = df['Bradypnea'].value_counts()

            anomalies = pd.DataFrame({
                'Stroke': stroke_counts,
                'Tachycardia': tachycardia_counts,
                'Bradycardia': bradycardia_counts,
                'Hypertension': hypertension_counts,
                'Hypotension': hypotension_counts,
                'Tachypnea': tachypnea_counts,
                'Bradypnea': bradypnea_counts
            }).fillna(0).astype(int)

            st.bar_chart(anomalies)