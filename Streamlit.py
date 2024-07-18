import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Preprocessing import static_scale, dynamic_scale
from streamlit_option_menu import option_menu
import time
import mlflow
import mlflow.keras


# Load the saved model
# model = tf.keras.models.load_model('best_model_loss.h5')

# Specify the model name and version
model_name = "StrokePredictionCustomModel"
model_version = 3
# Load the model
model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/{model_version}"
)

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

if selected == "Training":
    st.header('Training')

    # Static input fields
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

    # Dynamic input fields
    st.subheader('Dynamic Input Fields')
    timeseries_data = {
        'Heart Rate': [],
        'Non Invasive Blood Pressure diastolic': [],
        'Non Invasive Blood Pressure mean': [],
        'Non Invasive Blood Pressure systolic': [],
        'Respiratory Rate': []
    }

    for t in range(4):
        st.write(f"Timestep {t+1}")
        dynamic_col1, dynamic_col2 = st.columns(2)
        with dynamic_col1:
            for key in ['Heart Rate', 'Non Invasive Blood Pressure diastolic']:
                value = st.number_input(f'{key} (Timestep {t+1})', min_value=0.0, max_value=200.0, step=0.1, key=f"{key}_{t}")
                timeseries_data[key].append(value)
        with dynamic_col2:
            for key in ['Non Invasive Blood Pressure mean', 'Non Invasive Blood Pressure systolic', 'Respiratory Rate']:
                value = st.number_input(f'{key} (Timestep {t+1})', min_value=0.0, max_value=200.0, step=0.1, key=f"{key}_{t}")
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
    timeseries_input = dynamic_df.values.reshape(1, 4, -1)

    # Predict button
    if st.button('Predict'):
        prediction = model.predict([timeseries_input, static_input])
        st.write(f"Prediction: {prediction[0][0]:.4f}")

if selected == 'Data Viz':
    st.header('Data Visualization')

    # Load the locally stored DataFrame
    data_path = st.text_input('Enter the path of your data file:')
    if data_path:
        try:
            df = pd.read_csv(data_path)  # Assuming the file is a CSV
            st.write(df.head())
            
            # Visualizations

            # Number of patients
            num_patients = df['SUBJECT_ID'].nunique()
            st.write(f"Number of patients: {num_patients}")

            # Average hospitalization time
            hospitalization_time = df.groupby('HADM_ID').size().mean()
            st.write(f"Average hospitalization time: {hospitalization_time:.2f} timesteps")

            # Real-time Heart Rate for a selected patient
            selected_patient = st.selectbox('Select Patient ID', df['SUBJECT_ID'].unique())
            if selected_patient:
                patient_data = df[df['SUBJECT_ID'] == selected_patient]
                heart_rate_data = patient_data[['HADM_ID', 'Heart Rate']].set_index('HADM_ID')

                st.subheader('Real-time Heart Rate')
                placeholder = st.empty()
                for _ in range(100):
                    heart_rate_data = heart_rate_data.append(
                        pd.DataFrame({'Heart Rate': [np.random.randn()]}, index=[heart_rate_data.index[-1] + 1])
                    )
                    heart_rate_data = heart_rate_data.iloc[-100:]
                    with placeholder.container():
                        st.line_chart(heart_rate_data)
                    time.sleep(0.1)  # Update every 100 ms

            # Additional Visualizations

            # Age distribution
            st.subheader('Age Distribution')
            plt.figure(figsize=(10, 6))
            plt.hist(df['AGE'], bins=20, color='skyblue', edgecolor='black')
            plt.xlabel('Age')
            plt.ylabel('Frequency')
            plt.title('Distribution of Ages')
            st.pyplot(plt)

            # Gender distribution
            st.subheader('Gender Distribution')
            gender_counts = df['GENDER'].value_counts()
            st.bar_chart(gender_counts)

            # Blood pressure analysis by age
            st.subheader('Blood Pressure by Age')
            fig, ax = plt.subplots(figsize=(10, 6))
            for bp_type in ['Non Invasive Blood Pressure diastolic', 'Non Invasive Blood Pressure mean', 'Non Invasive Blood Pressure systolic']:
                ax.plot(df['AGE'], df[bp_type], label=bp_type)
            ax.set_xlabel('Age')
            ax.set_ylabel('Blood Pressure')
            ax.set_title('Blood Pressure by Age')
            ax.legend()
            st.pyplot(fig)

            # Vital parameters over time for a selected patient
            st.subheader('Vital Parameters Over Time')
            if selected_patient:
                vitals = ['Heart Rate', 'Non Invasive Blood Pressure diastolic', 'Non Invasive Blood Pressure mean', 'Non Invasive Blood Pressure systolic', 'Respiratory Rate']
                fig, ax = plt.subplots(figsize=(10, 6))
                for vital in vitals:
                    ax.plot(patient_data['HADM_ID'], patient_data[vital], label=vital)
                ax.set_xlabel('Time')
                ax.set_ylabel('Value')
                ax.set_title(f'Vital Parameters Over Time for Patient {selected_patient}')
                ax.legend()
                st.pyplot(fig)

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

        except Exception as e:
            st.error(f"Error loading data: {e}")
