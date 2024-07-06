import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from Preprocessing import static_scale, dynamic_scale
from streamlit_option_menu import option_menu

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
     st.header('Stroke Prediction App')
    # Create a row layout
     c1, c2 = st.columns(2)
     c3, c4 = st.columns(2)

     with st.container():
        c1.write("c1")
        c2.write("c2")

     with st.container():
        c3.write("c3")
        c4.write("c4")

     with c1:
        chart_data = pd.DataFrame(np.random.randn(20, 3), columns=['a', 'b', 'c'])
        st.area_chart(chart_data)

     with c2:
        chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
        st.bar_chart(chart_data)

     with c3:
        chart_data = pd.DataFrame(np.random.randn(20, 3), columns=['a', 'b', 'c'])
        st.line_chart(chart_data)

     with c4:
        chart_data = pd.DataFrame(np.random.randn(20, 3), columns=['a', 'b', 'c'])
        st.line_chart(chart_data)