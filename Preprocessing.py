import pandas as pd
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm

def dynamic_scale(df):
    final_df = pd.DataFrame()
    cols = ['Heart Rate', 'Non Invasive Blood Pressure diastolic', 'Non Invasive Blood Pressure mean',
            'Non Invasive Blood Pressure systolic', 'Respiratory Rate']
    for hadm in tqdm(df['HADM_ID'].unique()):
        scaler = RobustScaler()
        hadm_df = df[df['HADM_ID'] == hadm]
        hadm_df[cols] = scaler.fit_transform(hadm_df[cols])
        final_df = pd.concat([final_df, hadm_df])
    return final_df

def static_scale(df):
    data = df.copy()
    scaler = RobustScaler()
    static_cols = ["AGE", 'Admission Weight (Kg)', "Height (cm)"]
    data[static_cols] = scaler.fit_transform(data[static_cols])
    return data, scaler
