import pandas as pd

# Load the dataset
data_path = 'F:/MLOPS/data V2/train.csv'  # Replace with the path to your dataset
df = pd.read_csv(data_path)

# Group by SUBJECT_ID and count the number of unique HADM_IDs for each patient
hospitalizations_count = df.groupby('SUBJECT_ID')['HADM_ID'].nunique()

# Filter to get patients with multiple hospitalizations
patients_multiple_hospitalizations = hospitalizations_count[hospitalizations_count > 2].index.tolist()

print("Patients with multiple hospitalizations:")
print(patients_multiple_hospitalizations)
