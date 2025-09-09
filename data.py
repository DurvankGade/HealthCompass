import pandas as pd
import numpy as np
import datetime

# --- Configuration ---
N_PATIENTS = 200  # Increased patient count for a richer dataset
N_DAYS = 180     # Extended to one year
START_DATE = datetime.date(2024, 1, 1)

# --- Patient Archetype Baselines ---
# Added baselines for new features and smoker probability
archetype_params = {
    'stable': {
        'hr_base': 70, 'bp_s_base': 120, 'spo2_base': 98, 'glucose_base': 95,
        'creatinine_base': 0.8, 'sleep_hours_base': 7.5, 'exercise_mins_base': 45,
        'adherence_prob': 0.95, 'deterioration_base_prob': 0.005,
        'age_range': (45, 65), 'smoker_prob': 0.1
    },
    'unstable': {
        'hr_base': 85, 'bp_s_base': 150, 'spo2_base': 95, 'glucose_base': 160,
        'creatinine_base': 1.1, 'sleep_hours_base': 5.5, 'exercise_mins_base': 10,
        'adherence_prob': 0.60, 'deterioration_base_prob': 0.05,
        'age_range': (60, 80), 'smoker_prob': 0.5
    },
    'declining': {
        'hr_base': 75, 'bp_s_base': 135, 'spo2_base': 97, 'glucose_base': 110,
        'creatinine_base': 0.9, 'sleep_hours_base': 6.5, 'exercise_mins_base': 25,
        'adherence_prob': 0.85, 'deterioration_base_prob': 0.01,
        'age_range': (55, 75), 'smoker_prob': 0.3
    }
}

all_patient_data = []
patient_static_features = {}

print(f"Generating data for {N_PATIENTS} patients over {N_DAYS} days...")

# --- Static Feature Generation Loop ---
for patient_id in range(1, N_PATIENTS + 1):
    archetype_name = np.random.choice(['stable', 'unstable', 'declining'], p=[0.4, 0.3, 0.3])
    params = archetype_params[archetype_name]
    
    # Generate static (once per patient) features
    age = np.random.randint(params['age_range'][0], params['age_range'][1])
    gender = np.random.choice(['Male', 'Female'], p=[0.5, 0.5])
    smoker = np.random.choice([1, 0], p=[params['smoker_prob'], 1 - params['smoker_prob']])
    
    patient_static_features[patient_id] = {
        'archetype': archetype_name,
        'age': age,
        'gender': gender,
        'smoker': smoker
    }

# --- Main Time-Series Generation Loop ---
for patient_id in range(1, N_PATIENTS + 1):
    static_info = patient_static_features[patient_id]
    archetype_name = static_info['archetype']
    params = archetype_params[archetype_name]
    
    # Initialize patient's state
    missed_meds_yesterday = 0
    creatinine_drift = 0

    for day in range(N_DAYS):
        # --- Simulate Medication Adherence ---
        med_adherence = np.random.choice([1, 0], p=[params['adherence_prob'], 1 - params['adherence_prob']])
        
        # --- Simulate Lifestyle Factors ---
        sleep_hours = round(max(3, params['sleep_hours_base'] + np.random.normal(0, 1)), 1)
        exercise_mins = int(max(0, params['exercise_mins_base'] + np.random.normal(0, 10)))

        # --- Calculate Penalties/Bonuses ---
        adherence_penalty = 5 * missed_meds_yesterday
        smoker_penalty_bp = 10 if static_info['smoker'] == 1 else 0
        smoker_penalty_hr = 5 if static_info['smoker'] == 1 else 0
        smoker_penalty_spo2 = 1 if static_info['smoker'] == 1 else 0
        
        # Apply gradual decline if applicable
        decline_factor = (day / N_DAYS) * 25 if archetype_name == 'declining' else 0

        # --- Simulate Vitals ---
        heart_rate = int(params['hr_base'] + np.random.normal(0, 3) + adherence_penalty / 2 + decline_factor / 3 + smoker_penalty_hr)
        systolic_bp = int(params['bp_s_base'] + np.random.normal(0, 5) + adherence_penalty + decline_factor + smoker_penalty_bp)
        diastolic_bp = int(systolic_bp * 0.65 + np.random.normal(0, 4))
        spo2 = round(max(88, min(100, params['spo2_base'] - np.random.normal(0, 0.5) - (adherence_penalty / 10) - (decline_factor / 10) - smoker_penalty_spo2)), 1)
        
        # --- Simulate Lab Results ---
        glucose = int(params['glucose_base'] + np.random.normal(0, 10) + (adherence_penalty * 2) + decline_factor)
        
        # Creatinine drifts up with poor health
        if systolic_bp > 140 or glucose > 180:
            creatinine_drift += 0.001
        creatinine = round(params['creatinine_base'] + np.random.normal(0, 0.05) + creatinine_drift, 2)

        # --- Calculate Deterioration Probability ---
        prob = params['deterioration_base_prob']
        if systolic_bp > 140: prob += 0.05
        if glucose > 180: prob += 0.05
        if spo2 < 92: prob += 0.05
        if static_info['smoker']: prob += 0.02
        if missed_meds_yesterday: prob += 0.1
        
        deterioration_event = 1 if np.random.random() < prob else 0
        
        # --- Record Daily Data ---
        all_patient_data.append({
            'patient_id': patient_id,
            'day': day + 1,
            'age': static_info['age'],
            'gender': static_info['gender'],
            'smoker': static_info['smoker'],
            'heart_rate': heart_rate,
            'bp_sys': systolic_bp,
            'bp_dia': diastolic_bp,
            'spo2': spo2,
            'glucose': glucose,
            'creatinine': creatinine,
            'adherence': med_adherence,
            'sleep_hours': sleep_hours,
            'exercise_mins': exercise_mins,
            'deterioration_event': deterioration_event # Temporary column
        })
        
        # Update state for the next day
        missed_meds_yesterday = 1 if med_adherence == 0 else 0

# --- Create DataFrame and Calculate Target Variable ---
df = pd.DataFrame(all_patient_data)

# Calculate 'deteriorated_90d'
# This looks forward 90 days to see if a deterioration event occurs.
# We calculate the max event in a rolling window on the reversed timeline for each patient.
df_sorted = df.sort_values(by=['patient_id', 'day'])
s = df_sorted.groupby('patient_id')['deterioration_event'].transform(
    lambda x: x.iloc[::-1].rolling(window=90, min_periods=1).max().iloc[::-1]
)
# The above gives the max event in the next 90 days *including the current day*.
# We shift it to represent the *next* 90 days, excluding the current one.
df['deteriorated_90d'] = s.groupby(df['patient_id']).shift(-1).fillna(0)

# --- Finalize DataFrame ---
# Drop the temporary event column
df = df.drop(columns=['deterioration_event'])

# Add BMI (as a function of archetype, for simplicity)
archetype_bmi_map = {'stable': 24, 'unstable': 32, 'declining': 28}
df['archetype'] = df['patient_id'].map({pid: info['archetype'] for pid, info in patient_static_features.items()})
df['bmi'] = df['archetype'].map(archetype_bmi_map) + np.random.normal(0, 1, size=len(df))
df['bmi'] = df['bmi'].round(1)
df = df.drop(columns=['archetype'])

# Reorder columns to match the user's request
final_columns = [
    'patient_id', 'day', 'age', 'gender', 'bmi', 'smoker', 'heart_rate',
    'bp_sys', 'bp_dia', 'spo2', 'glucose', 'creatinine', 'adherence',
    'sleep_hours', 'exercise_mins', 'deteriorated_90d'
]
df = df[final_columns]

# --- Save and Display ---
output_filename = 'dataset.csv'
df.to_csv(output_filename, index=False)

print("\n--- Data Generation Complete ---")
print(f"✅ Successfully generated dataset with {len(df)} records.")
print(f"Saved to '{output_filename}'")

# --- Display Sample Data ---
print("\n--- Sample of the Generated Data ---")
print(df.head())

print("\n--- Data Summary ---")
print(df.describe())

print("\n--- Deterioration Target Distribution ---")
print(df['deteriorated_90d'].value_counts(normalize=True))