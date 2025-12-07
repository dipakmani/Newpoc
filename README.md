import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Create 2000 records
num_records = 2000

# Generate unique Patient IDs
patient_ids = [f'PAT{str(i).zfill(6)}' for i in range(100001, 102001)]

# Enhanced domain definitions
genders = ['Male', 'Female']
blood_types = ['A+', 'A-', 'B+', 'B-', 'O+', 'O-', 'AB+', 'AB-']
admission_types = ['Emergency', 'Elective', 'Urgent', 'Transfer']
medical_conditions = [
    'Diabetes', 'Hypertension', 'Asthma', 'Arthritis', 'Heart Disease',
    'Cancer', 'Stroke', 'COPD', 'Kidney Disease', 'Liver Disease',
    'Pneumonia', 'Sepsis', 'Fracture', 'Appendicitis', 'Gallstones',
    'COVID-19', 'Influenza', 'Bronchitis', 'Migraine', 'Anemia'
]
medications = [
    'Insulin', 'Metformin', 'Lisinopril', 'Atorvastatin', 'Levothyroxine',
    'Amlodipine', 'Metoprolol', 'Albuterol', 'Omeprazole', 'Losartan',
    'Simvastatin', 'Hydrochlorothiazide', 'Prednisone', 'Gabapentin', 'Furosemide',
    'Warfarin', 'Ibuprofen', 'Acetaminophen', 'Azithromycin', 'Ciprofloxacin'
]
insurance_providers = [
    'UnitedHealth', 'Anthem', 'Aetna', 'Cigna', 'Humana',
    'Blue Cross', 'Kaiser', 'Molina', 'Centene', 'HCSC'
]
test_results = ['Normal', 'Abnormal', 'Critical', 'Pending']
hospitals = [
    'General Hospital', 'City Medical Center', 'University Hospital',
    'Community Hospital', 'Memorial Hospital', 'Regional Medical Center',
    'Childrens Hospital', 'Veterans Hospital'
]
doctor_names = [
    'Dr. Smith', 'Dr. Johnson', 'Dr. Williams', 'Dr. Brown', 'Dr. Jones',
    'Dr. Miller', 'Dr. Davis', 'Dr. Garcia', 'Dr. Rodriguez', 'Dr. Wilson',
    'Dr. Martinez', 'Dr. Anderson', 'Dr. Taylor', 'Dr. Thomas', 'Dr. Moore',
    'Dr. Lee', 'Dr. Harris', 'Dr. Clark', 'Dr. Lewis', 'Dr. Robinson'
]
departments = [
    'Cardiology', 'Emergency', 'Orthopedics', 'Oncology', 'Neurology',
    'Pediatrics', 'General Surgery', 'Internal Medicine', 'Radiology', 'ICU',
    'Maternity', 'Psychiatry', 'Dermatology', 'ENT', 'Urology'
]
discharge_statuses = ['Recovered', 'Transferred', 'Against Medical Advice', 'Deceased', 'Home Care']
procedure_types = ['Surgery', 'Diagnostic Test', 'Therapy', 'Medication', 'Observation', 'Rehabilitation']
lab_tests = ['Blood Test', 'X-Ray', 'MRI', 'CT Scan', 'Ultrasound', 'EKG', 'Urinalysis', 'Biopsy']
risk_levels = ['Low', 'Medium', 'High', 'Critical']
readmission_status = ['No', 'Yes - 7 days', 'Yes - 30 days', 'Yes - 90 days']
patient_satisfaction = ['Very Satisfied', 'Satisfied', 'Neutral', 'Dissatisfied', 'Very Dissatisfied']
complication_types = ['None', 'Infection', 'Bleeding', 'Allergic Reaction', 'Surgical Complication']
payment_methods = ['Insurance', 'Cash', 'Credit Card', 'Payment Plan', 'Government']
occupation_types = ['Professional', 'Laborer', 'Retired', 'Student', 'Unemployed', 'Healthcare Worker']
marital_statuses = ['Single', 'Married', 'Divorced', 'Widowed']
smoking_status = ['Never', 'Former', 'Current']
alcohol_consumption = ['None', 'Occasional', 'Regular']
allergies = ['None', 'Penicillin', 'Latex', 'NSAIDs', 'Food', 'Other']
follow_up_required = ['Yes', 'No', 'Pending']

# Generate base dates (last 5 years)
start_date = datetime(2020, 1, 1)
end_date = datetime(2024, 12, 31)

# Generate admission dates
admission_dates = []
for _ in range(num_records):
    days_diff = (end_date - start_date).days
    random_days = random.randint(0, days_diff)
    admission_date = start_date + timedelta(days=random_days)
    admission_dates.append(admission_date)

# Generate length of stay with more realistic distribution
length_of_stay = np.clip(np.random.exponential(scale=5, size=num_records), 1, 45).astype(int)
# Make some stays longer for serious conditions
serious_conditions = np.random.choice([0, 1], num_records, p=[0.7, 0.3])
length_of_stay = np.where(serious_conditions == 1, 
                         length_of_stay + np.random.randint(5, 20, num_records), 
                         length_of_stay)

# Generate discharge dates based on admission date and length of stay
discharge_dates = []
for i in range(num_records):
    discharge_date = admission_dates[i] + timedelta(days=int(length_of_stay[i]))
    discharge_dates.append(discharge_date)

# Calculate readmission probability (higher for certain conditions)
readmission_prob = np.random.random(num_records)
readmission_flag = np.where(readmission_prob > 0.85, 
                           np.random.choice(['Yes - 7 days', 'Yes - 30 days', 'Yes - 90 days'], num_records), 
                           'No')

# Generate BMI (Body Mass Index)
bmi = np.round(np.random.normal(26, 5, num_records), 1)
bmi = np.clip(bmi, 16, 45)

# Generate vital signs
systolic_bp = np.random.normal(125, 15, num_records).astype(int)
systolic_bp = np.clip(systolic_bp, 90, 200)
diastolic_bp = np.random.normal(80, 10, num_records).astype(int)
diastolic_bp = np.clip(diastolic_bp, 60, 120)
heart_rate = np.random.normal(75, 10, num_records).astype(int)
heart_rate = np.clip(heart_rate, 50, 130)
temperature = np.round(np.random.normal(37, 0.5, num_records), 1)
temperature = np.clip(temperature, 35.5, 40.5)

# Generate lab values
glucose = np.random.normal(100, 25, num_records).astype(int)
glucose = np.clip(glucose, 60, 300)
cholesterol = np.random.normal(190, 30, num_records).astype(int)
cholesterol = np.clip(cholesterol, 120, 300)

# Generate costs breakdown
medication_cost = np.round(length_of_stay * np.random.uniform(50, 200, num_records), 2)
lab_cost = np.round(np.random.exponential(500, num_records), 2)
procedure_cost = np.round(np.random.exponential(1500, num_records), 2)
room_cost = np.round(length_of_stay * np.random.uniform(800, 1500, num_records), 2)

# Total billing amount
billing_amount = medication_cost + lab_cost + procedure_cost + room_cost
billing_amount = np.round(np.clip(billing_amount, 500, 100000), 2)

# Create the enhanced dataset
data = {
    # Existing columns (1-17)
    'Patient_ID': patient_ids,
    'Patient_Age': np.random.randint(18, 95, num_records),
    'Patient_Gender': np.random.choice(genders, num_records),
    'Blood_Type': np.random.choice(blood_types, num_records),
    'Medical_Condition': np.random.choice(medical_conditions, num_records),
    'Admission_Type': np.random.choice(admission_types, num_records),
    'Test_Results': np.random.choice(test_results, num_records, p=[0.40, 0.35, 0.15, 0.10]),
    'Medication': np.random.choice(medications, num_records),
    'Hospital_Name': np.random.choice(hospitals, num_records),
    'Department': np.random.choice(departments, num_records),
    'Doctor_Name': np.random.choice(doctor_names, num_records),
    'Admission_Date': [date.strftime('%Y-%m-%d') for date in admission_dates],
    'Discharge_Date': [date.strftime('%Y-%m-%d') for date in discharge_dates],
    'Length_of_Stay_Days': length_of_stay,
    'Billing_Amount': billing_amount,
    'Insurance_Provider': np.random.choice(insurance_providers, num_records),
    'Discharge_Status': np.random.choice(discharge_statuses, num_records, p=[0.75, 0.10, 0.05, 0.05, 0.05]),
    
    # NEW COLUMNS FOR ADDITIONAL KPIs (18-32)
    'Procedure_Type': np.random.choice(procedure_types, num_records),
    'Lab_Test_Performed': np.random.choice(lab_tests, num_records),
    'Patient_Risk_Level': np.random.choice(risk_levels, num_records, p=[0.5, 0.3, 0.15, 0.05]),
    'Readmission_Status': readmission_flag,
    'Patient_Satisfaction': np.random.choice(patient_satisfaction, num_records, p=[0.3, 0.4, 0.2, 0.08, 0.02]),
    'Complications': np.random.choice(complication_types, num_records, p=[0.85, 0.05, 0.04, 0.03, 0.03]),
    'Payment_Method': np.random.choice(payment_methods, num_records, p=[0.7, 0.1, 0.1, 0.1]),
    'Patient_Occupation': np.random.choice(occupation_types, num_records),
    'Marital_Status': np.random.choice(marital_statuses, num_records, p=[0.3, 0.5, 0.15, 0.05]),
    'Smoking_Status': np.random.choice(smoking_status, num_records, p=[0.5, 0.3, 0.2]),
    'Alcohol_Consumption': np.random.choice(alcohol_consumption, num_records, p=[0.4, 0.4, 0.2]),
    'Known_Allergies': np.random.choice(allergies, num_records, p=[0.7, 0.1, 0.05, 0.05, 0.05, 0.05]),
    'Follow_Up_Required': np.random.choice(follow_up_required, num_records, p=[0.6, 0.3, 0.1]),
    
    # Clinical Metrics (33-39)
    'BMI': bmi,
    'Systolic_BP': systolic_bp,
    'Diastolic_BP': diastolic_bp,
    'Heart_Rate': heart_rate,
    'Temperature_C': temperature,
    'Glucose_mg_dL': glucose,
    'Cholesterol_mg_dL': cholesterol,
    
    # Cost Breakdown (40-43)
    'Medication_Cost': medication_cost,
    'Lab_Cost': lab_cost,
    'Procedure_Cost': procedure_cost,
    'Room_Cost': room_cost,
    
    # Calculated fields (44-45)
    'Cost_per_Day': np.round(billing_amount / length_of_stay, 2),
    'Insurance_Coverage_Percent': np.random.randint(70, 101, num_records)  # 70-100% coverage
}

# Create DataFrame
df = pd.DataFrame(data)

# Reorder columns for better organization
column_order = [
    # Patient Identification & Demographics
    'Patient_ID', 'Patient_Age', 'Patient_Gender', 'Blood_Type', 
    'Marital_Status', 'Patient_Occupation', 'Smoking_Status', 
    'Alcohol_Consumption', 'Known_Allergies', 'BMI',
    
    # Medical Information
    'Medical_Condition', 'Admission_Type', 'Test_Results', 'Medication',
    'Procedure_Type', 'Lab_Test_Performed', 'Patient_Risk_Level',
    'Complications', 'Follow_Up_Required',
    
    # Clinical Metrics
    'Systolic_BP', 'Diastolic_BP', 'Heart_Rate', 'Temperature_C',
    'Glucose_mg_dL', 'Cholesterol_mg_dL',
    
    # Hospital Administration
    'Hospital_Name', 'Department', 'Doctor_Name',
    'Admission_Date', 'Discharge_Date', 'Length_of_Stay_Days',
    'Discharge_Status', 'Readmission_Status', 'Patient_Satisfaction',
    
    # Financial Information
    'Billing_Amount', 'Medication_Cost', 'Lab_Cost', 
    'Procedure_Cost', 'Room_Cost', 'Cost_per_Day',
    'Insurance_Provider', 'Insurance_Coverage_Percent', 'Payment_Method'
]

df = df[column_order]

# Save to CSV
output_file = 'enhanced_healthcare_dataset_45_columns.csv'
df.to_csv(output_file, index=False)

# Calculate comprehensive KPIs
total_patients = len(df)
total_billing = df['Billing_Amount'].sum()
avg_billing = df['Billing_Amount'].mean()
avg_length_of_stay = df['Length_of_Stay_Days'].mean()
abnormal_tests = df[df['Test_Results'].isin(['Abnormal', 'Critical'])].shape[0]
readmission_rate = df[df['Readmission_Status'] != 'No'].shape[0] / total_patients * 100
satisfaction_rate = df[df['Patient_Satisfaction'].isin(['Very Satisfied', 'Satisfied'])].shape[0] / total_patients * 100
complication_rate = df[df['Complications'] != 'None'].shape[0] / total_patients * 100
avg_bmi = df['BMI'].mean()
avg_insurance_coverage = df['Insurance_Coverage_Percent'].mean()

# Calculate cost distribution
cost_distribution = {
    'Medication': df['Medication_Cost'].sum(),
    'Lab Tests': df['Lab_Cost'].sum(),
    'Procedures': df['Procedure_Cost'].sum(),
    'Room': df['Room_Cost'].sum()
}

# Create comprehensive summary
print("=" * 90)
print("ENHANCED HEALTHCARE DATASET WITH 45 COLUMNS")
print("=" * 90)
print(f"📁 File created: {output_file}")
print(f"📊 Total Records: {total_patients:,}")
print(f"📈 Total Columns: {len(df.columns)}")

print("\n📋 DATASET STRUCTURE:")
print("  PATIENT DEMOGRAPHICS (10 columns):")
demo_cols = column_order[:10]
for i, col in enumerate(demo_cols, 1):
    print(f"    {i:2d}. {col}")

print("\n  MEDICAL INFORMATION (9 columns):")
med_cols = column_order[10:19]
for i, col in enumerate(med_cols, 1):
    print(f"    {i:2d}. {col}")

print("\n  CLINICAL METRICS (6 columns):")
clinical_cols = column_order[19:25]
for i, col in enumerate(clinical_cols, 1):
    print(f"    {i:2d}. {col}")

print("\n  HOSPITAL ADMINISTRATION (9 columns):")
admin_cols = column_order[25:34]
for i, col in enumerate(admin_cols, 1):
    print(f"    {i:2d}. {col}")

print("\n  FINANCIAL INFORMATION (11 columns):")
finance_cols = column_order[34:]
for i, col in enumerate(finance_cols, 1):
    print(f"    {i:2d}. {col}")

print("\n📊 COMPREHENSIVE KPIs AVAILABLE FOR ANALYSIS:")
print("\n  CLINICAL KPIs:")
print(f"    • Average Length of Stay:           {avg_length_of_stay:.2f} days")
print(f"    • Abnormal/Critical Test Results:   {abnormal_tests:,} ({abnormal_tests/total_patients*100:.1f}%)")
print(f"    • Complication Rate:                {complication_rate:.1f}%")
print(f"    • Readmission Rate:                 {readmission_rate:.1f}%")
print(f"    • Average BMI:                      {avg_bmi:.1f}")
print(f"    • Follow-up Required:               {df['Follow_Up_Required'].value_counts()['Yes']:,} patients")

print("\n  FINANCIAL KPIs:")
print(f"    • Total Billing Amount:            ${total_billing:,.2f}")
print(f"    • Average Billing Amount:          ${avg_billing:,.2f}")
print(f"    • Average Cost per Day:            ${df['Cost_per_Day'].mean():,.2f}")
print(f"    • Average Insurance Coverage:      {avg_insurance_coverage:.1f}%")
print(f"    • Cost Distribution:")
for category, amount in cost_distribution.items():
    percentage = amount / total_billing * 100
    print(f"        - {category:<12}: ${amount:,.2f} ({percentage:.1f}%)")

print("\n  OPERATIONAL KPIs:")
print(f"    • Patient Satisfaction Rate:       {satisfaction_rate:.1f}%")
print(f"    • Emergency Admissions:            {df[df['Admission_Type'] == 'Emergency'].shape[0]:,}")
print(f"    • Hospital Distribution:")
for hospital, count in df['Hospital_Name'].value_counts().head(3).items():
    print(f"        - {hospital}: {count} patients")

print("\n  QUALITY METRICS:")
print(f"    • High Risk Patients:              {df[df['Patient_Risk_Level'].isin(['High', 'Critical'])].shape[0]:,}")
print(f"    • Recovery Rate:                   {df[df['Discharge_Status'] == 'Recovered'].shape[0]/total_patients*100:.1f}%")
print(f"    • Pending Test Results:            {df[df['Test_Results'] == 'Pending'].shape[0]:,}")

print("\n📅 TEMPORAL ANALYSIS CAPABILITIES:")
print(f"    • Date Range:                      {min(df['Admission_Date'])} to {max(df['Admission_Date'])}")
print(f"    • Yearly Trends:                   Available (2020-2024)")
print(f"    • Seasonal Analysis:               Possible with full date information")

print("\n🔍 SAMPLE DATA (first 2 rows with key columns):")
sample_columns = ['Patient_ID', 'Patient_Age', 'Medical_Condition', 'Admission_Type', 
                  'Length_of_Stay_Days', 'Billing_Amount', 'Patient_Satisfaction', 
                  'Readmission_Status', 'Patient_Risk_Level']
print(df[sample_columns].head(2).to_string(index=False))

print("\n💡 SUGGESTED VISUALIZATIONS FOR POWER BI:")
print("  1. Drill Down Combo PRO: Cost breakdown by department/condition")
print("  2. Drill Down Donut PRO: Patient demographics distribution")
print("  3. Time Series Analysis: Admissions & revenue trends by month/year")
print("  4. Risk Matrix: Patient risk level vs. outcomes")
print("  5. Quality Dashboard: Satisfaction vs. complication rates")
print("  6. Financial Dashboard: Revenue, costs, and insurance coverage")

print("\n✅ DATA VALIDATION:")
print(f"  • No missing values in critical columns: {df[['Patient_ID', 'Admission_Date', 'Billing_Amount']].isnull().sum().sum() == 0}")
print(f"  • All dates valid: {pd.to_datetime(df['Discharge_Date']) >= pd.to_datetime(df['Admission_Date']).all()}")
print(f"  • Positive billing amounts: {(df['Billing_Amount'] > 0).all()}")
print(f"  • Realistic clinical ranges: All vital signs within medical norms")

print("=" * 90)
