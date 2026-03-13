import pandas as pd
import numpy as np

# Create a synthetic PCOS dataset
np.random.seed(42)

n_samples = 1000

# Generate synthetic data
data = {
    'Age': np.random.normal(28, 5, n_samples).clip(18, 45),
    'BMI': np.random.normal(25, 4, n_samples).clip(18, 40),
    'Cycle_Length': np.random.normal(28, 7, n_samples).clip(21, 45),
    'Acne': np.random.choice(['No', 'Yes'], n_samples, p=[0.6, 0.4]),
    'Hair_Growth': np.random.choice(['No', 'Yes'], n_samples, p=[0.7, 0.3]),
    'Insulin': np.random.normal(10, 3, n_samples).clip(3, 25),
    'LH_FSH_Ratio': np.random.normal(1.5, 0.5, n_samples).clip(0.5, 3.0),
    'Weight_Gain': np.random.choice(['No', 'Yes'], n_samples, p=[0.5, 0.5]),
    'Irregular_Periods': np.random.choice(['No', 'Yes'], n_samples, p=[0.4, 0.6])
}

df = pd.DataFrame(data)

# Create PCOS target based on risk factors
def calculate_pcos_risk(row):
    risk_score = 0
    
    # Age factor
    if row['Age'] > 30:
        risk_score += 1
    
    # BMI factor
    if row['BMI'] > 25:
        risk_score += 2
    
    # Cycle length
    if row['Cycle_Length'] > 35 or row['Cycle_Length'] < 21:
        risk_score += 2
    
    # Symptoms
    if row['Acne'] == 'Yes':
        risk_score += 1
    if row['Hair_Growth'] == 'Yes':
        risk_score += 1
    if row['Weight_Gain'] == 'Yes':
        risk_score += 1
    if row['Irregular_Periods'] == 'Yes':
        risk_score += 2
    
    # Insulin resistance
    if row['Insulin'] > 12:
        risk_score += 1
    
    # LH/FSH ratio
    if row['LH_FSH_Ratio'] > 2.0:
        risk_score += 1
    
    # Determine PCOS (simplified threshold)
    return 1 if risk_score >= 5 else 0

df['PCOS'] = df.apply(calculate_pcos_risk, axis=1)

# Add some noise to make it more realistic
noise_indices = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
df.loc[noise_indices, 'PCOS'] = 1 - df.loc[noise_indices, 'PCOS']

# Save to CSV
df.to_csv('data/pcos_dataset.csv', index=False)

print("Synthetic PCOS dataset created with", len(df), "samples")
print("PCOS positive cases:", df['PCOS'].sum())
print("PCOS negative cases:", len(df) - df['PCOS'].sum())