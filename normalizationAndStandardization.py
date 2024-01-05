import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


np.random.seed(42)
num_students = 100
grades = {
    'Math': np.random.randint(60, 100, num_students),
    'English': np.random.randint(50, 95, num_students),
    'History': np.random.randint(40, 88, num_students)
}
df = pd.DataFrame(grades)

print("Original Grades:")
print(df.head())

# Normalizasyon (Min-Max)
min_max_scaler = MinMaxScaler()
normalized_grades = min_max_scaler.fit_transform(df)

normalized_df = pd.DataFrame(normalized_grades, columns=df.columns)
print("\nNormalized Grades (Min-Max):")
print(normalized_df.head())

# Standartizasyon (Z-score)
standard_scaler = StandardScaler()
standardized_grades = standard_scaler.fit_transform(df)

standardized_df = pd.DataFrame(standardized_grades, columns=df.columns)
print("\nStandardized Grades (Z-score):")
print(standardized_df.head())
