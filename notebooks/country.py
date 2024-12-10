from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pandas as pd
import numpy as np

# Load the dataset (replace with your file path)
file_path = '../data/country.csv'
data = pd.read_csv(file_path)

# Replace '#NULL!' with NaN for processing
data.replace("#NULL!", np.nan, inplace=True)

# Convert data to numeric where possible, excluding 'country'
columns_to_impute = ['pdi', 'idv', 'mas', 'uai', 'ltowvs', 'ivr']
data[columns_to_impute] = data[columns_to_impute].apply(pd.to_numeric, errors='coerce')

# Apply MICE for imputation
mice_imputer = IterativeImputer(random_state=0)
data[columns_to_impute] = mice_imputer.fit_transform(data[columns_to_impute])

# Output the imputed data
print(data.head())

# Save the imputed dataset (optional)
data.to_csv('../data/imputed_country.csv', index=False)


file_path = '../data/imputed_country.csv'
imputed_data = pd.read_csv(file_path)

import matplotlib.pyplot as plt
import seaborn as sns

for column in columns_to_impute:
    plt.figure(figsize=(8, 4))
    sns.histplot(data[column], kde=True, label='Imputed', color='blue')
    plt.title(f'Distribution of {column} (Imputed)')
    plt.legend()
    plt.show()


from sklearn.metrics import mean_squared_error

# Mask some values as missing for evaluation
mask = data[column].notnull()
original_values = data.loc[mask, column]
imputed_values = imputed_data.loc[mask, column]

mse = mean_squared_error(original_values, imputed_values)
print(f'Mean Squared Error for {column}: {mse}')



original_corr = data[columns_to_impute].corr()
imputed_corr = imputed_data[columns_to_impute].corr()

print("Original Correlations:\n", original_corr)
print("Imputed Correlations:\n", imputed_corr)


print(f"Convergence status: {mice_imputer.converged_}")

