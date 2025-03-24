from cmath import sqrt
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

file_path = './communities.data'
columns_path = './communities.names'

with open(columns_path, 'r') as f:
    column_lines = [line.strip() for line in f if not line.startswith('@') and line.strip()]

columns = column_lines[-128:]

# dod unikalnych jesli sa duplikaty
unique_columns = []
column_counts = {}
for col in columns:
    if col in column_counts:
        column_counts[col] += 1
        unique_columns.append(f"{col}_{column_counts[col]}")
    else:
        column_counts[col] = 1
        unique_columns.append(col)

# wczytywanie z unikalnymi nazw kol
data = pd.read_csv(file_path, header=None, names=unique_columns, na_values=['?'])

# usuwanie kolumn z brak wart
cleaned_data = data.dropna(axis=1)

# usuwanie kolumn nienumerycznych
cleaned_data = cleaned_data.select_dtypes(include=[np.number])

# target column
target_column = cleaned_data.columns[-1]  # ostatnia kolumna jako "ViolentCrimesPerPop"


# Definiowanie zmiennych
X = cleaned_data.drop([target_column], axis=1)
y = cleaned_data[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# normalizacja
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# liniowy bez regularyzacji
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)
linear_predictions = linear_model.predict(X_test_scaled)
linear_rmse = root_mean_squared_error(y_test, linear_predictions)

# Ridge z regularyzacja
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train)
ridge_predictions = ridge_model.predict(X_test_scaled)
ridge_rmse = root_mean_squared_error(y_test, ridge_predictions)

difference = linear_rmse - ridge_rmse

print("RMSE dla modelu liniowego bez regularyzacji:", linear_rmse)
print("RMSE dla modelu Ridge z regularyzacją:", ridge_rmse)
print(f"Różnica: {difference:.10f}")

