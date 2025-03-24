import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('vehicles/vehicles.csv')

data = data.iloc[28:]

data = data[['price', 'manufacturer', 'model', 'year', 'region', 'condition']]

data.dropna(inplace=True)

data = data[(data['price'] > 500) & (data['price'] < 100000)]

from sklearn.preprocessing import LabelEncoder, StandardScaler
label_encoders = {}
for column in ['manufacturer', 'model', 'region', 'condition']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

X = data[['manufacturer', 'model', 'year', 'region', 'condition']]
y = data['price']

original_X_test = None

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
original_X_test = X_test.copy()

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

results = original_X_test.copy()
results['actual_price'] = y_test.values
results['predicted_price'] = y_pred

for column in ['manufacturer', 'model', 'region', 'condition']:
    results[column] = label_encoders[column].inverse_transform(results[column].astype(int))

print("Sample of predictions:")
print(results[['manufacturer', 'model', 'year', 'region', 'condition', 'actual_price', 'predicted_price']].head(10))

import os
os.makedirs('vehicles', exist_ok=True)
joblib.dump(model, 'vehicles/car_price_model.pkl')
joblib.dump(scaler, 'vehicles/scaler.pkl')
for column, le in label_encoders.items():
    joblib.dump(le, f'vehicles/label_encoder_{column}.pkl')

data.to_csv('vehicles/cleaned_craigslist_vehicles.csv', index=False)
